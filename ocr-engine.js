/* ============================================================================
 * PPOCR — PaddleOCR (PP-OCR) 纯浏览器推理引擎
 * 文本检测（DB 算法）+ 文字识别（CTC），onnxruntime-web / WebAssembly
 *
 * 预处理与后处理参数严格对齐 RapidOCR 的 ch_ppocr_det / ch_ppocr_rec，
 * 因此浏览器端结果可与本地 Python 版逐项对照（实测框位置误差 < 0.3%）。
 *
 * 相对 Python 参考实现的两处简化（对表格类文档无实质影响）：
 *   · 检测后处理用「连通域 + 轴对齐外接框 + 按面积/周长等比外扩」，
 *     替代 cv2.minAreaRect + pyclipper —— 省掉 opencv/shapely 依赖。
 *   · 省略 180° 方向分类模型（cls），表格与证明材料均为正向排版。
 * ==========================================================================*/
var PPOCR = (function () {
  'use strict';

  var DET = {
    LIMIT_SIDE: 736,   // limit_type = "min"：短边不足则放大到 736
    SCALE: 1 / 255,
    MEAN: 0.5,
    STD: 0.5,
    THRESH: 0.3,       // 概率图二值化阈值
    BOX_THRESH: 0.5,   // 文本框置信度阈值
    UNCLIP: 1.6,       // unclip_ratio
    MIN_SIZE: 3,
    MAX_CAND: 1000
  };

  /* BASE_W 是旧版的固定裁剪宽度：以前每批都按 320 宽算，等于白烧 2~3 倍算力，
   * 现已改为「按批内最宽框的自然比例」，所以这里只作为历史参数保留。 */
  var REC = { H: 48, BASE_W: 320, MIN_W: 96, PAD_W: 8, MAX_W: 1600, BATCH: 6 };

  var ort = null;
  var detSession = null;
  var recSession = null;
  var charset = null;
  var ready = false;

  /* ------------------------------------------------------------ 基础工具 */

  function newCanvas(w, h) {
    var c = document.createElement('canvas');
    c.width = Math.max(1, w | 0);
    c.height = Math.max(1, h | 0);
    return c;
  }

  function ctx2d(cv) {
    return cv.getContext('2d', { willReadFrequently: true });
  }

  function resizeCanvas(src, w, h) {
    var c = newCanvas(w, h);
    var g = ctx2d(c);
    g.imageSmoothingEnabled = true;
    g.imageSmoothingQuality = 'high';
    g.drawImage(src, 0, 0, c.width, c.height);
    return c;
  }

  function b64ToText(b64) {
    var bin = atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (typeof TextDecoder !== 'undefined') return new TextDecoder('utf-8').decode(bytes);
    var out = '', j = 0;
    while (j < bytes.length) {
      var c = bytes[j++];
      if (c < 0x80) out += String.fromCharCode(c);
      else if (c < 0xE0) out += String.fromCharCode(((c & 0x1F) << 6) | (bytes[j++] & 0x3F));
      else if (c < 0xF0) out += String.fromCharCode(((c & 0x0F) << 12) | ((bytes[j++] & 0x3F) << 6) | (bytes[j++] & 0x3F));
      else {
        var cp = ((c & 0x07) << 18) | ((bytes[j++] & 0x3F) << 12) | ((bytes[j++] & 0x3F) << 6) | (bytes[j++] & 0x3F);
        cp -= 0x10000;
        out += String.fromCharCode(0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
      }
    }
    return out;
  }

  function metadataOf(sess) {
    var out = {};
    try {
      var md = sess.modelMetadata;
      if (!md) return out;
      var cm = md.customMetadata;
      if (!cm) return out;
      if (typeof cm.forEach === 'function' && typeof cm.get === 'function') {
        cm.forEach(function (v, k) { out[k] = v; });
      } else { out = cm; }
    } catch (e) { /* 某些版本不暴露元数据 */ }
    return out;
  }

  function yield0() { return new Promise(function (r) { setTimeout(r, 0); }); }

  /* ------------------------------------------------------------ 生命周期 */

  function setOrt(o) { ort = o; }

  function resolveCharset(sess, b64) {
    var raw = null;
    var md = metadataOf(sess);
    if (md.character) raw = md.character;
    if (!raw && md.dict) raw = md.dict;
    if (!raw && b64) raw = b64ToText(b64);
    if (!raw) throw new Error('取不到识别模型字符表（模型未内嵌 character，也未提供 charset）');
    var lines = String(raw).split('\n');
    while (lines.length && lines[lines.length - 1] === '') lines.pop();
    // RapidOCR 规则：末尾补空格、头部补 blank（CTC 空白符）
    return { charset: ['blank'].concat(lines, [' ']), count: lines.length };
  }

  async function initModels(detBuf, recBuf, opts) {
    opts = opts || {};
    ort.env.wasm.numThreads = opts.threads || 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
    var eps = opts.executionProviders || ['wasm'];

    detSession = await ort.InferenceSession.create(detBuf, { executionProviders: eps, graphOptimizationLevel: 'all' });
    recSession = await ort.InferenceSession.create(recBuf, { executionProviders: eps, graphOptimizationLevel: 'all' });

    var cs = resolveCharset(recSession, opts.charsetB64);
    charset = cs.charset;
    ready = true;
    return { classes: charset.length, dictChars: cs.count };
  }

  function isReady() { return ready; }

  /* -------------------------------------------------------- 检测 前 / 后 */

  function detPreprocess(srcCanvas, maxSide) {
    var baseW = srcCanvas.width, baseH = srcCanvas.height;
    var w = baseW, h = baseH;
    var src = srcCanvas;
    var capped = false;
    if (maxSide && Math.max(w, h) > maxSide) {
      var k = maxSide / Math.max(w, h);
      src = resizeCanvas(srcCanvas, Math.round(w * k), Math.round(h * k));
      w = src.width; h = src.height;
      capped = true;
    }
    /* 短边不足 LIMIT_SIDE 时放大补足；但若调用方已主动限过最长边，
     * 就不再反着放大——否则「降到 560 短边」这类更省的请求会被弹回 736。 */
    var ratio = 1;
    if (!capped) {
      var shortSide = Math.min(h, w);
      if (shortSide < DET.LIMIT_SIDE) ratio = DET.LIMIT_SIDE / shortSide;
    }
    var rh = Math.max(32, Math.round(Math.round(h * ratio) / 32) * 32);
    var rw = Math.max(32, Math.round(Math.round(w * ratio) / 32) * 32);

    var small = resizeCanvas(src, rw, rh);
    var px = ctx2d(small).getImageData(0, 0, rw, rh).data;

    var n = rw * rh;
    var data = new Float32Array(3 * n);
    var s = DET.SCALE, m = DET.MEAN, d = DET.STD;
    for (var i = 0, p = 0; i < n; i++, p += 4) {
      data[i]         = (px[p]     * s - m) / d;
      data[n + i]     = (px[p + 1] * s - m) / d;
      data[2 * n + i] = (px[p + 2] * s - m) / d;
    }
    /* ow/oh 始终是**原画布**尺寸：识别裁剪用的是原画布，
     * 因此检测框必须映射回原画布坐标（否则限制最长边后框会整体偏小）。 */
    return { data: data, rw: rw, rh: rh, ow: baseW, oh: baseH, detW: w, detH: h, scaled: capped };
  }

  /* 简化版 DB 后处理：连通域 → 轴对齐外接框 → 等距外扩。
     表格/证明材料文字均为水平排布，与 minAreaRect + pyclipper 的结果高度一致，
     但无需 opencv / shapely / pyclipper。 */
  function detPostprocess(prob, rw, rh, ow, oh) {
    var n = rw * rh, i;
    var bin = new Uint8Array(n);
    for (i = 0; i < n; i++) bin[i] = prob[i] > DET.THRESH ? 1 : 0;

    // cv2.dilate，2x2 核
    var dil = new Uint8Array(n);
    for (var y = 0; y < rh; y++) {
      for (var x = 0; x < rw; x++) {
        var v = 0;
        for (var dy = -1; dy <= 0 && !v; dy++) {
          var yy = y + dy;
          if (yy < 0) continue;
          for (var dx = -1; dx <= 0; dx++) {
            var xx = x + dx;
            if (xx < 0) continue;
            if (bin[yy * rw + xx]) { v = 1; break; }
          }
        }
        dil[y * rw + x] = v;
      }
    }

    // 8 连通标记 + 统计
    var label = new Int32Array(n);
    var stack = new Int32Array(n);
    var boxes = [], tights = [], scores = [], lab = 0;

    for (var start = 0; start < n; start++) {
      if (!dil[start] || label[start]) continue;
      lab++;
      var sp = 0;
      stack[sp++] = start;
      label[start] = lab;
      var minx = rw, maxx = -1, miny = rh, maxy = -1, cnt = 0;
      while (sp > 0) {
        var cur = stack[--sp];
        var cy = (cur / rw) | 0, cx = cur - cy * rw;
        if (cx < minx) minx = cx;
        if (cx > maxx) maxx = cx;
        if (cy < miny) miny = cy;
        if (cy > maxy) maxy = cy;
        cnt++;
        for (var oy = -1; oy <= 1; oy++) {
          var ny = cy + oy;
          if (ny < 0 || ny >= rh) continue;
          for (var ox = -1; ox <= 1; ox++) {
            var nx = cx + ox;
            if (nx < 0 || nx >= rw) continue;
            var idx = ny * rw + nx;
            if (dil[idx] && !label[idx]) { label[idx] = lab; stack[sp++] = idx; }
          }
        }
      }
      if (cnt < 4) continue;

      var bw = maxx - minx + 1, bh = maxy - miny + 1;
      if (Math.min(bw, bh) < DET.MIN_SIZE) continue;

      // box_score_fast：外接矩形内的概率均值
      var sum = 0, num = 0;
      for (var yy2 = miny; yy2 <= maxy; yy2++) {
        var row = yy2 * rw;
        for (var xx2 = minx; xx2 <= maxx; xx2++) { sum += prob[row + xx2]; num++; }
      }
      var score = num ? sum / num : 0;
      if (score < DET.BOX_THRESH) continue;

      // unclip：按 面积×ratio/周长 等比外扩
      var perim = 2 * (bw + bh);
      var dist = perim > 0 ? (bw * bh) * DET.UNCLIP / perim : 0;
      var x0 = minx - dist, y0 = miny - dist, x1 = maxx + dist, y1 = maxy + dist;
      if (Math.min(x1 - x0, y1 - y0) < DET.MIN_SIZE + 2) continue;

      var sx = ow / rw, sy = oh / rh;
      /* 紧贴框（外扩前）：宽高比 ≈ 真实字数。
       * 外扩量 = 面积×1.6/周长，对小框的高度放大量远大于宽度，
       * 会把宽高比整体压向 2~5 —— 于是「7 个字的加分项目」估出来只有 4，
       * 用外扩框估字数几乎分不出「姓名」和「一整行内容」。紧贴框才能。
       * 识别裁剪仍用外扩框（照旧，留白对识别有利）。 */
      var tx0 = Math.max(0, Math.round(minx * sx)), ty0 = Math.max(0, Math.round(miny * sy));
      var tx1 = Math.min(ow, Math.round((maxx + 1) * sx)), ty1 = Math.min(oh, Math.round((maxy + 1) * sy));

      x0 = Math.max(0, Math.round(x0 * sx));
      y0 = Math.max(0, Math.round(y0 * sy));
      x1 = Math.min(ow, Math.round(x1 * sx));
      y1 = Math.min(oh, Math.round(y1 * sy));
      if (x1 - x0 <= 3 || y1 - y0 <= 3) continue;

      boxes.push([x0, y0, x1, y1]);
      tights.push([tx0, ty0, tx1, ty1]);
      scores.push(score);
      if (boxes.length >= DET.MAX_CAND) break;
    }
    return { boxes: boxes, tight: tights, scores: scores };
  }

  /* -------------------------------------------------------- 识别 前 / 后 */

  /* 一个批次内所有文本框共用同一宽度裁剪尺寸，一次前向搞定，省掉逐框调用开销。
   *
   * 裁剪宽度按**批内最宽框的自然比例**取，不再固定 320：
   * 缩放到高 48 后宽度 = 48×(宽/高)，不足 imgW 的部分右侧补灰（归一化后为 0），
   * 这正是 PaddleOCR 的做法 —— 所以只要 imgW ≥ 自然宽，识别结果**逐位相同**，
   * 而张量宽度可以从 320 降到 96~200，识别耗时按宽度线性下降（识别是按批算的，
   * 之前无论框多窄都按 320 算，等于白烧 2~3 倍算力）。 */
  function recPrepareBatch(srcCanvas, boxes, idxs) {
    var n = idxs.length;
    var maxWhRatio = 0;
    var k, box, w, h, ratio;
    for (k = 0; k < n; k++) {
      box = boxes[idxs[k]];
      w = box[2] - box[0]; h = box[3] - box[1];
      if (w <= 0 || h <= 0) continue;
      ratio = w / h;
      if (ratio > maxWhRatio) maxWhRatio = ratio;
    }
    var natural = Math.ceil(REC.H * maxWhRatio) + REC.PAD_W;
    var imgW = Math.min(REC.MAX_W, Math.max(REC.MIN_W, natural));
    var plane = REC.H * imgW;
    var data = new Float32Array(n * 3 * plane);

    for (k = 0; k < n; k++) {
      box = boxes[idxs[k]];
      w = box[2] - box[0]; h = box[3] - box[1];
      if (w <= 0 || h <= 0) continue;
      var crop = newCanvas(w, h);
      ctx2d(crop).drawImage(srcCanvas, box[0], box[1], w, h, 0, 0, w, h);

      var resizedW = Math.max(1, Math.min(Math.ceil(REC.H * (w / h)), imgW));
      var rs = resizeCanvas(crop, resizedW, REC.H);
      var px = ctx2d(rs).getImageData(0, 0, resizedW, REC.H).data;

      var base = k * 3 * plane;
      for (var yy = 0; yy < REC.H; yy++) {
        var srcRow = yy * resizedW * 4;
        var dstRow = yy * imgW;
        for (var xx = 0; xx < resizedW; xx++) {
          var p = srcRow + xx * 4;
          var q = dstRow + xx;
          data[base + q]               = (px[p]     / 255 - 0.5) / 0.5;
          data[base + plane + q]       = (px[p + 1] / 255 - 0.5) / 0.5;
          data[base + 2 * plane + q]   = (px[p + 2] / 255 - 0.5) / 0.5;
        }
      }
    }
    return { data: data, dims: [n, 3, REC.H, imgW] };
  }

  function ctcDecodeBatch(tensor, n) {
    var dims = tensor.dims;          // [n, T, C]
    var T = dims[1], C = dims[2];
    var out = tensor.data;
    var res = [];
    for (var b = 0; b < n; b++) {
      var base = b * T * C;
      var text = '', conf = [], prev = -1;
      for (var t = 0; t < T; t++) {
        var off = base + t * C;
        var best = 0, bestI = 0;
        for (var c = 0; c < C; c++) {
          var v = out[off + c];
          if (v > best) { best = v; bestI = c; }
        }
        if (bestI !== 0 && bestI !== prev) {
          var ch = charset[bestI];
          if (ch !== undefined) { text += ch; conf.push(best); }
        }
        prev = bestI;
      }
      var mean = 0;
      for (var i = 0; i < conf.length; i++) mean += conf[i];
      res.push({ text: text, score: conf.length ? mean / conf.length : 0 });
    }
    return res;
  }

  /* --------------------------------------------------------------- 主入口 */

  /** 只做检测：返回文本框（不做识别） */
  async function detect(srcCanvas, opts) {
    opts = opts || {};
    if (!ready) throw new Error('OCR 引擎尚未初始化');
    var t0 = now();
    var pre = detPreprocess(srcCanvas, opts.detMaxSide);
    var feeds = {};
    feeds[detSession.inputNames[0]] = new ort.Tensor('float32', pre.data, [1, 3, pre.rh, pre.rw]);
    var detOut = await detSession.run(feeds);
    var probTensor = detOut[detSession.outputNames[0]] || detOut[Object.keys(detOut)[0]];
    var d = detPostprocess(probTensor.data, pre.rw, pre.rh, pre.ow, pre.oh);
    return { boxes: d.boxes, tight: d.tight, scores: d.scores, ms: now() - t0 };
  }

  /**
   * 按给定顺序识别检测框。
   * @param order 框索引的识别顺序（null = 检测原始顺序）；配合 onBatch 早停可少认很多框
   */
  async function recognize(srcCanvas, boxes, order, opts) {
    opts = opts || {};
    var idxAll = [];
    var i;
    for (i = 0; i < boxes.length; i++) idxAll.push(i);
    if (order && order.length) idxAll = order.slice();

    var batchSize = Math.max(1, opts.batchSize || REC.BATCH);
    var items = [], stoppedEarly = false, t0 = now();

    for (var s = 0; s < idxAll.length; s += batchSize) {
      var idxs = [];
      for (var q = s; q < Math.min(s + batchSize, idxAll.length); q++) idxs.push(idxAll[q]);
      var prep = recPrepareBatch(srcCanvas, boxes, idxs);
      var f = {};
      f[recSession.inputNames[0]] = new ort.Tensor('float32', prep.data, prep.dims);
      var res = await recSession.run(f);
      var ot = res[recSession.outputNames[0]] || res[Object.keys(res)[0]];
      var dec = ctcDecodeBatch(ot, idxs.length);

      for (var k = 0; k < idxs.length; k++) {
        if (!dec[k].text) continue;
        var b = boxes[idxs[k]];
        items.push({ text: dec[k].text, score: dec[k].score, x0: b[0], y0: b[1], x1: b[2], y1: b[3] });
      }
      if (opts.onProgress) opts.onProgress(Math.min(s + batchSize, idxAll.length), idxAll.length);
      if (opts.onBatch && opts.onBatch(items) === true) { stoppedEarly = true; break; }
      await yield0();
    }
    return {
      items: items,
      recognized: Math.min(items.length ? idxAll.length : 0, idxAll.length),
      ms: { rec: now() - t0, boxes: boxes.length, stoppedEarly: stoppedEarly }
    };
  }

  /**
   * 对整页 canvas 做 OCR。
   * @param srcCanvas 页面画布（原图分辨率，用于裁剪识别）
   * @param opts {
   *   batchSize    识别批大小，默认 6
   *   detMaxSide   检测输入的最长边上限
   *   order        框的识别顺序（索引数组），不传则按检测顺序
   *   onBatch(items) 每识别完一批回调；返回 true 可提前停止（命中即停）
   *   onProgress(done,total)
   * }
   */
  async function runPage(srcCanvas, opts) {
    opts = opts || {};
    var t0 = now();
    var d = await detect(srcCanvas, opts);
    var r = await recognize(srcCanvas, d.boxes, opts.order || null, opts);
    return {
      items: r.items,
      ms: { det: d.ms, rec: r.ms.rec, total: now() - t0, boxes: d.boxes.length, stoppedEarly: r.ms.stoppedEarly },
      det: d
    };
  }

  /* ------------------------------------------------- 识别顺序（早停加速） */

  function medianHeights(boxes) {
    var hs = [];
    for (var i = 0; i < boxes.length; i++) hs.push(Math.max(1, boxes[i][3] - boxes[i][1]));
    hs.sort(function (a, b) { return a - b; });
    return hs.length ? hs[(hs.length / 2) | 0] : 12;
  }

  /** 取「用来估字数的框」：优先用紧贴框（外扩前的原框） */
  function shapeSource(boxes, tight) {
    return (tight && tight.length === boxes.length) ? tight : boxes;
  }

  /** 估算字数：紧贴框的 宽/高。汉字近似方形，所以这个比值 ≈ 字数，且与字号无关
   *  （用「宽 / 本页中位高」会被混排字号带偏，用「宽 / 自身高」才稳） */
  function estChars(src, i) {
    var b = src[i];
    return (b[2] - b[0]) / Math.max(1, b[3] - b[1]);
  }

  /**
   * 姓名形状优先，并让「同一批识别的框宽度接近」。
   *
   * 为什么按「字数分桶」排，而不是直接按 |宽 - 目标| 排：
   * recognize 一个批次共用同一裁剪宽度（取批内最大值），批内混进一个宽框整批都按它算。
   * 而 |1-3| == |5-3|，「1 个字」和「5 个字」会被排到相邻 —— 白多算 5 倍。
   * 分桶后批内宽度只差一档；桶与桶之间仍按「离姓名字数由近到远」，命中即停不受影响。
   *
   * @param subset 只对这些索引排序（不传 = 全部）
   * @param targetRatio 目标字数（= 姓名字数），默认 3
   * @param tight 紧贴框（可选；有它时用紧贴框估字数，更准）
   */
  function orderByNameShape(boxes, subset, targetRatio, tight) {
    var tgt = targetRatio || 3;
    var src = shapeSource(boxes, tight), i;
    var idx = subset ? subset.slice() : [];
    if (!subset) for (i = 0; i < boxes.length; i++) idx.push(i);

    var groups = {}, keys = [];
    for (i = 0; i < idx.length; i++) {
      var k = Math.max(1, Math.round(estChars(src, idx[i])));
      if (!groups[k]) { groups[k] = []; keys.push(k); }
      groups[k].push(idx[i]);
    }
    keys.sort(function (a, b) {
      var da = Math.abs(a - tgt), db = Math.abs(b - tgt);
      return da !== db ? da - db : a - b;
    });
    var out = [];
    for (i = 0; i < keys.length; i++) {
      var g = groups[keys[i]];
      g.sort(function (a, b) { return boxes[a][1] - boxes[b][1] || boxes[a][0] - boxes[b][0]; });
      for (var m = 0; m < g.length; m++) out.push(g[m]);
    }
    return out;
  }

  /**
   * 只挑「可能构成姓名命中」的框。
   *
   * 依据：findName 的三条命中规则都要求识别文本与姓名长度相近
   * （整框相等 / 只多 2 个字 / 相邻框拼接后只多 4 个字），
   * 所以**长行（整行标题、整段说明）永远不可能命中**。
   * 跳过它们不损失任何结果，却能省掉识别成本最高的宽框
   * —— 识别耗时 ≈ Σ(裁剪宽度)，一条 8 字宽的整行 ≈ 3 个姓名框。
   *
   * 判断必须用**紧贴框的宽高比**（≈ 真实字数），不能用外扩框：
   * 外扩量 = 面积×1.6/周长，对小框的高度放大量远大于宽度，
   * 会把宽高比整体压向 2~5 —— 「7 个字的加分项目」估出来只有 4，
   * 于是被当成候选框白认一遍（这正是之前慢的主因）。
   *
   * @param opts.maxChars 允许的最大估算字数（建议 = 姓名字数 + 2）
   * @param opts.minChars 允许的最小估算字数，默认 1
   * @param opts.tight    detect 返回的紧贴框（与外扩框同序）
   */
  function nameBoxIndices(boxes, opts) {
    opts = opts || {};
    var maxChars = opts.maxChars || 8;
    var minChars = opts.minChars || 1;
    var minH = opts.minHeight || 6;
    var src = shapeSource(boxes, opts.tight);
    var out = [];
    for (var i = 0; i < boxes.length; i++) {
      var b = src[i];
      var w = b[2] - b[0], h = b[3] - b[1];
      if (w < 4 || h < minH) continue;
      var est = estChars(src, i);
      if (est > maxChars || est < minChars) continue;
      out.push(i);
    }
    return out;
  }

  /** 直接给出「姓名候选框」的识别顺序（已过滤 + 已按姓名形状分桶排序） */
  function orderNameCandidates(boxes, opts) {
    opts = opts || {};
    return orderByNameShape(boxes, nameBoxIndices(boxes, opts), opts.targetRatio, opts.tight);
  }

  /**
   * 让检测输入的短边 ≈ targetShort，返回该传给 detect 的 detMaxSide（0 = 不限制）。
   * 检测分辨率与识别裁剪用的画布分辨率是两件事：画布要清晰（给识别裁图），
   * 检测按 PP-OCR 的标准输入（短边 736）就够，喂更大只是白烧算力。
   */
  function detMaxSideFor(canvas, targetShort) {
    var lim = targetShort || DET.LIMIT_SIDE;
    var mx = Math.max(canvas.width, canvas.height), mn = Math.min(canvas.width, canvas.height);
    if (!(mx > 0) || !(mn > 0)) return 0;
    if (mn <= lim) return 0;                  // 画布本来就够小，走「短边补足」逻辑
    return Math.round(mx * lim / mn);
  }

  /** 长文本优先：学院名、标题这类长串先认 */
  function orderByLongText(boxes) {
    var idx = [];
    for (var i = 0; i < boxes.length; i++) idx.push(i);
    idx.sort(function (a, b) {
      return (boxes[b][2] - boxes[b][0]) - (boxes[a][2] - boxes[a][0]);
    });
    return idx;
  }

  /**
   * 快扫用：先认最长的一小撮（抓学院/标题，命中就早停），
   * 其余按姓名形状排（顺带抓名字），总数控制在 ratio 以内。
   * @param ratio 只返回前 ratio 比例的框索引，认完即止
   */
  function orderMixed(boxes, ratio) {
    if (!boxes.length) return [];
    var n = boxes.length;
    // 至少认 12 个框：封面这类框很少的页，按比例算出来会少到连标题都认不到
    var total = Math.min(n, Math.max(12, Math.round(n * (ratio || 0.4))));
    var longN = Math.min(Math.max(1, total - 1), Math.max(2, Math.round(n * 0.15)));

    var longPart = orderByLongText(boxes).slice(0, longN);
    var namePart = orderByNameShape(boxes);
    var seen = {}, out = [], i;
    for (i = 0; i < longPart.length; i++) { seen[longPart[i]] = 1; out.push(longPart[i]); }
    for (i = 0; i < namePart.length && out.length < total; i++) {
      if (!seen[namePart[i]]) out.push(namePart[i]);
    }
    return out;
  }

  function now() {
    return (typeof performance !== 'undefined' ? performance.now() : Date.now());
  }

  /* ------------------------------------------------------------ 姓名定位 */

  function normText(s) {
    return String(s || '')
      .replace(/\s+/g, '')
      .replace(/[·•.．。:：,，;；|｜_\-—–]/g, '');
  }

  /**
   * 在识别结果里查找姓名。
   * 命中条件（从严到宽）：
   *   1) 某个识别框整框等于姓名
   *   2) 某个识别框包含姓名，且多出的字符不超过 extraChars
   *   3) 相邻 2~3 个识别框拼起来包含姓名（汉字被切分的情况）
   * @returns null 或 {items:[...], kind:'exact'|'contains'|'joined'}
   */
  function findName(items, name, extraChars) {
    var target = normText(name);
    if (!target) return null;
    var maxExtra = (extraChars === undefined) ? 2 : extraChars;

    var clean = [];
    for (var i = 0; i < items.length; i++) {
      var nt = normText(items[i].text);
      if (!nt) continue;
      clean.push({ t: nt, it: items[i] });

      if (nt === target) return { items: [items[i]], kind: 'exact' };
      if (nt.indexOf(target) >= 0 && nt.length - target.length <= maxExtra) {
        return { items: [items[i]], kind: 'contains' };
      }
    }

    /* 下面的拼接规则要求「相邻」是**空间相邻**（阅读顺序），
     * 而 items 的顺序是识别顺序（会被「按形状分桶」打乱），所以显式按 (行, 列) 排一次 */
    clean.sort(function (a, b) {
      if (Math.abs(a.it.y0 - b.it.y0) > 4) return a.it.y0 - b.it.y0;
      return a.it.x0 - b.it.x0;
    });

    for (var a = 0; a < clean.length; a++) {
      var joined = clean[a].t;
      var list = [clean[a].it];
      for (var b = a + 1; b < Math.min(a + 4, clean.length); b++) {
        if (!sameLine(clean[a].it, clean[b].it)) break;   // 只拼同一行，避免「上一行末字 + 下一行首字」凑出姓名
        joined += clean[b].t;
        list.push(clean[b].it);
        if (joined.length > target.length + maxExtra * 3) break;
        if (joined.indexOf(target) >= 0 && joined.length - target.length <= maxExtra * 2) {
          return { items: list, kind: 'joined' };
        }
      }
    }
    return null;
  }

  /** 两个识别框是否大致在同一行（竖向中心差不超过较高者的 0.8 倍） */
  function sameLine(a, b) {
    var ha = Math.abs(a.y1 - a.y0), hb = Math.abs(b.y1 - b.y0);
    var ca = (a.y0 + a.y1) / 2, cb = (b.y0 + b.y1) / 2;
    return Math.abs(ca - cb) <= 0.8 * Math.max(ha, hb, 1);
  }

  /** 命中结果 → 归一化矩形 [x0,y0,x1,y1] */
  function hitToRect(hit, pageW, pageH) {
    var list = hit.items, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (var i = 0; i < list.length; i++) {
      x0 = Math.min(x0, list[i].x0); y0 = Math.min(y0, list[i].y0);
      x1 = Math.max(x1, list[i].x1); y1 = Math.max(y1, list[i].y1);
    }
    return [x0 / pageW, y0 / pageH, x1 / pageW, y1 / pageH];
  }

  /** 给命中框加一点余量，避免红框贴字太紧 */
  function padRect(rect, padX, padY) {
    padX = padX || 0.004;
    padY = padY || 0.002;
    return [
      Math.max(0, rect[0] - padX), Math.max(0, rect[1] - padY),
      Math.min(1, rect[2] + padX), Math.min(1, rect[3] + padY)
    ];
  }

  return {
    setOrt: setOrt,
    initModels: initModels,
    isReady: isReady,
    detect: detect,
    recognize: recognize,
    runPage: runPage,
    orderByNameShape: orderByNameShape,
    orderNameCandidates: orderNameCandidates,
    nameBoxIndices: nameBoxIndices,
    detMaxSideFor: detMaxSideFor,
    orderByLongText: orderByLongText,
    orderMixed: orderMixed,
    findName: findName,
    hitToRect: hitToRect,
    padRect: padRect,
    normText: normText,
    b64ToText: b64ToText
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = PPOCR;
