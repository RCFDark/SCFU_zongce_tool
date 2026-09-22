/* ============================================================================
 * sw.js —— 为页面注入 COOP / COEP 响应头，使其进入 crossOriginIsolated 状态。
 * onnxruntime-web 只有在这种状态下才能启用多线程 WASM（约 2~3 倍提速）。
 *
 * 采用 COEP: require-corp + 对 Worker 脚本同样注入头。
 * 关键点：HTML 规范要求 Worker 脚本响应自带 COEP 头（继承检查），否则隔离页面里
 * new Worker() 一律 ERR_BLOCKED_BY_RESPONSE——onnxruntime 的 pthread（多线程推理）
 * 与 pdf.js 的 Worker 都会挂。托管平台（如 GitHub Pages）没法给文件配响应头，
 * 所以必须在 SW 里补上。credentialless 模式同样绕不开该检查，故弃用。
 * ==========================================================================*/

var COEP_CREDENTIALLESS = false;

self.addEventListener('install', function () { self.skipWaiting(); });

self.addEventListener('activate', function (e) { e.waitUntil(self.clients.claim()); });

self.addEventListener('message', function (ev) {
  if (ev.data && ev.data.type === 'coepCredentialless') {
    COEP_CREDENTIALLESS = !!ev.data.value;
  }
});

self.addEventListener('fetch', function (event) {
  var r = event.request;

/* 关键：处理「导航请求」（HTML 文档）+「Worker 脚本请求」（destination=worker）。
 *
 * 跨源隔离要求**文档响应**带 COOP/COEP；子资源由浏览器按文档上的 COEP 原生校验。
 * 但 Worker 是例外：HTML 规范规定 Worker 脚本响应必须自带 COEP 头（继承检查），
 * 否则隔离页面里 new Worker() 一律报 ERR_BLOCKED_BY_RESPONSE——
 * onnxruntime 的 pthread、pdf.js 的 Worker 全都会挂。因此 Worker 脚本也必须注入头。
 *
 * 若 SW 拦截请求并重建 Response，会踩到一个致命坑：SW 里 fetch() 拿到的 response.body
 * 已是**解压后的明文**，但复制过来的响应头仍带 Content-Encoding: gzip/br ——
 * 浏览器再解一次 → 内容损坏 → 脚本报 "Uncaught SyntaxError: Unexpected end of input"，
 * pdf.js 的 Worker 因此起不来，退化成 fake worker 并抛
 * "Cannot read properties of undefined (reading 'WorkerMessageHandler')"。
 *
 * 因此这里对其它请求直接放行（不调用 respondWith），让浏览器走原生流程。 */
  var isNav = r.mode === 'navigate';
  var isWorkerScript = r.destination === 'worker';
  if (!isNav && !isWorkerScript) return;

  event.respondWith(
    fetch(r).then(function (response) {
      if (response.status === 0) return response;          // opaque，原样返回
      var h = new Headers(response.headers);
      // 防御：重建 Response 时 body 已被 SW 解压，必须去掉编码相关头，避免二次解码
      h.delete('Content-Encoding');
      h.delete('Content-Length');
      h.set('Cross-Origin-Embedder-Policy', COEP_CREDENTIALLESS ? 'credentialless' : 'require-corp');
      h.set('Cross-Origin-Opener-Policy', 'same-origin');
      if (!COEP_CREDENTIALLESS) h.set('Cross-Origin-Resource-Policy', 'cross-origin');
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: h
      });
    }).catch(function () { /* 失败则不干预，交给浏览器默认处理 */ })
  );
});
