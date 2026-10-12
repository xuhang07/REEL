import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../docs');
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.webp': 'image/webp', '.pdf': 'application/pdf' };
export function previewServer() {
  return createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/REEL(?=\/)/, '');
      const target = resolve(root, `.${pathname.endsWith('/') ? pathname + 'index.html' : pathname}`);
      if (!target.startsWith(root + sep)) { response.writeHead(403).end(); return; }
      const metadata = await stat(target);
      if (!metadata.isFile()) { response.writeHead(404).end(); return; }
      response.writeHead(200, { 'Content-Type': types[extname(target)] || 'application/octet-stream', 'Content-Length': metadata.size });
      if (request.method === 'HEAD') response.end();
      else createReadStream(target).pipe(response);
    } catch { response.writeHead(404).end(); }
  });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 4173);
  previewServer().listen(port, '127.0.0.1', () => console.log(`REEL preview: http://127.0.0.1:${port}/REEL/`));
}
