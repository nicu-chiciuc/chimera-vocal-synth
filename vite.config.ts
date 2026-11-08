import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

/**
 * The comparison page's data (`?voicefit`, dev server only): `/voicefit/<file>` serves a voice fit's
 * folder, read-only (tools/voicefit: the recording, the band, the renders, candidates.json). The
 * folder is `VOICEFIT_DIR`, or
 * `out/voicefit` (gitignored; a link to the fit's folder). Nothing of it is built or published:
 * the recording is usually someone else's.
 */
function voicefit(): Plugin {
  const root = () => path.resolve(process.env.VOICEFIT_DIR ?? "out/voicefit");
  const TYPES: Record<string, string> = { ".wav": "audio/wav", ".json": "application/json" };
  return {
    name: "voicefit",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/voicefit/", (req, res) => {
        const rel = decodeURIComponent((req.url ?? "").split("?")[0]).replace(/^\/+/, "");
        const type = TYPES[path.extname(rel)];
        if (!/^[\w.-]+(\/[\w.-]+)*$/.test(rel) || rel.split("/").includes("..") || !type) {
          res.statusCode = 400;
          return res.end("bad path");
        }
        let dir: string;
        try {
          dir = fs.realpathSync(root());
        } catch {
          res.statusCode = 404;
          return res.end(`no voice fit folder: set VOICEFIT_DIR or link out/voicefit`);
        }
        const file = path.join(dir, rel);
        if (req.method !== "GET" && req.method !== "HEAD") {
          res.statusCode = 405;
          return res.end();
        }
        let real: string;
        try {
          real = fs.realpathSync(file);
        } catch {
          res.statusCode = 404;
          return res.end();
        }
        if (!real.startsWith(dir + path.sep)) {
          res.statusCode = 403;
          return res.end();
        }
        res.setHeader("Content-Type", type);
        res.setHeader("Cache-Control", "no-store");
        fs.createReadStream(real).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), voicefit()],
  worker: { format: "es" },
  build: { target: "es2022", sourcemap: true },
  server: { host: "127.0.0.1", port: 5317, strictPort: true, allowedHosts: true },
  preview: { host: "127.0.0.1", port: 5318, strictPort: true, allowedHosts: true },
});
