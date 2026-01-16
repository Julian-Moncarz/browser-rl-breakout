/**
 * Simple HTTP server to serve the model viewer.
 * Serves static files and the trained model.
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = 3000;
const USE_BEST = process.argv.includes('--best');

const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.json': 'application/json',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.bin': 'application/octet-stream'
};

function serveFile(filePath, res) {
    const ext = path.extname(filePath);
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, (err, data) => {
        if (err) {
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('Not Found');
            return;
        }
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
    });
}

const server = http.createServer((req, res) => {
    // Parse URL
    let urlPath = req.url.split('?')[0];

    // Default to watch.html
    if (urlPath === '/') {
        urlPath = '/watch.html';
    }

    // Security: prevent directory traversal
    const safePath = path.normalize(urlPath).replace(/^(\.\.[\/\\])+/, '');
    const filePath = path.join(__dirname, safePath);

    // Check file exists
    if (!fs.existsSync(filePath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
        return;
    }

    // Serve the file
    serveFile(filePath, res);
});

// Check if model exists
const modelDir = USE_BEST ? 'model.best' : 'model';
const modelPath = path.join(__dirname, modelDir, 'model.json');
if (!fs.existsSync(modelPath)) {
    console.log(`\x1b[33mWarning: No trained model found at ./${modelDir}/\x1b[0m`);
    console.log('Run "npm run train" first to train a model.\n');
}

server.listen(PORT, () => {
    console.log(`\n  Breakout RL Viewer`);
    console.log(`  ------------------`);
    console.log(`  Model: \x1b[33m${USE_BEST ? 'Best (model.best)' : 'Latest (model)'}\x1b[0m`);
    console.log(`  Server: \x1b[36mhttp://localhost:${PORT}\x1b[0m`);
    console.log(`\n  Press Ctrl+C to stop.\n`);

    // Try to open browser with model query param
    const url = USE_BEST
        ? `http://localhost:${PORT}?model=model.best`
        : `http://localhost:${PORT}`;
    const start = process.platform === 'darwin' ? 'open' :
                  process.platform === 'win32' ? 'start' : 'xdg-open';

    import('child_process').then(({ exec }) => {
        exec(`${start} ${url}`);
    });
});
