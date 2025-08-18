import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import fs from 'fs/promises';
const STATE_FILE = path.join(process.cwd(), 'state.json');
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
        define: {
            'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
            'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
        },
        resolve: {
            alias: {
                '@': path.resolve(__dirname, '.'),
            }
        },
        plugins: [
            {
                name: 'state-api',
                configureServer(server) {
                    server.middlewares.use('/api/state', async (req, res, next) => {
                        try {
                            const data = await fs.readFile(STATE_FILE, 'utf-8');
                            res.setHeader('Content-Type', 'application/json');
                            res.end(data);
                        }
                        catch (err) {
                            res.statusCode = 500;
                            res.end(JSON.stringify({ error: 'Failed to read state file' }));
                        }
                    });
                },
            },
        ]
    };
});
