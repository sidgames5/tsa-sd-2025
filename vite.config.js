import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:5000',
                changeOrigin: true,
                // rewrite: (path) => path.replace(/^\/api/, ''),
                secure: false,
                configure: (proxy) => {
                    proxy.on('proxyReq', (proxyReq, req) => {
                        if (req.headers['authorization']) {
                            proxyReq.setHeader('Authorization', req.headers['authorization']);
                        }
                        if (req.headers['content-type']) {
                            proxyReq.setHeader('Content-Type', req.headers['content-type']);
                        }
                        if (req.headers['content-length']) {
                            proxyReq.setHeader('Content-Length', req.headers['content-length']);
                        }
                    });
                },
            },
        },
    },
});
