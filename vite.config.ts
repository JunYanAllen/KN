import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { host: true, port: 5173 },
  build: {
    target: 'es2022',
    assetsInlineLimit: 100_000_000, // inline everything we can; the game ships as one file
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        entryFileNames: 'game.js',
        assetFileNames: 'game.[ext]',
      },
    },
  },
});
