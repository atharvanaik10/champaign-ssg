import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/plan': 'http://localhost:8000',
      '/graph': 'http://localhost:8000',
      '/jobs': 'http://localhost:8000'
    }
  }
});
