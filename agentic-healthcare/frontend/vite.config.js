import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const projectDirectory = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  publicDir: path.resolve(projectDirectory, '../ml/results'),
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
})