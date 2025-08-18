/// <reference types="vitest" />
import { defineConfig } from 'vite';
export default defineConfig({
    test: {
        include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}', '__tests__/**/*.{test,spec}.{js,ts,jsx,tsx}'],
        exclude: ['node_modules', 'dist', '**/*.d.ts'],
        environment: 'node',
        tsconfig: 'tsconfig.json',
        coverage: { reporter: ['text'], include: ['src/**/*.{ts,tsx}'] },
        setupFiles: ['./test/setup.ts'],
        outputFile: './test.log'
    }
});
