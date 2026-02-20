/**
 * Vite 설정 파일
 * https://vitejs.dev/config/
 */

import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    // 개발 서버 설정
    server: {
        port: 3000,           // 포트 번호
        open: true,           // 브라우저 자동 열기
        cors: true,           // CORS 허용
        host: true,           // 네트워크에서 접근 허용
        // 프록시 설정 (API 서버 연동 시)
        // proxy: {
        //     '/api': {
        //         target: 'http://localhost:8080',
        //         changeOrigin: true,
        //         rewrite: (path) => path.replace(/^\/api/, '')
        //     }
        // }
    },

    // 빌드 설정
    build: {
        outDir: 'dist',       // 출력 디렉토리
        sourcemap: true,      // 소스맵 생성
        minify: 'terser',     // 압축 방식 (terser 또는 esbuild)
        target: 'es2020',     // 타겟 브라우저

        // 번들 분할 설정
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                // 다중 페이지 앱의 경우
                // about: resolve(__dirname, 'about.html'),
            },
            output: {
                // 청크 파일 이름 패턴
                chunkFileNames: 'assets/js/[name]-[hash].js',
                entryFileNames: 'assets/js/[name]-[hash].js',
                assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',

                // 벤더 번들 분리
                manualChunks: {
                    // vendor: ['lodash', 'axios'],
                }
            }
        },

        // 청크 크기 경고 임계값 (KB)
        chunkSizeWarningLimit: 500,
    },

    // 경로 별칭 설정
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@components': resolve(__dirname, 'src/components'),
            '@utils': resolve(__dirname, 'src/utils'),
            '@styles': resolve(__dirname, 'src/styles'),
        }
    },

    // CSS 설정
    css: {
        // CSS 모듈 설정
        modules: {
            localsConvention: 'camelCase',
        },
        // PostCSS 설정
        postcss: {
            plugins: [
                // autoprefixer 등 플러그인 추가
            ]
        },
        // 전처리기 옵션
        preprocessorOptions: {
            scss: {
                // 전역 변수 파일 자동 import
                // additionalData: `@import "@/styles/variables.scss";`
            }
        }
    },

    // 환경 변수 접두사
    envPrefix: 'VITE_',

    // 플러그인
    plugins: [
        // @vitejs/plugin-react
        // @vitejs/plugin-vue
        // @vitejs/plugin-legacy (구형 브라우저 지원)
    ],

    // 최적화 설정
    optimizeDeps: {
        // 사전 번들링할 의존성
        include: [],
        // 사전 번들링에서 제외할 의존성
        exclude: []
    },

    // 미리보기 서버 설정
    preview: {
        port: 4173,
        open: true,
    }
});
