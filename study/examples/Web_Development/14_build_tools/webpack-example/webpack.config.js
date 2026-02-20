/**
 * Webpack 설정 파일
 * https://webpack.js.org/configuration/
 */

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
    const isProduction = argv.mode === 'production';

    return {
        // 진입점
        entry: {
            main: './src/index.js',
            // 다중 진입점 예시
            // vendor: './src/vendor.js',
        },

        // 출력 설정
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: isProduction ? '[name].[contenthash].js' : '[name].js',
            chunkFilename: isProduction ? '[name].[contenthash].chunk.js' : '[name].chunk.js',
            clean: true,  // 빌드 전 dist 폴더 정리
            publicPath: '/',
        },

        // 개발 서버 설정
        devServer: {
            static: {
                directory: path.join(__dirname, 'public'),
            },
            port: 3000,
            open: true,
            hot: true,  // HMR 활성화
            historyApiFallback: true,  // SPA 라우팅 지원
            compress: true,
            // 프록시 설정
            // proxy: {
            //     '/api': {
            //         target: 'http://localhost:8080',
            //         changeOrigin: true,
            //     }
            // }
        },

        // 모듈 로더 설정
        module: {
            rules: [
                // JavaScript/JSX 처리
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                            cacheDirectory: true,
                        }
                    }
                },

                // CSS 처리
                {
                    test: /\.css$/,
                    use: [
                        isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
                        {
                            loader: 'css-loader',
                            options: {
                                sourceMap: !isProduction,
                            }
                        }
                    ]
                },

                // 이미지 처리 (asset modules)
                {
                    test: /\.(png|jpe?g|gif|svg|webp)$/i,
                    type: 'asset',
                    parser: {
                        dataUrlCondition: {
                            maxSize: 8 * 1024, // 8KB 이하는 인라인
                        }
                    },
                    generator: {
                        filename: 'images/[name].[hash:8][ext]'
                    }
                },

                // 폰트 처리
                {
                    test: /\.(woff|woff2|eot|ttf|otf)$/i,
                    type: 'asset/resource',
                    generator: {
                        filename: 'fonts/[name].[hash:8][ext]'
                    }
                }
            ]
        },

        // 플러그인
        plugins: [
            // HTML 템플릿 처리
            new HtmlWebpackPlugin({
                template: './src/index.html',
                filename: 'index.html',
                inject: 'body',
                minify: isProduction ? {
                    removeComments: true,
                    collapseWhitespace: true,
                    removeAttributeQuotes: true,
                } : false,
            }),

            // CSS 추출 (프로덕션)
            ...(isProduction ? [
                new MiniCssExtractPlugin({
                    filename: 'css/[name].[contenthash].css',
                    chunkFilename: 'css/[name].[contenthash].chunk.css',
                })
            ] : []),
        ],

        // 경로 별칭
        resolve: {
            extensions: ['.js', '.json'],
            alias: {
                '@': path.resolve(__dirname, 'src'),
                '@components': path.resolve(__dirname, 'src/components'),
                '@utils': path.resolve(__dirname, 'src/utils'),
                '@styles': path.resolve(__dirname, 'src/styles'),
            }
        },

        // 최적화
        optimization: {
            // 코드 분할
            splitChunks: {
                chunks: 'all',
                cacheGroups: {
                    // 벤더 번들 분리
                    vendors: {
                        test: /[\\/]node_modules[\\/]/,
                        name: 'vendors',
                        priority: -10,
                    },
                    // 공통 모듈 분리
                    common: {
                        minChunks: 2,
                        priority: -20,
                        reuseExistingChunk: true,
                    }
                }
            },
            // 런타임 청크 분리
            runtimeChunk: 'single',
        },

        // 소스맵
        devtool: isProduction ? 'source-map' : 'eval-source-map',

        // 성능 힌트
        performance: {
            hints: isProduction ? 'warning' : false,
            maxEntrypointSize: 250000,  // 250KB
            maxAssetSize: 250000,
        },

        // 캐시 설정
        cache: {
            type: 'filesystem',
            buildDependencies: {
                config: [__filename],
            }
        },

        // 통계 출력 설정
        stats: {
            colors: true,
            modules: false,
            children: false,
        }
    };
};
