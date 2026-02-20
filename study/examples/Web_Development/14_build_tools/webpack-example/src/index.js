/**
 * Webpack 프로젝트 메인 진입점
 *
 * Webpack 특징:
 * - 모듈 번들링
 * - 코드 분할 (Code Splitting)
 * - 트리 쉐이킹 (Tree Shaking)
 * - 로더와 플러그인 시스템
 */

// CSS 임포트
import './styles/main.css';

// 컴포넌트 임포트
import { greeting } from './components/greeting';
import { formatDate } from './utils/helpers';

// 앱 초기화
function initApp() {
    console.log('📦 Webpack 앱이 시작되었습니다!');

    const content = document.getElementById('content');
    if (content) {
        content.innerHTML = greeting('Webpack 사용자');
    }

    console.log(`📅 현재 시간: ${formatDate(new Date())}`);

    // 동적 임포트 (Code Splitting) 예제
    const loadMoreBtn = document.getElementById('loadMore');
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', async () => {
            // 동적 임포트 - 별도 청크로 분리됨
            const { loadExtraContent } = await import(
                /* webpackChunkName: "extra" */
                './components/extra'
            );
            loadExtraContent(content);
        });
    }
}

// DOM 로드 후 초기화
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

// HMR (Hot Module Replacement)
if (module.hot) {
    module.hot.accept('./components/greeting', () => {
        console.log('🔄 greeting 모듈이 업데이트되었습니다!');
        const content = document.getElementById('content');
        if (content) {
            const { greeting } = require('./components/greeting');
            content.innerHTML = greeting('Webpack 사용자');
        }
    });
}
