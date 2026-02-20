/**
 * Vite 프로젝트 메인 진입점
 *
 * Vite 특징:
 * - 네이티브 ES 모듈 사용 (빠른 개발 서버)
 * - HMR (Hot Module Replacement) 지원
 * - 최적화된 프로덕션 빌드
 */

// CSS 임포트 (Vite가 자동으로 처리)
import './styles/main.css';

// 모듈 임포트
import { setupCounter } from './components/counter.js';
import { formatDate } from './utils/helpers.js';

// 앱 초기화
function initApp() {
    console.log('🚀 Vite 앱이 시작되었습니다!');
    console.log(`📅 현재 시간: ${formatDate(new Date())}`);

    // 카운터 설정
    const counterButton = document.getElementById('counter');
    if (counterButton) {
        setupCounter(counterButton);
    }

    // 개발 모드에서만 실행되는 코드
    if (import.meta.env.DEV) {
        console.log('🔧 개발 모드로 실행 중');
        console.log('환경 변수:', import.meta.env);
    }

    // 프로덕션 모드에서만 실행되는 코드
    if (import.meta.env.PROD) {
        console.log('🚀 프로덕션 모드로 실행 중');
    }
}

// DOM 로드 후 초기화
document.addEventListener('DOMContentLoaded', initApp);

// HMR (Hot Module Replacement) 예제
if (import.meta.hot) {
    import.meta.hot.accept('./components/counter.js', (newModule) => {
        console.log('🔄 counter 모듈이 업데이트되었습니다!');
        // 필요한 경우 상태 복원 로직 추가
    });
}
