/**
 * 유틸리티 함수
 */

export function formatDate(date) {
    return new Intl.DateTimeFormat('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    }).format(date);
}

// 사용되지 않는 함수 (Tree Shaking 대상)
export function unusedFunction() {
    console.log('이 함수는 사용되지 않아 프로덕션 빌드에서 제거됩니다.');
}
