/**
 * 추가 컨텐츠 컴포넌트
 * 동적 임포트 (Code Splitting) 예제
 */

export function loadExtraContent(container) {
    const extraContent = document.createElement('div');
    extraContent.className = 'extra-content';
    extraContent.innerHTML = `
        <h3>추가 컨텐츠</h3>
        <p>이 컨텐츠는 동적으로 로드되었습니다!</p>
        <p>별도의 청크(chunk)로 분리되어 필요할 때만 로드됩니다.</p>
        <ul>
            <li>초기 로드 시간 단축</li>
            <li>필요한 코드만 로드</li>
            <li>캐시 효율성 향상</li>
        </ul>
    `;
    container.appendChild(extraContent);

    console.log('✅ 추가 컨텐츠가 로드되었습니다!');
}
