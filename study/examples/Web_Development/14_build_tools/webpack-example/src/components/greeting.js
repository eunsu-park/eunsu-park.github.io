/**
 * 인사 컴포넌트
 */

export function greeting(name) {
    return `
        <div class="greeting">
            <h2>안녕하세요, ${name}님!</h2>
            <p>Webpack으로 빌드된 모듈입니다.</p>
        </div>
    `;
}
