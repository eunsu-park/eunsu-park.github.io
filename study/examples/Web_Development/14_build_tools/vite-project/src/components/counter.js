/**
 * 카운터 컴포넌트
 */

export function setupCounter(element) {
    let count = 0;

    const updateButton = () => {
        element.textContent = `카운터: ${count}`;
    };

    element.addEventListener('click', () => {
        count++;
        updateButton();
    });

    updateButton();
}
