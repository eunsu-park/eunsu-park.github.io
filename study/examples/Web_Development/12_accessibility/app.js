/**
 * 웹 접근성 예제 JavaScript
 * - 아코디언
 * - 탭 컴포넌트
 * - 모달
 * - 커스텀 리스트박스
 * - 포커스 관리
 */

// ============================================
// 아코디언
// ============================================
function initAccordion() {
    const button = document.getElementById('accordion-btn');
    const content = document.getElementById('accordion-content');

    if (!button || !content) return;

    button.addEventListener('click', () => {
        const isExpanded = button.getAttribute('aria-expanded') === 'true';

        button.setAttribute('aria-expanded', !isExpanded);
        content.hidden = isExpanded;
    });
}

// ============================================
// 탭 컴포넌트
// ============================================
function initTabs() {
    const tablist = document.querySelector('[role="tablist"]');
    if (!tablist) return;

    const tabs = tablist.querySelectorAll('[role="tab"]');
    const panels = document.querySelectorAll('[role="tabpanel"]');

    // 탭 클릭 이벤트
    tabs.forEach(tab => {
        tab.addEventListener('click', () => activateTab(tab, tabs, panels));
    });

    // 키보드 네비게이션
    tablist.addEventListener('keydown', (e) => {
        const currentTab = document.activeElement;
        const currentIndex = Array.from(tabs).indexOf(currentTab);

        let newIndex;

        switch (e.key) {
            case 'ArrowLeft':
                newIndex = currentIndex - 1;
                if (newIndex < 0) newIndex = tabs.length - 1;
                break;
            case 'ArrowRight':
                newIndex = currentIndex + 1;
                if (newIndex >= tabs.length) newIndex = 0;
                break;
            case 'Home':
                newIndex = 0;
                break;
            case 'End':
                newIndex = tabs.length - 1;
                break;
            default:
                return;
        }

        e.preventDefault();
        tabs[newIndex].focus();
        activateTab(tabs[newIndex], tabs, panels);
    });
}

function activateTab(selectedTab, tabs, panels) {
    // 모든 탭 비활성화
    tabs.forEach(tab => {
        tab.setAttribute('aria-selected', 'false');
        tab.setAttribute('tabindex', '-1');
    });

    // 모든 패널 숨기기
    panels.forEach(panel => {
        panel.hidden = true;
    });

    // 선택된 탭 활성화
    selectedTab.setAttribute('aria-selected', 'true');
    selectedTab.setAttribute('tabindex', '0');

    // 해당 패널 표시
    const panelId = selectedTab.getAttribute('aria-controls');
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.hidden = false;
    }
}

// ============================================
// Live Region 데모
// ============================================
function initLiveRegion() {
    const button = document.getElementById('update-live-btn');
    const liveRegion = document.getElementById('live-region');

    if (!button || !liveRegion) return;

    const messages = [
        '새로운 알림이 도착했습니다.',
        '작업이 완료되었습니다.',
        '3개의 새 메시지가 있습니다.',
        '파일이 성공적으로 업로드되었습니다.',
        '설정이 저장되었습니다.'
    ];

    let messageIndex = 0;

    button.addEventListener('click', () => {
        liveRegion.textContent = messages[messageIndex];
        messageIndex = (messageIndex + 1) % messages.length;
    });
}

// ============================================
// 모달
// ============================================
let previousFocusElement = null;

function initModal() {
    const openBtn = document.getElementById('open-modal-btn');
    const modal = document.getElementById('modal');
    const overlay = document.getElementById('modal-overlay');
    const closeBtn = document.getElementById('modal-close');
    const confirmBtn = document.getElementById('modal-confirm');

    if (!openBtn || !modal) return;

    // 열기
    openBtn.addEventListener('click', () => openModal(modal, overlay));

    // 닫기 버튼
    closeBtn?.addEventListener('click', () => closeModal(modal, overlay));
    confirmBtn?.addEventListener('click', () => {
        alert('확인되었습니다!');
        closeModal(modal, overlay);
    });

    // 오버레이 클릭
    overlay?.addEventListener('click', () => closeModal(modal, overlay));

    // ESC 키
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.hidden) {
            closeModal(modal, overlay);
        }
    });

    // 포커스 트랩
    modal.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;

        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    });
}

function openModal(modal, overlay) {
    // 현재 포커스 저장
    previousFocusElement = document.activeElement;

    // 모달 표시
    modal.hidden = false;
    overlay.hidden = false;

    // 배경 스크롤 방지
    document.body.style.overflow = 'hidden';

    // 첫 번째 포커스 가능한 요소에 포커스
    const firstFocusable = modal.querySelector('button, [href], input');
    if (firstFocusable) {
        firstFocusable.focus();
    }
}

function closeModal(modal, overlay) {
    modal.hidden = true;
    overlay.hidden = true;

    // 배경 스크롤 복원
    document.body.style.overflow = '';

    // 이전 포커스 복원
    if (previousFocusElement) {
        previousFocusElement.focus();
    }
}

// ============================================
// 커스텀 리스트박스
// ============================================
function initListbox() {
    const listbox = document.getElementById('custom-listbox');
    const output = document.getElementById('listbox-output');

    if (!listbox) return;

    const options = listbox.querySelectorAll('[role="option"]');
    let currentIndex = 0;

    // 초기 선택 상태 설정
    updateSelection(options, currentIndex);

    listbox.addEventListener('keydown', (e) => {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                currentIndex = Math.min(currentIndex + 1, options.length - 1);
                updateSelection(options, currentIndex);
                break;
            case 'ArrowUp':
                e.preventDefault();
                currentIndex = Math.max(currentIndex - 1, 0);
                updateSelection(options, currentIndex);
                break;
            case 'Home':
                e.preventDefault();
                currentIndex = 0;
                updateSelection(options, currentIndex);
                break;
            case 'End':
                e.preventDefault();
                currentIndex = options.length - 1;
                updateSelection(options, currentIndex);
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                selectOption(options[currentIndex], output);
                break;
        }
    });

    // 클릭으로 선택
    options.forEach((option, index) => {
        option.addEventListener('click', () => {
            currentIndex = index;
            updateSelection(options, currentIndex);
            selectOption(option, output);
        });
    });
}

function updateSelection(options, index) {
    options.forEach((option, i) => {
        option.setAttribute('aria-selected', i === index);
    });

    // listbox의 aria-activedescendant 업데이트
    const listbox = options[0]?.parentElement;
    if (listbox) {
        listbox.setAttribute('aria-activedescendant', options[index].id);
    }
}

function selectOption(option, output) {
    const text = option.textContent.trim();
    const fruitName = text.replace(/^.+\s/, ''); // 이모지 제거
    output.textContent = `선택된 항목: ${fruitName}`;
}

// ============================================
// 폼 유효성 검사
// ============================================
function initFormValidation() {
    const form = document.getElementById('accessible-form');
    if (!form) return;

    const emailInput = document.getElementById('email');
    const emailError = document.getElementById('email-error');

    // 실시간 이메일 유효성 검사
    emailInput?.addEventListener('blur', () => {
        validateEmail(emailInput, emailError);
    });

    emailInput?.addEventListener('input', () => {
        if (emailInput.getAttribute('aria-invalid') === 'true') {
            validateEmail(emailInput, emailError);
        }
    });

    // 폼 제출
    form.addEventListener('submit', (e) => {
        e.preventDefault();

        let isValid = true;

        // 이메일 검증
        if (!validateEmail(emailInput, emailError)) {
            isValid = false;
        }

        // 필수 필드 검증
        const requiredFields = form.querySelectorAll('[required]');
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                isValid = false;
                field.setAttribute('aria-invalid', 'true');
            } else {
                field.setAttribute('aria-invalid', 'false');
            }
        });

        if (isValid) {
            alert('폼이 성공적으로 제출되었습니다!');
            form.reset();
        } else {
            // 첫 번째 오류 필드로 포커스
            const firstError = form.querySelector('[aria-invalid="true"]');
            if (firstError) {
                firstError.focus();
            }
        }
    });
}

function validateEmail(input, errorElement) {
    if (!input || !errorElement) return true;

    const email = input.value.trim();
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!email) {
        showError(input, errorElement, '이메일을 입력해주세요.');
        return false;
    }

    if (!emailRegex.test(email)) {
        showError(input, errorElement, '유효한 이메일 형식이 아닙니다.');
        return false;
    }

    hideError(input, errorElement);
    return true;
}

function showError(input, errorElement, message) {
    input.setAttribute('aria-invalid', 'true');
    errorElement.textContent = message;
    errorElement.hidden = false;
}

function hideError(input, errorElement) {
    input.setAttribute('aria-invalid', 'false');
    errorElement.textContent = '';
    errorElement.hidden = true;
}

// ============================================
// 포커스 트랩 데모
// ============================================
function initFocusTrap() {
    const trapArea = document.getElementById('focus-trap-demo');
    if (!trapArea) return;

    const focusableElements = trapArea.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    trapArea.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    });
}

// ============================================
// 초기화
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initAccordion();
    initTabs();
    initLiveRegion();
    initModal();
    initListbox();
    initFormValidation();
    initFocusTrap();

    console.log('웹 접근성 예제가 로드되었습니다.');
    console.log('스크린 리더나 키보드로 테스트해 보세요!');
});
