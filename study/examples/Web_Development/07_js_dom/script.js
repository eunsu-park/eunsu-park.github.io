/*
 * DOM 조작과 이벤트 예제
 */

// ============================================
// 1. 요소 선택
// ============================================
function demoSelection() {
    const output = document.getElementById('selectOutput');
    let result = '';

    // getElementById
    const byId = document.getElementById('uniqueId');
    result += `getElementById: "${byId.textContent}"\n`;

    // getElementsByClassName
    const byClass = document.getElementsByClassName('myClass');
    result += `getElementsByClassName: ${byClass.length}개 요소\n`;

    // querySelector (첫 번째 일치)
    const byQuery = document.querySelector('.myClass');
    result += `querySelector('.myClass'): "${byQuery.textContent}"\n`;

    // querySelectorAll (모든 일치)
    const byQueryAll = document.querySelectorAll('.myClass');
    result += `querySelectorAll('.myClass'): ${byQueryAll.length}개 요소\n`;

    // 속성 선택자
    const byData = document.querySelector('[data-info="test"]');
    result += `[data-info="test"]: "${byData.textContent}"\n`;

    // 복합 선택자
    const complex = document.querySelector('#selectDemo .container span');
    result += `복합 선택자: "${complex.textContent}"`;

    output.textContent = result;
}

// ============================================
// 2. 요소 생성 및 추가
// ============================================
function addItem() {
    const input = document.getElementById('newItemInput');
    const list = document.getElementById('itemList');
    const text = input.value.trim();

    if (!text) return;

    // 요소 생성
    const li = document.createElement('li');
    li.textContent = text;

    // 삭제 버튼 추가
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '×';
    deleteBtn.style.cssText = 'margin-left: 10px; padding: 2px 8px; background: #e74c3c;';
    deleteBtn.onclick = () => li.remove();

    li.appendChild(deleteBtn);
    list.appendChild(li);  // 끝에 추가

    input.value = '';
    input.focus();
}

function addItemBefore() {
    const input = document.getElementById('newItemInput');
    const list = document.getElementById('itemList');
    const text = input.value.trim();

    if (!text) return;

    const li = document.createElement('li');
    li.textContent = text;

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '×';
    deleteBtn.style.cssText = 'margin-left: 10px; padding: 2px 8px; background: #e74c3c;';
    deleteBtn.onclick = () => li.remove();

    li.appendChild(deleteBtn);

    // 맨 앞에 추가
    list.insertBefore(li, list.firstChild);

    input.value = '';
}

function clearItems() {
    const list = document.getElementById('itemList');
    // 모든 자식 제거
    list.innerHTML = '';
    // 또는: while (list.firstChild) list.removeChild(list.firstChild);
}

// Enter 키로 추가
document.getElementById('newItemInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addItem();
});

// ============================================
// 3. 스타일 및 클래스 조작
// ============================================
function changeColor() {
    const box = document.getElementById('styleBox');
    const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'];
    const randomColor = colors[Math.floor(Math.random() * colors.length)];
    box.style.backgroundColor = randomColor;
    box.style.color = 'white';
}

function toggleHighlight() {
    const box = document.getElementById('styleBox');
    box.classList.toggle('highlight');
}

function addBorder() {
    const box = document.getElementById('styleBox');
    box.style.border = '3px solid #2c3e50';
    box.style.borderRadius = '15px';
}

function resetStyle() {
    const box = document.getElementById('styleBox');
    box.style.cssText = 'transition: all 0.3s;';
    box.classList.remove('highlight');
}

// ============================================
// 4. 속성 조작
// ============================================
function changeImage(num) {
    const img = document.getElementById('demoImage');
    img.src = `https://via.placeholder.com/200x100?text=Image+${num}`;
    img.alt = `데모 이미지 ${num}`;
    updateAltDisplay();
}

function changeAlt() {
    const img = document.getElementById('demoImage');
    const altInput = document.getElementById('altInput');
    img.setAttribute('alt', altInput.value);
    updateAltDisplay();
    altInput.value = '';
}

function updateAltDisplay() {
    const img = document.getElementById('demoImage');
    document.getElementById('currentAlt').textContent = img.getAttribute('alt');
}

// 초기 alt 표시
updateAltDisplay();

// ============================================
// 5. 이벤트 리스너
// ============================================
const eventBox = document.getElementById('eventBox');
const eventLog = document.getElementById('eventLog');

function logEvent(message) {
    const time = new Date().toLocaleTimeString();
    eventLog.innerHTML = `[${time}] ${message}<br>` + eventLog.innerHTML;
}

function clearEventLog() {
    eventLog.innerHTML = '';
}

// 클릭 이벤트
eventBox.addEventListener('click', (e) => {
    logEvent(`클릭! 좌표: (${e.offsetX}, ${e.offsetY})`);
});

// 더블클릭
eventBox.addEventListener('dblclick', () => {
    logEvent('더블클릭!');
});

// 마우스 진입/이탈
eventBox.addEventListener('mouseenter', () => {
    logEvent('마우스 진입');
    eventBox.style.backgroundColor = '#ecf0f1';
});

eventBox.addEventListener('mouseleave', () => {
    logEvent('마우스 이탈');
    eventBox.style.backgroundColor = '';
});

// 마우스 이동 (throttle 적용)
let lastMoveLog = 0;
eventBox.addEventListener('mousemove', (e) => {
    const now = Date.now();
    if (now - lastMoveLog > 500) {  // 500ms마다 로깅
        logEvent(`마우스 이동: (${e.offsetX}, ${e.offsetY})`);
        lastMoveLog = now;
    }
});

// 컨텍스트 메뉴 (우클릭)
eventBox.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    logEvent('우클릭 (기본 동작 방지됨)');
});

// ============================================
// 6. 이벤트 위임
// ============================================
const cardContainer = document.getElementById('cardContainer');
let cardCount = 5;

cardContainer.addEventListener('click', (e) => {
    const card = e.target.closest('.card');
    if (!card) return;

    // 모든 카드에서 selected 제거
    cardContainer.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));

    // 클릭된 카드에 selected 추가
    card.classList.add('selected');

    document.getElementById('selectedCard').textContent = `카드 ${card.dataset.id}`;
});

function addCard() {
    cardCount++;
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.id = cardCount;
    card.textContent = cardCount;
    cardContainer.appendChild(card);
}

// ============================================
// 7. 폼 이벤트
// ============================================
const demoForm = document.getElementById('demoForm');
const formOutput = document.getElementById('formOutput');

// input 이벤트 (실시간)
document.getElementById('nameInput').addEventListener('input', (e) => {
    console.log('입력 중:', e.target.value);
});

// change 이벤트 (포커스 이탈 시)
document.getElementById('countrySelect').addEventListener('change', (e) => {
    formOutput.textContent = `국가 변경: ${e.target.value || '선택 안함'}`;
});

// focus/blur 이벤트
document.getElementById('emailInput').addEventListener('focus', (e) => {
    e.target.style.borderColor = '#3498db';
});

document.getElementById('emailInput').addEventListener('blur', (e) => {
    e.target.style.borderColor = '';
});

// submit 이벤트
demoForm.addEventListener('submit', (e) => {
    e.preventDefault();  // 기본 동작 방지

    const formData = new FormData(demoForm);
    const name = document.getElementById('nameInput').value;
    const email = document.getElementById('emailInput').value;
    const country = document.getElementById('countrySelect').value;

    formOutput.innerHTML = `
        <strong>제출된 데이터:</strong><br>
        이름: ${name}<br>
        이메일: ${email}<br>
        국가: ${country || '미선택'}
    `;
});

// ============================================
// 8. 키보드 이벤트
// ============================================
const keyInput = document.getElementById('keyInput');
const keyOutput = document.getElementById('keyOutput');

keyInput.addEventListener('keydown', (e) => {
    keyOutput.innerHTML = `
        <strong>keydown</strong><br>
        key: "${e.key}"<br>
        code: "${e.code}"<br>
        keyCode: ${e.keyCode} (deprecated)<br>
        ctrlKey: ${e.ctrlKey}, shiftKey: ${e.shiftKey}, altKey: ${e.altKey}
    `;

    // 특수 키 처리
    if (e.key === 'Escape') {
        keyInput.value = '';
        keyOutput.innerHTML += '<br><em>Escape로 입력 초기화</em>';
    }

    if (e.ctrlKey && e.key === 'Enter') {
        keyOutput.innerHTML += '<br><em>Ctrl+Enter 감지!</em>';
    }
});

// ============================================
// 9. 드래그 앤 드롭
// ============================================
let draggedElement = null;

// 드래그 가능한 요소들
document.querySelectorAll('.draggable').forEach(elem => {
    elem.addEventListener('dragstart', (e) => {
        draggedElement = e.target;
        e.target.style.opacity = '0.5';
    });

    elem.addEventListener('dragend', (e) => {
        e.target.style.opacity = '';
        draggedElement = null;
    });
});

// 드롭 존
document.querySelectorAll('.drop-zone').forEach(zone => {
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();  // 드롭 허용
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');

        if (draggedElement) {
            zone.appendChild(draggedElement);
        }
    });
});

// ============================================
// 10. 스크롤 이벤트
// ============================================
const scrollBox = document.getElementById('scrollBox');
const scrollPosition = document.getElementById('scrollPosition');

scrollBox.addEventListener('scroll', () => {
    scrollPosition.textContent = Math.round(scrollBox.scrollTop);
});

// 스크롤 방향 감지 (보너스)
let lastScrollTop = 0;
scrollBox.addEventListener('scroll', () => {
    const st = scrollBox.scrollTop;
    const direction = st > lastScrollTop ? '아래' : '위';
    // console.log(`스크롤 방향: ${direction}`);
    lastScrollTop = st;
});

// ============================================
// 페이지 로드 완료
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM 로드 완료!');
});

window.addEventListener('load', () => {
    console.log('모든 리소스 로드 완료!');
});
