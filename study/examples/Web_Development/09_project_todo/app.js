/*
 * Todo App
 * 기능: 추가, 삭제, 수정, 완료, 필터링, 로컬스토리지 저장
 */

// ============================================
// State
// ============================================
let todos = [];
let currentFilter = 'all';

// ============================================
// DOM Elements
// ============================================
const todoInput = document.getElementById('todoInput');
const addBtn = document.getElementById('addBtn');
const todoList = document.getElementById('todoList');
const todoCount = document.getElementById('todoCount');
const clearCompletedBtn = document.getElementById('clearCompleted');
const filterBtns = document.querySelectorAll('.filter-btn');
const currentDateEl = document.getElementById('currentDate');

// ============================================
// Initialize
// ============================================
function init() {
    // 날짜 표시
    displayCurrentDate();

    // 로컬 스토리지에서 데이터 로드
    loadTodos();

    // 이벤트 리스너 등록
    addEventListeners();

    // 초기 렌더링
    render();
}

function displayCurrentDate() {
    const now = new Date();
    const options = {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long'
    };
    currentDateEl.textContent = now.toLocaleDateString('ko-KR', options);
}

// ============================================
// Event Listeners
// ============================================
function addEventListeners() {
    // 추가 버튼 클릭
    addBtn.addEventListener('click', addTodo);

    // Enter 키로 추가
    todoInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addTodo();
        }
    });

    // 완료 항목 삭제
    clearCompletedBtn.addEventListener('click', clearCompleted);

    // 필터 버튼
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            setFilter(btn.dataset.filter);
        });
    });

    // Todo 리스트 이벤트 위임
    todoList.addEventListener('click', handleTodoClick);
    todoList.addEventListener('change', handleTodoChange);
}

// ============================================
// Todo CRUD Operations
// ============================================
function addTodo() {
    const text = todoInput.value.trim();

    if (!text) {
        todoInput.focus();
        return;
    }

    const newTodo = {
        id: Date.now(),
        text: text,
        completed: false,
        createdAt: new Date().toISOString()
    };

    todos.unshift(newTodo);
    saveTodos();
    render();

    todoInput.value = '';
    todoInput.focus();
}

function deleteTodo(id) {
    todos = todos.filter(todo => todo.id !== id);
    saveTodos();
    render();
}

function toggleTodo(id) {
    todos = todos.map(todo =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
    );
    saveTodos();
    render();
}

function editTodo(id) {
    const todoItem = document.querySelector(`[data-id="${id}"]`);
    const todo = todos.find(t => t.id === id);

    if (!todoItem || !todo) return;

    // 편집 모드로 변경
    todoItem.innerHTML = `
        <input type="checkbox" ${todo.completed ? 'checked' : ''} disabled>
        <input type="text" class="edit-input" value="${escapeHtml(todo.text)}">
        <div class="todo-actions" style="opacity: 1;">
            <button class="save-btn" data-action="save">저장</button>
            <button class="cancel-btn" data-action="cancel">취소</button>
        </div>
    `;

    const editInput = todoItem.querySelector('.edit-input');
    editInput.focus();
    editInput.select();

    // Enter 키로 저장
    editInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveTodoEdit(id, editInput.value);
        }
    });

    // Escape 키로 취소
    editInput.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            render();
        }
    });
}

function saveTodoEdit(id, newText) {
    const text = newText.trim();

    if (!text) {
        render();
        return;
    }

    todos = todos.map(todo =>
        todo.id === id ? { ...todo, text: text } : todo
    );
    saveTodos();
    render();
}

function clearCompleted() {
    todos = todos.filter(todo => !todo.completed);
    saveTodos();
    render();
}

// ============================================
// Event Handlers
// ============================================
function handleTodoClick(e) {
    const todoItem = e.target.closest('.todo-item');
    if (!todoItem) return;

    const id = parseInt(todoItem.dataset.id);
    const action = e.target.dataset.action;

    switch (action) {
        case 'delete':
            deleteTodo(id);
            break;
        case 'edit':
            editTodo(id);
            break;
        case 'save':
            const editInput = todoItem.querySelector('.edit-input');
            if (editInput) {
                saveTodoEdit(id, editInput.value);
            }
            break;
        case 'cancel':
            render();
            break;
    }
}

function handleTodoChange(e) {
    if (e.target.type === 'checkbox') {
        const todoItem = e.target.closest('.todo-item');
        if (todoItem) {
            const id = parseInt(todoItem.dataset.id);
            toggleTodo(id);
        }
    }
}

// ============================================
// Filter
// ============================================
function setFilter(filter) {
    currentFilter = filter;

    // 버튼 활성화 상태 업데이트
    filterBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });

    render();
}

function getFilteredTodos() {
    switch (currentFilter) {
        case 'active':
            return todos.filter(todo => !todo.completed);
        case 'completed':
            return todos.filter(todo => todo.completed);
        default:
            return todos;
    }
}

// ============================================
// Render
// ============================================
function render() {
    const filteredTodos = getFilteredTodos();

    if (filteredTodos.length === 0) {
        todoList.innerHTML = `
            <li class="empty-state">
                <p>${getEmptyMessage()}</p>
            </li>
        `;
    } else {
        todoList.innerHTML = filteredTodos.map(todo => `
            <li class="todo-item ${todo.completed ? 'completed' : ''}" data-id="${todo.id}">
                <input type="checkbox" ${todo.completed ? 'checked' : ''}>
                <span class="todo-text">${escapeHtml(todo.text)}</span>
                <div class="todo-actions">
                    <button class="edit-btn" data-action="edit">편집</button>
                    <button class="delete-btn" data-action="delete">삭제</button>
                </div>
            </li>
        `).join('');
    }

    updateCounter();
}

function getEmptyMessage() {
    switch (currentFilter) {
        case 'active':
            return '진행중인 할 일이 없습니다! 🎉';
        case 'completed':
            return '완료된 할 일이 없습니다.';
        default:
            return '할 일을 추가해보세요! ✏️';
    }
}

function updateCounter() {
    const activeCount = todos.filter(todo => !todo.completed).length;
    const totalCount = todos.length;
    todoCount.textContent = `${activeCount}개 항목 남음 (전체 ${totalCount}개)`;
}

// ============================================
// Local Storage
// ============================================
function saveTodos() {
    localStorage.setItem('todos', JSON.stringify(todos));
}

function loadTodos() {
    const stored = localStorage.getItem('todos');
    if (stored) {
        try {
            todos = JSON.parse(stored);
        } catch (e) {
            console.error('Failed to load todos:', e);
            todos = [];
        }
    }
}

// ============================================
// Utility
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Start App
// ============================================
init();
