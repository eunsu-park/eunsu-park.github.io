/*
 * 비동기 JavaScript 예제
 * - 콜백, Promise, async/await
 * - Fetch API
 * - 타이머 함수
 */

// ============================================
// 유틸리티 함수
// ============================================
function log(outputId, message, clear = false) {
    const output = document.getElementById(outputId);
    if (clear) {
        output.textContent = '';
    }
    output.textContent += message + '\n';
    output.scrollTop = output.scrollHeight;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// 1. 콜백 함수
// ============================================
function runCallback() {
    const output = 'callbackOutput';
    log(output, '=== 콜백 함수 실행 ===', true);
    log(output, '[시작] 데이터 요청...');

    function fetchData(callback) {
        setTimeout(() => {
            const data = { id: 1, name: '홍길동', email: 'hong@example.com' };
            callback(null, data);
        }, 1000);
    }

    fetchData((error, data) => {
        if (error) {
            log(output, `[에러] ${error.message}`);
        } else {
            log(output, `[완료] 데이터 수신: ${JSON.stringify(data, null, 2)}`);
        }
    });

    log(output, '[비동기] 다른 작업 수행 중...');
}

function runCallbackHell() {
    const output = 'callbackOutput';
    log(output, '=== 콜백 지옥 예시 ===', true);
    log(output, '사용자 정보 조회 시작...');

    // 콜백 지옥 (피라미드 형태)
    setTimeout(() => {
        log(output, '1. 사용자 ID 조회 완료: user_123');
        setTimeout(() => {
            log(output, '2. 사용자 프로필 조회 완료: { name: "홍길동" }');
            setTimeout(() => {
                log(output, '3. 사용자 게시글 조회 완료: 15개');
                setTimeout(() => {
                    log(output, '4. 게시글 댓글 조회 완료: 42개');
                    log(output, '--- 모든 작업 완료 ---');
                    log(output, '\n⚠️ 이런 중첩 구조는 가독성이 떨어집니다.');
                    log(output, '→ Promise나 async/await로 개선하세요!');
                }, 500);
            }, 500);
        }, 500);
    }, 500);
}

// ============================================
// 2. Promise
// ============================================
function runPromise() {
    const output = 'promiseOutput';
    log(output, '=== Promise 기본 ===', true);

    const promise = new Promise((resolve, reject) => {
        log(output, '작업 시작...');
        setTimeout(() => {
            const success = Math.random() > 0.3;
            if (success) {
                resolve({ status: 'success', data: '처리 완료!' });
            } else {
                reject(new Error('처리 실패'));
            }
        }, 1000);
    });

    promise
        .then(result => {
            log(output, `✅ 성공: ${JSON.stringify(result)}`);
        })
        .catch(error => {
            log(output, `❌ 실패: ${error.message}`);
        })
        .finally(() => {
            log(output, '🏁 Promise 완료 (성공/실패 무관)');
        });
}

function runPromiseChain() {
    const output = 'promiseOutput';
    log(output, '=== Promise 체이닝 ===', true);

    function step1() {
        return new Promise(resolve => {
            setTimeout(() => resolve(1), 300);
        });
    }

    function step2(prev) {
        return new Promise(resolve => {
            setTimeout(() => resolve(prev + 10), 300);
        });
    }

    function step3(prev) {
        return new Promise(resolve => {
            setTimeout(() => resolve(prev * 2), 300);
        });
    }

    log(output, '체이닝 시작...');

    step1()
        .then(result => {
            log(output, `Step 1: ${result}`);
            return step2(result);
        })
        .then(result => {
            log(output, `Step 2: ${result}`);
            return step3(result);
        })
        .then(result => {
            log(output, `Step 3: ${result}`);
            log(output, `최종 결과: ${result}`);
        });
}

function runPromiseError() {
    const output = 'promiseOutput';
    log(output, '=== Promise 에러 처리 ===', true);

    function riskyOperation(shouldFail) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (shouldFail) {
                    reject(new Error('의도적 에러 발생!'));
                } else {
                    resolve('성공적으로 처리됨');
                }
            }, 500);
        });
    }

    log(output, '에러 발생 시나리오 실행...');

    riskyOperation(true)
        .then(result => {
            log(output, `결과: ${result}`);
        })
        .catch(error => {
            log(output, `❌ 에러 캐치: ${error.message}`);
            log(output, '→ 에러가 catch 블록에서 처리됨');
            return '에러 복구됨';
        })
        .then(result => {
            log(output, `복구 후 진행: ${result}`);
        });
}

// ============================================
// 3. async/await
// ============================================
async function runAsyncAwait() {
    const output = 'asyncOutput';
    log(output, '=== async/await 기본 ===', true);

    async function fetchUserData() {
        log(output, '사용자 데이터 요청 중...');
        await delay(500);
        return { id: 1, name: '김철수' };
    }

    async function fetchUserPosts(userId) {
        log(output, `사용자 ${userId}의 게시글 요청 중...`);
        await delay(500);
        return ['게시글 1', '게시글 2', '게시글 3'];
    }

    try {
        const user = await fetchUserData();
        log(output, `사용자: ${JSON.stringify(user)}`);

        const posts = await fetchUserPosts(user.id);
        log(output, `게시글: ${posts.join(', ')}`);

        log(output, '✅ 모든 작업 완료!');
    } catch (error) {
        log(output, `❌ 에러: ${error.message}`);
    }
}

async function runAsyncError() {
    const output = 'asyncOutput';
    log(output, '=== async/await 에러 처리 ===', true);

    async function unstableOperation() {
        await delay(500);
        throw new Error('네트워크 연결 실패');
    }

    log(output, '불안정한 작업 시도...');

    try {
        await unstableOperation();
        log(output, '작업 성공');
    } catch (error) {
        log(output, `❌ try-catch로 에러 캐치: ${error.message}`);
        log(output, '→ async/await에서는 try-catch 사용');
    } finally {
        log(output, '🏁 finally 블록 실행');
    }
}

// ============================================
// 4. Promise.all, Promise.race, Promise.allSettled
// ============================================
async function runPromiseAll() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.all ===', true);
    log(output, '3개의 작업을 병렬로 실행합니다...');

    const startTime = Date.now();

    const task1 = delay(1000).then(() => '작업 1 완료');
    const task2 = delay(1500).then(() => '작업 2 완료');
    const task3 = delay(800).then(() => '작업 3 완료');

    try {
        const results = await Promise.all([task1, task2, task3]);
        const elapsed = Date.now() - startTime;

        log(output, `\n결과: ${results.join(', ')}`);
        log(output, `총 소요 시간: ${elapsed}ms`);
        log(output, '\n💡 Promise.all은 모든 Promise가 완료될 때까지 대기');
        log(output, '→ 가장 오래 걸리는 작업(1500ms) 기준으로 완료');
    } catch (error) {
        log(output, `에러: ${error.message}`);
    }
}

async function runPromiseRace() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.race ===', true);
    log(output, '가장 빠른 작업의 결과를 반환합니다...');

    const startTime = Date.now();

    const slow = delay(2000).then(() => '느린 작업');
    const medium = delay(1000).then(() => '중간 작업');
    const fast = delay(500).then(() => '빠른 작업');

    const winner = await Promise.race([slow, medium, fast]);
    const elapsed = Date.now() - startTime;

    log(output, `\n🏆 승자: ${winner}`);
    log(output, `소요 시간: ${elapsed}ms`);
    log(output, '\n💡 Promise.race는 가장 먼저 완료되는 Promise 반환');
}

async function runPromiseAllSettled() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.allSettled ===', true);
    log(output, '성공/실패 상관없이 모든 결과를 수집합니다...');

    const tasks = [
        delay(500).then(() => '성공 1'),
        delay(300).then(() => { throw new Error('실패 1'); }),
        delay(400).then(() => '성공 2'),
        delay(600).then(() => { throw new Error('실패 2'); }),
    ];

    const results = await Promise.allSettled(tasks);

    log(output, '\n결과:');
    results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
            log(output, `  ${index + 1}. ✅ ${result.value}`);
        } else {
            log(output, `  ${index + 1}. ❌ ${result.reason.message}`);
        }
    });

    log(output, '\n💡 Promise.allSettled는 모든 Promise의 결과를 수집');
    log(output, '→ 일부가 실패해도 나머지 결과를 확인 가능');
}

// ============================================
// 5. Fetch API
// ============================================
async function fetchUsers() {
    const output = 'fetchOutput';
    const btn = document.getElementById('fetchUsersBtn');

    log(output, '=== 사용자 목록 조회 ===', true);
    btn.disabled = true;
    btn.innerHTML = '로딩 중... <span class="loading"></span>';

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/users?_limit=5');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const users = await response.json();

        log(output, `${users.length}명의 사용자를 찾았습니다:\n`);

        users.forEach(user => {
            log(output, `👤 ${user.name}`);
            log(output, `   이메일: ${user.email}`);
            log(output, `   회사: ${user.company.name}`);
            log(output, '');
        });
    } catch (error) {
        log(output, `❌ 에러: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = '사용자 목록 가져오기';
    }
}

async function fetchPost() {
    const output = 'fetchOutput';
    log(output, '=== 게시글 조회 ===', true);

    const postId = Math.floor(Math.random() * 100) + 1;
    log(output, `게시글 #${postId} 요청 중...`);

    try {
        const response = await fetch(`https://jsonplaceholder.typicode.com/posts/${postId}`);
        const post = await response.json();

        log(output, `\n📄 제목: ${post.title}`);
        log(output, `\n본문:\n${post.body}`);
    } catch (error) {
        log(output, `❌ 에러: ${error.message}`);
    }
}

// ============================================
// 6. POST 요청
// ============================================
async function createPost() {
    const output = 'postOutput';
    const title = document.getElementById('postTitle').value || '제목 없음';
    const body = document.getElementById('postBody').value || '내용 없음';

    log(output, '=== POST 요청 ===', true);
    log(output, '게시글 생성 중...');

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/posts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                body: body,
                userId: 1,
            }),
        });

        const data = await response.json();

        log(output, '\n✅ 게시글 생성 성공!');
        log(output, `응답 데이터:`);
        log(output, JSON.stringify(data, null, 2));
    } catch (error) {
        log(output, `❌ 에러: ${error.message}`);
    }
}

// ============================================
// 7. 에러 처리 패턴
// ============================================
async function testErrorHandling() {
    const output = 'errorOutput';
    log(output, '=== 에러 처리 패턴 ===', true);

    // 패턴 1: try-catch
    log(output, '1. try-catch 패턴:');
    try {
        const result = await fetch('https://jsonplaceholder.typicode.com/posts/1');
        const data = await result.json();
        log(output, `   ✅ 성공: ${data.title.substring(0, 20)}...`);
    } catch (error) {
        log(output, `   ❌ 실패: ${error.message}`);
    }

    // 패턴 2: 에러 래핑
    log(output, '\n2. 에러 래핑 패턴:');

    async function safeFetch(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return { data: await response.json(), error: null };
        } catch (error) {
            return { data: null, error: error.message };
        }
    }

    const { data, error } = await safeFetch('https://jsonplaceholder.typicode.com/posts/1');
    if (error) {
        log(output, `   ❌ 에러: ${error}`);
    } else {
        log(output, `   ✅ 데이터: ${data.title.substring(0, 20)}...`);
    }

    log(output, '\n💡 에러 래핑 패턴은 Go 언어 스타일');
    log(output, '→ 명시적인 에러 처리가 가능');
}

async function testNetworkError() {
    const output = 'errorOutput';
    log(output, '=== 네트워크 에러 테스트 ===', true);
    log(output, '존재하지 않는 URL에 요청...\n');

    try {
        // 타임아웃 구현
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch('https://invalid-url-that-does-not-exist.com/api', {
            signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const data = await response.json();
        log(output, `성공: ${data}`);
    } catch (error) {
        if (error.name === 'AbortError') {
            log(output, '❌ 타임아웃: 3초 내 응답 없음');
        } else {
            log(output, `❌ 네트워크 에러: ${error.message}`);
        }
        log(output, '\n💡 실제 앱에서는 재시도 로직 추가 권장');
    }
}

// ============================================
// 8. 사용자 카드 로드
// ============================================
async function loadUserCards() {
    const container = document.getElementById('userCards');
    const btn = document.getElementById('loadCardsBtn');

    container.innerHTML = '<p>로딩 중...</p>';
    btn.disabled = true;

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/users?_limit=6');
        const users = await response.json();

        container.innerHTML = users.map(user => `
            <div class="user-card">
                <img src="https://i.pravatar.cc/80?u=${user.id}" alt="${user.name}">
                <h4>${user.name}</h4>
                <p>@${user.username}</p>
                <p>${user.email}</p>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<p style="color: red;">에러: ${error.message}</p>`;
    } finally {
        btn.disabled = false;
    }
}

// ============================================
// 9. setTimeout, setInterval
// ============================================
let timerCount = 0;
let timerInterval = null;

function startTimer() {
    const output = document.getElementById('timerOutput');

    if (timerInterval) {
        output.textContent = '타이머가 이미 실행 중입니다.';
        return;
    }

    timerCount = 0;
    output.textContent = `타이머: ${timerCount}`;

    timerInterval = setInterval(() => {
        timerCount++;
        output.textContent = `타이머: ${timerCount}초`;

        if (timerCount >= 60) {
            stopTimer();
            output.textContent += ' (최대 시간 도달)';
        }
    }, 1000);
}

function stopTimer() {
    const output = document.getElementById('timerOutput');

    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        output.textContent = `타이머 중지: ${timerCount}초에서 멈춤`;
    } else {
        output.textContent = '실행 중인 타이머가 없습니다.';
    }
}

// Debounce 구현
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function runDebounceDemo() {
    const output = document.getElementById('timerOutput');
    let callCount = 0;

    const debouncedLog = debounce(() => {
        output.textContent = `Debounce 완료! 실제 실행 횟수: 1회 (시도: ${callCount}회)`;
    }, 500);

    output.textContent = '빠른 호출 시뮬레이션 중...';

    // 100ms 간격으로 10번 호출 시도
    for (let i = 0; i < 10; i++) {
        setTimeout(() => {
            callCount++;
            debouncedLog();
        }, i * 100);
    }

    setTimeout(() => {
        output.textContent += '\n💡 Debounce: 마지막 호출 후 500ms 대기 후 실행';
    }, 2000);
}
