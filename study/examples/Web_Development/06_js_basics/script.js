/*
 * JavaScript 기초 예제
 * 변수, 데이터 타입, 연산자, 조건문, 반복문, 함수, 배열, 객체, 클래스
 */

// 출력 함수
function log(message) {
    const output = document.getElementById('output');
    output.textContent += message + '\n';
    console.log(message);
}

function clearOutput() {
    document.getElementById('output').textContent = '';
}

// ============================================
// 1. 변수
// ============================================
function runVariables() {
    clearOutput();
    log('=== 변수 (var, let, const) ===\n');

    // var - 함수 스코프 (비권장)
    var oldVar = "I'm var";
    log(`var oldVar = "${oldVar}"`);

    // let - 블록 스코프, 재할당 가능
    let count = 0;
    log(`let count = ${count}`);
    count = 1;
    log(`count = ${count} (재할당 가능)`);

    // const - 블록 스코프, 재할당 불가
    const PI = 3.14159;
    log(`\nconst PI = ${PI}`);
    // PI = 3; // Error!

    const person = { name: "홍길동" };
    log(`const person = { name: "${person.name}" }`);
    person.name = "김철수";
    log(`person.name = "${person.name}" (객체 속성은 변경 가능)`);

    // 스코프 차이
    log('\n--- 스코프 차이 ---');
    if (true) {
        var varInBlock = "var는 블록 밖에서도 접근 가능";
        let letInBlock = "let은 블록 내에서만 접근 가능";
    }
    log(`블록 밖에서 var: ${varInBlock}`);
    // log(letInBlock); // Error: not defined
}

// ============================================
// 2. 데이터 타입
// ============================================
function runDataTypes() {
    clearOutput();
    log('=== 데이터 타입 ===\n');

    // 기본 타입
    let str = "문자열";
    let num = 42;
    let float = 3.14;
    let bool = true;
    let empty = null;
    let notDefined;
    let sym = Symbol("id");
    let big = 9007199254740991n;

    log('--- 기본 타입 (Primitive) ---');
    log(`String: "${str}" (typeof: ${typeof str})`);
    log(`Number (정수): ${num} (typeof: ${typeof num})`);
    log(`Number (실수): ${float} (typeof: ${typeof float})`);
    log(`Boolean: ${bool} (typeof: ${typeof bool})`);
    log(`Null: ${empty} (typeof: ${typeof empty})`); // 주의: object로 나옴
    log(`Undefined: ${notDefined} (typeof: ${typeof notDefined})`);
    log(`Symbol: ${String(sym)} (typeof: ${typeof sym})`);
    log(`BigInt: ${big} (typeof: ${typeof big})`);

    // 참조 타입
    log('\n--- 참조 타입 (Reference) ---');
    let obj = { key: "value" };
    let arr = [1, 2, 3];
    let func = function() {};

    log(`Object: ${JSON.stringify(obj)} (typeof: ${typeof obj})`);
    log(`Array: [${arr}] (typeof: ${typeof arr}, Array.isArray: ${Array.isArray(arr)})`);
    log(`Function: ${func.toString()} (typeof: ${typeof func})`);

    // 타입 변환
    log('\n--- 타입 변환 ---');
    log(`String(123) = "${String(123)}"`);
    log(`Number("42") = ${Number("42")}`);
    log(`Boolean(0) = ${Boolean(0)}`);
    log(`Boolean("") = ${Boolean("")}`);
    log(`Boolean("hello") = ${Boolean("hello")}`);
    log(`parseInt("42px") = ${parseInt("42px")}`);
    log(`parseFloat("3.14abc") = ${parseFloat("3.14abc")}`);
}

// ============================================
// 3. 연산자
// ============================================
function runOperators() {
    clearOutput();
    log('=== 연산자 ===\n');

    let a = 10, b = 3;

    // 산술 연산자
    log('--- 산술 연산자 ---');
    log(`${a} + ${b} = ${a + b}`);
    log(`${a} - ${b} = ${a - b}`);
    log(`${a} * ${b} = ${a * b}`);
    log(`${a} / ${b} = ${a / b}`);
    log(`${a} % ${b} = ${a % b} (나머지)`);
    log(`${a} ** ${b} = ${a ** b} (거듭제곱)`);

    // 비교 연산자
    log('\n--- 비교 연산자 ---');
    log(`5 == "5" : ${5 == "5"} (타입 변환 후 비교)`);
    log(`5 === "5" : ${5 === "5"} (타입까지 비교)`);
    log(`5 != "5" : ${5 != "5"}`);
    log(`5 !== "5" : ${5 !== "5"}`);

    // 논리 연산자
    log('\n--- 논리 연산자 ---');
    log(`true && false = ${true && false}`);
    log(`true || false = ${true || false}`);
    log(`!true = ${!true}`);

    // 삼항 연산자
    log('\n--- 삼항 연산자 ---');
    let age = 20;
    let status = age >= 18 ? "성인" : "미성년자";
    log(`나이: ${age}, 상태: ${status}`);

    // Nullish coalescing (??)
    log('\n--- Nullish Coalescing (??) ---');
    let value1 = null ?? "기본값";
    let value2 = 0 ?? "기본값";
    let value3 = "" ?? "기본값";
    log(`null ?? "기본값" = "${value1}"`);
    log(`0 ?? "기본값" = ${value2} (0은 null/undefined가 아님)`);
    log(`"" ?? "기본값" = "${value3}" (빈 문자열도 값)`);

    // || vs ?? 차이
    log('\n--- || vs ?? ---');
    log(`0 || "default" = "${0 || "default"}" (0은 falsy)`);
    log(`0 ?? "default" = ${0 ?? "default"} (0은 null이 아님)`);

    // Optional chaining (?.)
    log('\n--- Optional Chaining (?.) ---');
    let user = { profile: { name: "홍길동" } };
    let emptyUser = {};
    log(`user?.profile?.name = "${user?.profile?.name}"`);
    log(`emptyUser?.profile?.name = ${emptyUser?.profile?.name}`);
}

// ============================================
// 4. 조건문
// ============================================
function runConditions() {
    clearOutput();
    log('=== 조건문 ===\n');

    // if-else
    log('--- if-else ---');
    let score = 85;
    let grade;

    if (score >= 90) {
        grade = 'A';
    } else if (score >= 80) {
        grade = 'B';
    } else if (score >= 70) {
        grade = 'C';
    } else {
        grade = 'F';
    }
    log(`점수: ${score}, 학점: ${grade}`);

    // switch
    log('\n--- switch ---');
    let day = new Date().getDay();
    let dayName;

    switch (day) {
        case 0:
            dayName = "일요일";
            break;
        case 1:
            dayName = "월요일";
            break;
        case 2:
            dayName = "화요일";
            break;
        case 3:
            dayName = "수요일";
            break;
        case 4:
            dayName = "목요일";
            break;
        case 5:
            dayName = "금요일";
            break;
        case 6:
            dayName = "토요일";
            break;
        default:
            dayName = "알 수 없음";
    }
    log(`오늘은 ${dayName}입니다.`);

    // Truthy와 Falsy
    log('\n--- Truthy와 Falsy ---');
    const falsyValues = [false, 0, -0, 0n, "", null, undefined, NaN];
    log('Falsy 값들:');
    falsyValues.forEach(v => {
        log(`  ${String(v)} (${typeof v}) => ${Boolean(v) ? 'truthy' : 'falsy'}`);
    });

    log('\n모든 다른 값은 truthy입니다:');
    log(`  "0" => ${Boolean("0") ? 'truthy' : 'falsy'}`);
    log(`  [] => ${Boolean([]) ? 'truthy' : 'falsy'}`);
    log(`  {} => ${Boolean({}) ? 'truthy' : 'falsy'}`);
}

// ============================================
// 5. 반복문
// ============================================
function runLoops() {
    clearOutput();
    log('=== 반복문 ===\n');

    // for
    log('--- for 문 ---');
    let sum = 0;
    for (let i = 1; i <= 5; i++) {
        sum += i;
    }
    log(`1부터 5까지의 합: ${sum}`);

    // while
    log('\n--- while 문 ---');
    let count = 0;
    while (count < 3) {
        log(`count: ${count}`);
        count++;
    }

    // for...of (배열 순회)
    log('\n--- for...of (배열 순회) ---');
    const fruits = ["사과", "바나나", "오렌지"];
    for (const fruit of fruits) {
        log(`과일: ${fruit}`);
    }

    // for...in (객체 키 순회)
    log('\n--- for...in (객체 키 순회) ---');
    const person = { name: "홍길동", age: 25, city: "서울" };
    for (const key in person) {
        log(`${key}: ${person[key]}`);
    }

    // forEach
    log('\n--- forEach ---');
    fruits.forEach((fruit, index) => {
        log(`[${index}] ${fruit}`);
    });

    // break와 continue
    log('\n--- break와 continue ---');
    log('continue로 3 건너뛰기:');
    for (let i = 1; i <= 5; i++) {
        if (i === 3) continue;
        log(`  i = ${i}`);
    }

    log('break로 3에서 중단:');
    for (let i = 1; i <= 5; i++) {
        if (i === 3) break;
        log(`  i = ${i}`);
    }
}

// ============================================
// 6. 함수
// ============================================
function runFunctions() {
    clearOutput();
    log('=== 함수 ===\n');

    // 함수 선언식
    log('--- 함수 선언식 ---');
    function greet1(name) {
        return `Hello, ${name}!`;
    }
    log(greet1("홍길동"));

    // 함수 표현식
    log('\n--- 함수 표현식 ---');
    const greet2 = function(name) {
        return `Hi, ${name}!`;
    };
    log(greet2("김철수"));

    // 화살표 함수
    log('\n--- 화살표 함수 ---');
    const greet3 = (name) => `Hey, ${name}!`;
    log(greet3("이영희"));

    // 기본 매개변수
    log('\n--- 기본 매개변수 ---');
    const greet4 = (name = "Guest") => `Welcome, ${name}!`;
    log(greet4());
    log(greet4("박지민"));

    // Rest 파라미터
    log('\n--- Rest 파라미터 ---');
    const sumAll = (...numbers) => {
        return numbers.reduce((acc, num) => acc + num, 0);
    };
    log(`sumAll(1, 2, 3, 4, 5) = ${sumAll(1, 2, 3, 4, 5)}`);

    // 구조 분해 매개변수
    log('\n--- 구조 분해 매개변수 ---');
    const printPerson = ({ name, age }) => {
        return `${name}님은 ${age}세입니다.`;
    };
    log(printPerson({ name: "홍길동", age: 25 }));

    // 즉시 실행 함수 (IIFE)
    log('\n--- 즉시 실행 함수 (IIFE) ---');
    const result = (function(x, y) {
        return x + y;
    })(3, 4);
    log(`IIFE 결과: ${result}`);

    // 클로저
    log('\n--- 클로저 ---');
    function createCounter() {
        let count = 0;
        return {
            increment: () => ++count,
            decrement: () => --count,
            getCount: () => count
        };
    }
    const counter = createCounter();
    log(`count: ${counter.getCount()}`);
    log(`increment: ${counter.increment()}`);
    log(`increment: ${counter.increment()}`);
    log(`decrement: ${counter.decrement()}`);
}

// ============================================
// 7. 배열
// ============================================
function runArrays() {
    clearOutput();
    log('=== 배열 ===\n');

    // 배열 생성
    log('--- 배열 생성 ---');
    const arr1 = [1, 2, 3];
    const arr2 = new Array(3).fill(0);
    const arr3 = Array.from({ length: 5 }, (_, i) => i + 1);
    log(`리터럴: [${arr1}]`);
    log(`fill: [${arr2}]`);
    log(`Array.from: [${arr3}]`);

    // 기본 메서드
    log('\n--- 기본 메서드 ---');
    const fruits = ["사과", "바나나"];
    log(`원본: [${fruits}]`);

    fruits.push("오렌지");
    log(`push("오렌지"): [${fruits}]`);

    fruits.pop();
    log(`pop(): [${fruits}]`);

    fruits.unshift("딸기");
    log(`unshift("딸기"): [${fruits}]`);

    fruits.shift();
    log(`shift(): [${fruits}]`);

    // 배열 조작
    log('\n--- 배열 조작 ---');
    const numbers = [1, 2, 3, 4, 5];
    log(`원본: [${numbers}]`);
    log(`slice(1, 3): [${numbers.slice(1, 3)}]`);
    log(`concat([6, 7]): [${numbers.concat([6, 7])}]`);
    log(`indexOf(3): ${numbers.indexOf(3)}`);
    log(`includes(3): ${numbers.includes(3)}`);
    log(`join("-"): "${numbers.join("-")}"`);
    log(`reverse(): [${[...numbers].reverse()}]`);

    // 고차 함수
    log('\n--- 고차 함수 ---');
    const nums = [1, 2, 3, 4, 5];
    log(`원본: [${nums}]`);
    log(`map(n => n * 2): [${nums.map(n => n * 2)}]`);
    log(`filter(n => n > 2): [${nums.filter(n => n > 2)}]`);
    log(`find(n => n > 2): ${nums.find(n => n > 2)}`);
    log(`findIndex(n => n > 2): ${nums.findIndex(n => n > 2)}`);
    log(`reduce((acc, n) => acc + n, 0): ${nums.reduce((acc, n) => acc + n, 0)}`);
    log(`every(n => n > 0): ${nums.every(n => n > 0)}`);
    log(`some(n => n > 4): ${nums.some(n => n > 4)}`);
    log(`sort((a, b) => b - a): [${[...nums].sort((a, b) => b - a)}]`);

    // 스프레드 연산자
    log('\n--- 스프레드 연산자 ---');
    const arr = [1, 2, 3];
    const newArr = [...arr, 4, 5];
    log(`[...arr, 4, 5]: [${newArr}]`);
    log(`Math.max(...arr): ${Math.max(...arr)}`);

    // 구조 분해
    log('\n--- 구조 분해 ---');
    const [first, second, ...rest] = [1, 2, 3, 4, 5];
    log(`[first, second, ...rest] = [1, 2, 3, 4, 5]`);
    log(`first: ${first}, second: ${second}, rest: [${rest}]`);
}

// ============================================
// 8. 객체
// ============================================
function runObjects() {
    clearOutput();
    log('=== 객체 ===\n');

    // 객체 리터럴
    log('--- 객체 리터럴 ---');
    const person = {
        name: "홍길동",
        age: 25,
        city: "서울",
        greet() {
            return `안녕하세요, ${this.name}입니다.`;
        }
    };
    log(`person: ${JSON.stringify(person, null, 2)}`);
    log(`person.greet(): ${person.greet()}`);

    // 속성 접근
    log('\n--- 속성 접근 ---');
    log(`person.name: "${person.name}"`);
    log(`person["age"]: ${person["age"]}`);

    const key = "city";
    log(`person[key] (key="city"): "${person[key]}"`);

    // 속성 추가/수정/삭제
    log('\n--- 속성 추가/수정/삭제 ---');
    person.email = "hong@example.com";
    log(`추가: person.email = "${person.email}"`);

    person.age = 26;
    log(`수정: person.age = ${person.age}`);

    delete person.city;
    log(`삭제 후 keys: [${Object.keys(person)}]`);

    // 구조 분해 할당
    log('\n--- 구조 분해 할당 ---');
    const { name, age, email = "없음" } = person;
    log(`const { name, age, email = "없음" } = person`);
    log(`name: "${name}", age: ${age}, email: "${email}"`);

    // 별칭
    const { name: userName } = person;
    log(`const { name: userName } = person => userName: "${userName}"`);

    // 스프레드 연산자
    log('\n--- 스프레드 연산자 ---');
    const newPerson = { ...person, city: "부산", country: "한국" };
    log(`{ ...person, city: "부산" }:`);
    log(JSON.stringify(newPerson, null, 2));

    // Object 메서드
    log('\n--- Object 메서드 ---');
    const obj = { a: 1, b: 2, c: 3 };
    log(`Object.keys(obj): [${Object.keys(obj)}]`);
    log(`Object.values(obj): [${Object.values(obj)}]`);
    log(`Object.entries(obj): ${JSON.stringify(Object.entries(obj))}`);

    // Object.assign
    const merged = Object.assign({}, obj, { d: 4 });
    log(`Object.assign({}, obj, { d: 4 }): ${JSON.stringify(merged)}`);

    // 속성 존재 확인
    log('\n--- 속성 존재 확인 ---');
    log(`"a" in obj: ${"a" in obj}`);
    log(`obj.hasOwnProperty("a"): ${obj.hasOwnProperty("a")}`);
}

// ============================================
// 9. 클래스
// ============================================
function runClasses() {
    clearOutput();
    log('=== 클래스 (ES6) ===\n');

    // 기본 클래스
    log('--- 기본 클래스 ---');
    class Animal {
        constructor(name) {
            this.name = name;
        }

        speak() {
            return `${this.name} makes a sound.`;
        }
    }

    const animal = new Animal("동물");
    log(`new Animal("동물")`);
    log(`animal.speak(): "${animal.speak()}"`);

    // 상속
    log('\n--- 상속 ---');
    class Dog extends Animal {
        constructor(name, breed) {
            super(name);
            this.breed = breed;
        }

        speak() {
            return `${this.name}(${this.breed})가 멍멍 짖습니다!`;
        }

        fetch() {
            return `${this.name}가 공을 가져옵니다.`;
        }
    }

    const dog = new Dog("멍멍이", "진돗개");
    log(`new Dog("멍멍이", "진돗개")`);
    log(`dog.speak(): "${dog.speak()}"`);
    log(`dog.fetch(): "${dog.fetch()}"`);

    // Getter와 Setter
    log('\n--- Getter와 Setter ---');
    class Circle {
        constructor(radius) {
            this._radius = radius;
        }

        get radius() {
            return this._radius;
        }

        set radius(value) {
            if (value > 0) {
                this._radius = value;
            }
        }

        get area() {
            return Math.PI * this._radius ** 2;
        }
    }

    const circle = new Circle(5);
    log(`circle.radius: ${circle.radius}`);
    log(`circle.area: ${circle.area.toFixed(2)}`);
    circle.radius = 10;
    log(`circle.radius = 10 후 area: ${circle.area.toFixed(2)}`);

    // Static 메서드
    log('\n--- Static 메서드 ---');
    class MathUtil {
        static PI = 3.14159;

        static add(a, b) {
            return a + b;
        }

        static multiply(a, b) {
            return a * b;
        }
    }

    log(`MathUtil.PI: ${MathUtil.PI}`);
    log(`MathUtil.add(3, 4): ${MathUtil.add(3, 4)}`);
    log(`MathUtil.multiply(3, 4): ${MathUtil.multiply(3, 4)}`);

    // Private 필드 (# 문법, ES2022)
    log('\n--- Private 필드 ---');
    class BankAccount {
        #balance = 0;

        deposit(amount) {
            this.#balance += amount;
            return this.#balance;
        }

        getBalance() {
            return this.#balance;
        }
    }

    const account = new BankAccount();
    log(`account.deposit(1000): ${account.deposit(1000)}`);
    log(`account.deposit(500): ${account.deposit(500)}`);
    log(`account.getBalance(): ${account.getBalance()}`);
    // log(account.#balance); // Error: Private field
}

// 페이지 로드 시 안내 메시지
log('버튼을 클릭하여 각 섹션의 예제를 실행하세요.');
