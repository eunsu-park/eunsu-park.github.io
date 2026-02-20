/**
 * TypeScript 인터페이스와 제네릭
 * - 인터페이스
 * - 클래스와 타입
 * - 제네릭
 * - 유틸리티 타입
 */

// ============================================
// 1. 인터페이스 기초
// ============================================

interface User {
    id: number;
    name: string;
    email: string;
    age?: number;  // 선택적 속성
    readonly createdAt: Date;  // 읽기 전용
}

const user1: User = {
    id: 1,
    name: "홍길동",
    email: "hong@example.com",
    createdAt: new Date()
};

// user1.createdAt = new Date();  // 에러: readonly

console.log("=== 인터페이스 기초 ===");
console.log(`사용자: ${user1.name} (${user1.email})`);

// ============================================
// 2. 인터페이스 확장
// ============================================

interface Person {
    name: string;
    age: number;
}

interface Employee extends Person {
    employeeId: string;
    department: string;
}

interface Manager extends Employee {
    teamSize: number;
    reports: Employee[];
}

const manager: Manager = {
    name: "김부장",
    age: 45,
    employeeId: "E001",
    department: "개발팀",
    teamSize: 5,
    reports: []
};

console.log("\n=== 인터페이스 확장 ===");
console.log(`매니저: ${manager.name}, ${manager.department}, 팀원 ${manager.teamSize}명`);

// ============================================
// 3. 인터페이스 병합
// ============================================

interface Config {
    apiUrl: string;
}

interface Config {
    timeout: number;
}

// 두 선언이 자동으로 병합됨
const config: Config = {
    apiUrl: "https://api.example.com",
    timeout: 5000
};

console.log("\n=== 인터페이스 병합 ===");
console.log(`Config: ${config.apiUrl}, timeout: ${config.timeout}ms`);

// ============================================
// 4. 함수 인터페이스
// ============================================

interface MathFunc {
    (x: number, y: number): number;
}

interface Calculator {
    add: MathFunc;
    subtract: MathFunc;
    multiply: MathFunc;
    divide: MathFunc;
}

const calculator: Calculator = {
    add: (x, y) => x + y,
    subtract: (x, y) => x - y,
    multiply: (x, y) => x * y,
    divide: (x, y) => x / y
};

console.log("\n=== 함수 인터페이스 ===");
console.log(`10 + 5 = ${calculator.add(10, 5)}`);
console.log(`10 - 5 = ${calculator.subtract(10, 5)}`);

// ============================================
// 5. 인덱스 시그니처
// ============================================

interface StringDictionary {
    [key: string]: string;
}

interface NumberArray {
    [index: number]: string;
}

const translations: StringDictionary = {
    hello: "안녕하세요",
    goodbye: "안녕히 가세요",
    thanks: "감사합니다"
};

const colors: NumberArray = ["red", "green", "blue"];

console.log("\n=== 인덱스 시그니처 ===");
console.log(`hello = ${translations["hello"]}`);
console.log(`colors[0] = ${colors[0]}`);

// ============================================
// 6. 클래스와 인터페이스
// ============================================

interface Animal {
    name: string;
    makeSound(): void;
}

interface Movable {
    move(distance: number): void;
}

class Dog implements Animal, Movable {
    constructor(public name: string) {}

    makeSound(): void {
        console.log(`${this.name}: 멍멍!`);
    }

    move(distance: number): void {
        console.log(`${this.name}이(가) ${distance}m 이동했습니다.`);
    }
}

console.log("\n=== 클래스와 인터페이스 ===");
const dog = new Dog("바둑이");
dog.makeSound();
dog.move(10);

// ============================================
// 7. 제네릭 기초
// ============================================

// 제네릭 함수
function identity<T>(arg: T): T {
    return arg;
}

// 제네릭 배열 함수
function firstElement<T>(arr: T[]): T | undefined {
    return arr[0];
}

// 제네릭 객체 함수
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

console.log("\n=== 제네릭 기초 ===");
console.log(`identity("hello"): ${identity("hello")}`);
console.log(`identity(42): ${identity(42)}`);
console.log(`firstElement([1,2,3]): ${firstElement([1, 2, 3])}`);

const person = { name: "홍길동", age: 25 };
console.log(`getProperty: ${getProperty(person, "name")}`);

// ============================================
// 8. 제네릭 인터페이스
// ============================================

interface ApiResponse<T> {
    data: T;
    status: number;
    message: string;
}

interface UserData {
    id: number;
    name: string;
}

interface ProductData {
    id: number;
    title: string;
    price: number;
}

const userResponse: ApiResponse<UserData> = {
    data: { id: 1, name: "홍길동" },
    status: 200,
    message: "Success"
};

const productResponse: ApiResponse<ProductData> = {
    data: { id: 1, title: "노트북", price: 1000000 },
    status: 200,
    message: "Success"
};

console.log("\n=== 제네릭 인터페이스 ===");
console.log(`User: ${userResponse.data.name}`);
console.log(`Product: ${productResponse.data.title}`);

// ============================================
// 9. 제네릭 클래스
// ============================================

class Stack<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }

    peek(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }
}

console.log("\n=== 제네릭 클래스 ===");
const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
numberStack.push(3);
console.log(`Stack peek: ${numberStack.peek()}`);
console.log(`Stack pop: ${numberStack.pop()}`);
console.log(`Stack size: ${numberStack.size()}`);

// ============================================
// 10. 제네릭 제약 조건
// ============================================

interface HasLength {
    length: number;
}

function logLength<T extends HasLength>(arg: T): number {
    console.log(`길이: ${arg.length}`);
    return arg.length;
}

console.log("\n=== 제네릭 제약 조건 ===");
logLength("Hello");       // string은 length 있음
logLength([1, 2, 3, 4]);  // array도 length 있음
logLength({ length: 10 }); // 객체도 가능

// ============================================
// 11. 유틸리티 타입
// ============================================

interface Todo {
    title: string;
    description: string;
    completed: boolean;
    createdAt: Date;
}

// Partial<T>: 모든 속성을 선택적으로
type PartialTodo = Partial<Todo>;

// Required<T>: 모든 속성을 필수로
type RequiredTodo = Required<Todo>;

// Readonly<T>: 모든 속성을 읽기 전용으로
type ReadonlyTodo = Readonly<Todo>;

// Pick<T, K>: 특정 속성만 선택
type TodoPreview = Pick<Todo, "title" | "completed">;

// Omit<T, K>: 특정 속성 제외
type TodoWithoutDate = Omit<Todo, "createdAt">;

// Record<K, T>: 키-값 타입 정의
type PageInfo = Record<"home" | "about" | "contact", { title: string }>;

const pages: PageInfo = {
    home: { title: "홈" },
    about: { title: "소개" },
    contact: { title: "연락처" }
};

console.log("\n=== 유틸리티 타입 ===");

const partialTodo: PartialTodo = {
    title: "일부만"
};

const todoPreview: TodoPreview = {
    title: "미리보기",
    completed: false
};

console.log(`PartialTodo: ${JSON.stringify(partialTodo)}`);
console.log(`TodoPreview: ${JSON.stringify(todoPreview)}`);
console.log(`Pages: ${Object.keys(pages).join(", ")}`);

// ============================================
// 12. 조건부 타입
// ============================================

type IsString<T> = T extends string ? "yes" : "no";

type A = IsString<string>;   // "yes"
type B = IsString<number>;   // "no"

// infer 키워드
type GetReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type FuncReturn = GetReturnType<() => number>;  // number
type AsyncReturn = GetReturnType<() => Promise<string>>;  // Promise<string>

// Exclude, Extract
type T1 = Exclude<"a" | "b" | "c", "a">;  // "b" | "c"
type T2 = Extract<"a" | "b" | "c", "a" | "f">;  // "a"

// NonNullable
type T3 = NonNullable<string | null | undefined>;  // string

console.log("\n=== 조건부 타입 ===");
console.log("IsString<string> = 'yes'");
console.log("IsString<number> = 'no'");
console.log("Exclude<'a'|'b'|'c', 'a'> = 'b' | 'c'");

// ============================================
// 13. 매핑된 타입
// ============================================

type Nullable<T> = {
    [P in keyof T]: T[P] | null;
};

type Optional<T> = {
    [P in keyof T]?: T[P];
};

type Getters<T> = {
    [P in keyof T as `get${Capitalize<string & P>}`]: () => T[P];
};

interface Point {
    x: number;
    y: number;
}

type NullablePoint = Nullable<Point>;  // { x: number | null; y: number | null }
type PointGetters = Getters<Point>;    // { getX: () => number; getY: () => number }

console.log("\n=== 매핑된 타입 ===");
const nullablePoint: NullablePoint = { x: 10, y: null };
console.log(`NullablePoint: x=${nullablePoint.x}, y=${nullablePoint.y}`);

// ============================================
// 14. 템플릿 리터럴 타입
// ============================================

type EventName = "click" | "scroll" | "mousemove";
type Handler = `on${Capitalize<EventName>}`;  // "onClick" | "onScroll" | "onMousemove"

type Greeting = `Hello, ${string}!`;

const greeting: Greeting = "Hello, World!";

console.log("\n=== 템플릿 리터럴 타입 ===");
console.log(`Greeting: ${greeting}`);
console.log("Handler 타입: 'onClick' | 'onScroll' | 'onMousemove'");

// ============================================
// 실행 예시
// ============================================
console.log("\n=== TypeScript 인터페이스/제네릭 완료 ===");
console.log("컴파일: npx tsc interfaces.ts");
console.log("실행: node interfaces.js");
