/**
 * TypeScript 기초
 * - 기본 타입
 * - 타입 추론
 * - 유니온/리터럴 타입
 * - 타입 가드
 */

// ============================================
// 1. 기본 타입
// ============================================

// 원시 타입
const name: string = "홍길동";
const age: number = 25;
const isActive: boolean = true;
const nothing: null = null;
const notDefined: undefined = undefined;

// 배열
const numbers: number[] = [1, 2, 3, 4, 5];
const names: Array<string> = ["Alice", "Bob", "Charlie"];

// 튜플 (고정된 길이와 타입)
const person: [string, number] = ["홍길동", 25];
const rgb: [number, number, number] = [255, 128, 0];

// 객체
const user: { name: string; age: number; email?: string } = {
    name: "김철수",
    age: 30,
    // email은 선택적(optional)
};

console.log("=== 기본 타입 ===");
console.log(`이름: ${name}, 나이: ${age}, 활성: ${isActive}`);
console.log(`숫자 배열: ${numbers.join(", ")}`);
console.log(`튜플: ${person[0]}(${person[1]}세)`);

// ============================================
// 2. 타입 추론 (Type Inference)
// ============================================

// TypeScript가 타입을 자동으로 추론
let inferredString = "Hello";  // string으로 추론
let inferredNumber = 42;       // number로 추론
let inferredArray = [1, 2, 3]; // number[]로 추론

// 함수 반환 타입 추론
function add(a: number, b: number) {
    return a + b;  // number 반환으로 추론
}

const sum = add(10, 20);  // sum은 number로 추론

console.log("\n=== 타입 추론 ===");
console.log(`추론된 합계: ${sum}`);

// ============================================
// 3. 유니온 타입 (Union Types)
// ============================================

// 여러 타입 중 하나
type StringOrNumber = string | number;

function printId(id: StringOrNumber) {
    console.log(`ID: ${id}`);

    // 타입에 따른 처리
    if (typeof id === "string") {
        console.log(`  (문자열, 길이: ${id.length})`);
    } else {
        console.log(`  (숫자, 2배: ${id * 2})`);
    }
}

console.log("\n=== 유니온 타입 ===");
printId("user_123");
printId(42);

// ============================================
// 4. 리터럴 타입 (Literal Types)
// ============================================

// 특정 값만 허용
type Direction = "up" | "down" | "left" | "right";
type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

function move(direction: Direction) {
    console.log(`Moving ${direction}`);
}

function rollDice(): DiceValue {
    return Math.ceil(Math.random() * 6) as DiceValue;
}

console.log("\n=== 리터럴 타입 ===");
move("up");
move("left");
console.log(`주사위: ${rollDice()}`);

// ============================================
// 5. any, unknown, never
// ============================================

// any: 모든 타입 허용 (타입 체크 비활성화, 사용 자제)
let anything: any = "hello";
anything = 42;
anything = true;

// unknown: 모든 타입 허용, 사용 시 타입 체크 필요
let unknownValue: unknown = "hello";

// unknown은 바로 사용 불가, 타입 확인 필요
if (typeof unknownValue === "string") {
    console.log(`unknown 값: ${unknownValue.toUpperCase()}`);
}

// never: 절대 발생하지 않는 타입
function throwError(message: string): never {
    throw new Error(message);
}

function infiniteLoop(): never {
    while (true) {
        // 무한 루프
    }
}

console.log("\n=== any, unknown ===");
console.log(`any 값: ${anything}`);

// ============================================
// 6. 타입 별칭 (Type Alias)
// ============================================

type Point = {
    x: number;
    y: number;
};

type ID = string | number;

type UserRole = "admin" | "editor" | "viewer";

type UserProfile = {
    id: ID;
    name: string;
    role: UserRole;
    location: Point;
};

const admin: UserProfile = {
    id: "admin_001",
    name: "관리자",
    role: "admin",
    location: { x: 0, y: 0 }
};

console.log("\n=== 타입 별칭 ===");
console.log(`사용자: ${admin.name} (${admin.role})`);

// ============================================
// 7. 타입 가드 (Type Guards)
// ============================================

type Circle = { kind: "circle"; radius: number };
type Rectangle = { kind: "rectangle"; width: number; height: number };
type Shape = Circle | Rectangle;

// 판별 유니온 (Discriminated Union)
function getArea(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
    }
}

// in 연산자로 타입 가드
function printShape(shape: Shape) {
    if ("radius" in shape) {
        console.log(`원: 반지름 ${shape.radius}`);
    } else {
        console.log(`사각형: ${shape.width} x ${shape.height}`);
    }
}

console.log("\n=== 타입 가드 ===");
const circle: Circle = { kind: "circle", radius: 5 };
const rect: Rectangle = { kind: "rectangle", width: 10, height: 20 };

console.log(`원 면적: ${getArea(circle).toFixed(2)}`);
console.log(`사각형 면적: ${getArea(rect)}`);
printShape(circle);
printShape(rect);

// ============================================
// 8. 타입 단언 (Type Assertion)
// ============================================

// as 키워드로 타입 단언
const input = document.getElementById("myInput") as HTMLInputElement;
// 또는 angle-bracket 문법 (JSX에서 사용 불가)
// const input = <HTMLInputElement>document.getElementById("myInput");

// 주의: 타입 단언은 런타임에 영향 없음
const maybeString: unknown = "hello world";
const strLength = (maybeString as string).length;

console.log("\n=== 타입 단언 ===");
console.log(`문자열 길이: ${strLength}`);

// ============================================
// 9. 함수 타입
// ============================================

// 함수 타입 정의
type MathOperation = (a: number, b: number) => number;

const multiply: MathOperation = (a, b) => a * b;
const divide: MathOperation = (a, b) => a / b;

// 선택적 매개변수와 기본값
function greet(name: string, greeting: string = "안녕하세요"): string {
    return `${greeting}, ${name}님!`;
}

// 나머지 매개변수
function sumAll(...numbers: number[]): number {
    return numbers.reduce((acc, cur) => acc + cur, 0);
}

// 오버로드 시그니처
function format(value: string): string;
function format(value: number): string;
function format(value: string | number): string {
    if (typeof value === "string") {
        return value.toUpperCase();
    } else {
        return value.toFixed(2);
    }
}

console.log("\n=== 함수 타입 ===");
console.log(`곱셈: 3 * 4 = ${multiply(3, 4)}`);
console.log(`나눗셈: 10 / 3 = ${divide(10, 3).toFixed(2)}`);
console.log(greet("홍길동"));
console.log(greet("김철수", "반갑습니다"));
console.log(`합계: ${sumAll(1, 2, 3, 4, 5)}`);
console.log(`문자열 포맷: ${format("hello")}`);
console.log(`숫자 포맷: ${format(3.14159)}`);

// ============================================
// 10. 열거형 (Enum)
// ============================================

// 숫자 열거형
enum Direction2 {
    Up = 1,
    Down,
    Left,
    Right
}

// 문자열 열거형
enum HttpStatus {
    OK = "OK",
    NotFound = "NOT_FOUND",
    ServerError = "SERVER_ERROR"
}

// const enum (인라인됨)
const enum Color {
    Red = "#ff0000",
    Green = "#00ff00",
    Blue = "#0000ff"
}

console.log("\n=== 열거형 ===");
console.log(`Direction.Up: ${Direction2.Up}`);
console.log(`Direction.Right: ${Direction2.Right}`);
console.log(`HttpStatus.OK: ${HttpStatus.OK}`);
console.log(`Color.Red: ${Color.Red}`);

// ============================================
// 11. Null 체크
// ============================================

// strictNullChecks가 활성화된 경우
function processValue(value: string | null | undefined) {
    // 옵셔널 체이닝
    const length = value?.length;

    // Nullish 병합
    const defaulted = value ?? "기본값";

    // Non-null 단언 (확실할 때만 사용)
    // const definite = value!.length;

    console.log(`길이: ${length}, 값: ${defaulted}`);
}

console.log("\n=== Null 체크 ===");
processValue("hello");
processValue(null);
processValue(undefined);

// ============================================
// 실행 예시
// ============================================
console.log("\n=== TypeScript 기초 완료 ===");
console.log("컴파일: npx tsc basics.ts");
console.log("실행: node basics.js");
