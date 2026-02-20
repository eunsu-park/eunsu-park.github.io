-- =============================================================================
-- PostgreSQL CRUD 기본 예제
-- Basic CRUD Operations (Create, Read, Update, Delete)
-- =============================================================================

-- 이 파일은 PostgreSQL의 기본 CRUD 작업을 보여줍니다.
-- 실행 전 데이터베이스에 연결하세요: psql -U postgres -d your_database

-- =============================================================================
-- 1. CREATE - 테이블 생성 및 데이터 삽입
-- =============================================================================

-- 테이블 생성
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS departments CASCADE;

-- 부서 테이블
CREATE TABLE departments (
    dept_id SERIAL PRIMARY KEY,
    dept_name VARCHAR(50) NOT NULL UNIQUE,
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 직원 테이블
CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary NUMERIC(10, 2) CHECK (salary > 0),
    dept_id INTEGER REFERENCES departments(dept_id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_employees_dept ON employees(dept_id);
CREATE INDEX idx_employees_email ON employees(email);

-- =============================================================================
-- INSERT - 데이터 삽입
-- =============================================================================

-- 단일 행 삽입
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Seoul');

INSERT INTO departments (dept_name, location)
VALUES ('Marketing', 'Busan');

INSERT INTO departments (dept_name, location)
VALUES ('Sales', 'Daegu');

INSERT INTO departments (dept_name, location)
VALUES ('HR', 'Seoul');

-- 여러 행 한번에 삽입
INSERT INTO employees (first_name, last_name, email, hire_date, salary, dept_id)
VALUES
    ('철수', '김', 'kim.cs@company.com', '2020-01-15', 50000, 1),
    ('영희', '이', 'lee.yh@company.com', '2019-06-20', 55000, 1),
    ('민수', '박', 'park.ms@company.com', '2021-03-10', 48000, 2),
    ('수진', '정', 'jung.sj@company.com', '2018-11-05', 62000, 1),
    ('동욱', '최', 'choi.dw@company.com', '2022-08-01', 45000, 3),
    ('미영', '강', 'kang.my@company.com', '2020-05-15', 52000, 2),
    ('지훈', '조', 'cho.jh@company.com', '2019-09-20', 58000, 3),
    ('소영', '윤', 'yoon.sy@company.com', '2021-12-01', 47000, 4);

-- RETURNING 절로 삽입된 데이터 확인
INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('새직원', '테스트', 'test@company.com', 40000, 1)
RETURNING emp_id, first_name, last_name;

-- =============================================================================
-- 2. READ - 데이터 조회
-- =============================================================================

-- 전체 조회
SELECT * FROM employees;

-- 특정 컬럼만 조회
SELECT first_name, last_name, email, salary
FROM employees;

-- 조건부 조회 (WHERE)
SELECT first_name, last_name, salary
FROM employees
WHERE salary > 50000;

-- 여러 조건 (AND, OR)
SELECT *
FROM employees
WHERE dept_id = 1 AND salary > 50000;

SELECT *
FROM employees
WHERE dept_id = 1 OR salary > 55000;

-- BETWEEN, IN, LIKE
SELECT first_name, last_name, salary
FROM employees
WHERE salary BETWEEN 45000 AND 55000;

SELECT first_name, last_name, dept_id
FROM employees
WHERE dept_id IN (1, 2);

SELECT first_name, last_name, email
FROM employees
WHERE email LIKE '%@company.com';

-- NULL 체크
SELECT *
FROM employees
WHERE email IS NOT NULL;

-- 정렬 (ORDER BY)
SELECT first_name, last_name, salary
FROM employees
ORDER BY salary DESC;

SELECT first_name, last_name, dept_id, salary
FROM employees
ORDER BY dept_id ASC, salary DESC;

-- 제한 (LIMIT, OFFSET)
SELECT first_name, last_name, salary
FROM employees
ORDER BY salary DESC
LIMIT 5;

-- 페이지네이션 (2페이지, 페이지당 3개)
SELECT first_name, last_name, salary
FROM employees
ORDER BY emp_id
LIMIT 3 OFFSET 3;

-- DISTINCT - 중복 제거
SELECT DISTINCT dept_id
FROM employees;

-- 별칭 (AS)
SELECT
    first_name AS "이름",
    last_name AS "성",
    salary AS "연봉"
FROM employees;

-- 계산 컬럼
SELECT
    first_name,
    last_name,
    salary,
    salary * 12 AS annual_salary,
    salary * 1.1 AS after_raise
FROM employees;

-- =============================================================================
-- 3. UPDATE - 데이터 수정
-- =============================================================================

-- 단일 행 수정
UPDATE employees
SET salary = 52000
WHERE emp_id = 1;

-- 여러 컬럼 수정
UPDATE employees
SET salary = 55000, updated_at = CURRENT_TIMESTAMP
WHERE emp_id = 1;

-- 조건부 대량 수정
UPDATE employees
SET salary = salary * 1.1
WHERE dept_id = 1;

-- 서브쿼리를 이용한 수정
UPDATE employees
SET salary = salary * 1.05
WHERE dept_id = (SELECT dept_id FROM departments WHERE dept_name = 'Engineering');

-- RETURNING으로 수정된 데이터 확인
UPDATE employees
SET salary = salary * 1.02
WHERE emp_id = 3
RETURNING emp_id, first_name, salary;

-- =============================================================================
-- 4. DELETE - 데이터 삭제
-- =============================================================================

-- 조건부 삭제
DELETE FROM employees
WHERE emp_id = 9;

-- 전체 삭제 (주의!)
-- DELETE FROM employees;

-- TRUNCATE (빠른 전체 삭제, 트랜잭션 불가)
-- TRUNCATE TABLE employees RESTART IDENTITY;

-- RETURNING으로 삭제된 데이터 확인
DELETE FROM employees
WHERE email = 'test@company.com'
RETURNING *;

-- =============================================================================
-- 5. 트랜잭션 (Transaction)
-- =============================================================================

-- 트랜잭션 시작
BEGIN;

INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('트랜잭션', '테스트', 'trans@company.com', 45000, 1);

UPDATE employees
SET salary = salary * 1.05
WHERE email = 'trans@company.com';

-- 확인
SELECT * FROM employees WHERE email = 'trans@company.com';

-- 커밋 또는 롤백
COMMIT;
-- 또는 ROLLBACK;

-- =============================================================================
-- 6. UPSERT (INSERT ... ON CONFLICT)
-- =============================================================================

-- 중복 시 무시
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Incheon')
ON CONFLICT (dept_name) DO NOTHING;

-- 중복 시 업데이트
INSERT INTO departments (dept_name, location)
VALUES ('Engineering', 'Incheon')
ON CONFLICT (dept_name)
DO UPDATE SET location = EXCLUDED.location;

-- =============================================================================
-- 7. 테이블 확인 및 정보
-- =============================================================================

-- 테이블 구조 확인
\d employees

-- 테이블 목록
\dt

-- 인덱스 확인
\di

-- =============================================================================
-- 정리 (필요시)
-- =============================================================================
-- DROP TABLE IF EXISTS employees CASCADE;
-- DROP TABLE IF EXISTS departments CASCADE;
