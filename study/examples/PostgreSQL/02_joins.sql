-- =============================================================================
-- PostgreSQL JOIN 예제
-- Various Types of JOINs in PostgreSQL
-- =============================================================================

-- 먼저 01_basic_crud.sql을 실행하여 테이블과 데이터를 생성하세요.

-- =============================================================================
-- 테스트 데이터 추가
-- =============================================================================

-- 부서가 없는 직원 추가 (NULL dept_id)
INSERT INTO employees (first_name, last_name, email, salary, dept_id)
VALUES ('무소속', '직원', 'nodept@company.com', 40000, NULL);

-- 직원이 없는 부서 추가
INSERT INTO departments (dept_name, location)
VALUES ('Finance', 'Seoul');

-- =============================================================================
-- 1. INNER JOIN
-- =============================================================================
-- 양쪽 테이블에서 매칭되는 행만 반환

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.dept_name,
    d.location
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;

-- 테이블 별칭 없이
SELECT
    employees.first_name,
    employees.last_name,
    departments.dept_name
FROM employees
INNER JOIN departments ON employees.dept_id = departments.dept_id;

-- =============================================================================
-- 2. LEFT JOIN (LEFT OUTER JOIN)
-- =============================================================================
-- 왼쪽 테이블의 모든 행 + 매칭되는 오른쪽 테이블 행

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.dept_name,
    d.location
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;

-- 부서가 없는 직원만
SELECT
    e.first_name,
    e.last_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
WHERE d.dept_id IS NULL;

-- =============================================================================
-- 3. RIGHT JOIN (RIGHT OUTER JOIN)
-- =============================================================================
-- 오른쪽 테이블의 모든 행 + 매칭되는 왼쪽 테이블 행

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    d.dept_name,
    d.location
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- 직원이 없는 부서만
SELECT
    d.dept_name,
    d.location
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id
WHERE e.emp_id IS NULL;

-- =============================================================================
-- 4. FULL OUTER JOIN
-- =============================================================================
-- 양쪽 테이블의 모든 행 (매칭되지 않으면 NULL)

SELECT
    e.emp_id,
    e.first_name,
    e.last_name,
    d.dept_name,
    d.location
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- 매칭되지 않는 행만
SELECT
    e.emp_id,
    e.first_name,
    d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id
WHERE e.emp_id IS NULL OR d.dept_id IS NULL;

-- =============================================================================
-- 5. CROSS JOIN (카테시안 곱)
-- =============================================================================
-- 모든 가능한 조합

SELECT
    e.first_name,
    d.dept_name
FROM employees e
CROSS JOIN departments d
LIMIT 20;

-- CROSS JOIN은 ON 절 없이 콤마로도 표현 가능
SELECT
    e.first_name,
    d.dept_name
FROM employees e, departments d
WHERE e.dept_id IS NOT NULL
LIMIT 20;

-- =============================================================================
-- 6. SELF JOIN
-- =============================================================================
-- 같은 테이블을 자기 자신과 조인

-- 예제를 위한 관리자 컬럼 추가
ALTER TABLE employees ADD COLUMN IF NOT EXISTS manager_id INTEGER REFERENCES employees(emp_id);

-- 일부 직원에게 관리자 지정
UPDATE employees SET manager_id = 4 WHERE emp_id IN (1, 2);
UPDATE employees SET manager_id = 1 WHERE emp_id IN (5, 6);

-- 직원과 관리자 조회
SELECT
    e.emp_id,
    e.first_name || ' ' || e.last_name AS employee_name,
    m.first_name || ' ' || m.last_name AS manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;

-- =============================================================================
-- 7. 다중 테이블 JOIN
-- =============================================================================

-- 프로젝트 테이블 생성
DROP TABLE IF EXISTS projects CASCADE;
DROP TABLE IF EXISTS employee_projects CASCADE;

CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget NUMERIC(12, 2)
);

-- 직원-프로젝트 연결 테이블 (다대다 관계)
CREATE TABLE employee_projects (
    emp_id INTEGER REFERENCES employees(emp_id),
    project_id INTEGER REFERENCES projects(project_id),
    role VARCHAR(50),
    PRIMARY KEY (emp_id, project_id)
);

-- 데이터 삽입
INSERT INTO projects (project_name, start_date, end_date, budget)
VALUES
    ('웹사이트 리뉴얼', '2024-01-01', '2024-06-30', 100000),
    ('모바일 앱 개발', '2024-03-01', '2024-12-31', 200000),
    ('데이터 분석 플랫폼', '2024-02-01', '2024-08-31', 150000);

INSERT INTO employee_projects (emp_id, project_id, role)
VALUES
    (1, 1, 'Lead'),
    (2, 1, 'Developer'),
    (1, 2, 'Developer'),
    (3, 2, 'Lead'),
    (4, 3, 'Lead'),
    (2, 3, 'Analyst');

-- 3개 테이블 조인: 직원 + 부서 + 프로젝트
SELECT
    e.first_name || ' ' || e.last_name AS employee_name,
    d.dept_name,
    p.project_name,
    ep.role
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
JOIN employee_projects ep ON e.emp_id = ep.emp_id
JOIN projects p ON ep.project_id = p.project_id
ORDER BY e.first_name, p.project_name;

-- =============================================================================
-- 8. NATURAL JOIN
-- =============================================================================
-- 같은 이름의 컬럼으로 자동 조인 (사용 비권장 - 명시적 ON 절 권장)

-- dept_id가 같은 이름이므로 자동 매칭
-- SELECT * FROM employees NATURAL JOIN departments;

-- =============================================================================
-- 9. USING 절
-- =============================================================================
-- 같은 이름의 컬럼이 있을 때 ON 대신 사용

SELECT
    e.first_name,
    e.last_name,
    d.dept_name
FROM employees e
JOIN departments d USING (dept_id);

-- =============================================================================
-- 10. 조인 + 집계
-- =============================================================================

-- 부서별 직원 수와 평균 급여
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    MIN(e.salary) AS min_salary,
    MAX(e.salary) AS max_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY employee_count DESC;

-- 프로젝트별 참여 직원 수
SELECT
    p.project_name,
    COUNT(ep.emp_id) AS member_count,
    p.budget
FROM projects p
LEFT JOIN employee_projects ep ON p.project_id = ep.project_id
GROUP BY p.project_id, p.project_name, p.budget
ORDER BY member_count DESC;

-- =============================================================================
-- 11. 조인 성능 팁
-- =============================================================================

-- 조인에 사용되는 컬럼에 인덱스가 있는지 확인
-- CREATE INDEX idx_employees_dept ON employees(dept_id);
-- CREATE INDEX idx_employee_projects_emp ON employee_projects(emp_id);
-- CREATE INDEX idx_employee_projects_proj ON employee_projects(project_id);

-- 실행 계획 확인
EXPLAIN ANALYZE
SELECT
    e.first_name,
    d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- JOIN 유형 요약
-- =============================================================================
/*
| JOIN 유형        | 설명                                      |
|-----------------|-------------------------------------------|
| INNER JOIN      | 양쪽 모두 매칭되는 행만                      |
| LEFT JOIN       | 왼쪽 테이블 전체 + 매칭되는 오른쪽           |
| RIGHT JOIN      | 오른쪽 테이블 전체 + 매칭되는 왼쪽           |
| FULL OUTER JOIN | 양쪽 테이블 전체                            |
| CROSS JOIN      | 모든 가능한 조합 (카테시안 곱)               |
| SELF JOIN       | 같은 테이블끼리 조인                         |

팁:
- 항상 ON 절을 명시적으로 작성
- 조인 컬럼에 인덱스 생성
- EXPLAIN으로 실행 계획 확인
- 불필요한 CROSS JOIN 피하기
*/
