-- =============================================================================
-- PostgreSQL 서브쿼리와 CTE 예제
-- Subqueries and Common Table Expressions (CTE)
-- =============================================================================

-- 먼저 이전 예제 파일들을 실행하여 테이블과 데이터를 생성하세요.

-- =============================================================================
-- 1. 스칼라 서브쿼리 (단일 값 반환)
-- =============================================================================

-- SELECT 절에서 사용
SELECT
    first_name,
    last_name,
    salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees;

-- WHERE 절에서 사용
SELECT first_name, last_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 부서별 평균보다 높은 급여를 받는 직원
SELECT e.first_name, e.last_name, e.salary, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE dept_id = e.dept_id
);

-- =============================================================================
-- 2. 인라인 뷰 (FROM 절 서브쿼리)
-- =============================================================================

-- 부서별 통계를 서브쿼리로 구한 후 조인
SELECT
    d.dept_name,
    ds.employee_count,
    ds.avg_salary,
    ds.total_salary
FROM departments d
JOIN (
    SELECT
        dept_id,
        COUNT(*) AS employee_count,
        ROUND(AVG(salary), 2) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    WHERE dept_id IS NOT NULL
    GROUP BY dept_id
) ds ON d.dept_id = ds.dept_id;

-- 급여 순위와 함께 조회
SELECT *
FROM (
    SELECT
        first_name,
        last_name,
        salary,
        RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees
) ranked
WHERE salary_rank <= 5;

-- =============================================================================
-- 3. EXISTS / NOT EXISTS
-- =============================================================================

-- 프로젝트에 참여 중인 직원
SELECT e.first_name, e.last_name
FROM employees e
WHERE EXISTS (
    SELECT 1
    FROM employee_projects ep
    WHERE ep.emp_id = e.emp_id
);

-- 프로젝트에 참여하지 않는 직원
SELECT e.first_name, e.last_name
FROM employees e
WHERE NOT EXISTS (
    SELECT 1
    FROM employee_projects ep
    WHERE ep.emp_id = e.emp_id
);

-- 직원이 있는 부서
SELECT d.dept_name
FROM departments d
WHERE EXISTS (
    SELECT 1
    FROM employees e
    WHERE e.dept_id = d.dept_id
);

-- =============================================================================
-- 4. IN / NOT IN
-- =============================================================================

-- Engineering 부서 직원
SELECT first_name, last_name
FROM employees
WHERE dept_id IN (
    SELECT dept_id
    FROM departments
    WHERE dept_name = 'Engineering'
);

-- 프로젝트에 참여하지 않는 직원 (NOT IN 주의: NULL 처리)
SELECT first_name, last_name
FROM employees
WHERE emp_id NOT IN (
    SELECT emp_id FROM employee_projects
);

-- =============================================================================
-- 5. ANY / ALL
-- =============================================================================

-- Engineering 부서의 어떤 직원보다 급여가 높은 직원
SELECT first_name, last_name, salary
FROM employees
WHERE salary > ANY (
    SELECT salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.dept_name = 'Engineering'
);

-- Engineering 부서의 모든 직원보다 급여가 높은 직원
SELECT first_name, last_name, salary
FROM employees
WHERE salary > ALL (
    SELECT salary
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE d.dept_name = 'Engineering'
);

-- =============================================================================
-- 6. 상관 서브쿼리 (Correlated Subquery)
-- =============================================================================

-- 각 직원의 부서 평균과 비교
SELECT
    e.first_name,
    e.last_name,
    e.salary,
    (
        SELECT ROUND(AVG(e2.salary), 2)
        FROM employees e2
        WHERE e2.dept_id = e.dept_id
    ) AS dept_avg_salary
FROM employees e
WHERE e.dept_id IS NOT NULL;

-- 부서 내에서 가장 높은 급여를 받는 직원
SELECT e.first_name, e.last_name, e.salary, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary = (
    SELECT MAX(e2.salary)
    FROM employees e2
    WHERE e2.dept_id = e.dept_id
);

-- =============================================================================
-- 7. CTE (Common Table Expression) - WITH 절
-- =============================================================================

-- 기본 CTE
WITH dept_stats AS (
    SELECT
        dept_id,
        COUNT(*) AS employee_count,
        ROUND(AVG(salary), 2) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    WHERE dept_id IS NOT NULL
    GROUP BY dept_id
)
SELECT
    d.dept_name,
    ds.employee_count,
    ds.avg_salary,
    ds.total_salary
FROM departments d
JOIN dept_stats ds ON d.dept_id = ds.dept_id
ORDER BY ds.total_salary DESC;

-- 여러 CTE 사용
WITH
high_earners AS (
    SELECT emp_id, first_name, last_name, salary
    FROM employees
    WHERE salary > 50000
),
dept_names AS (
    SELECT e.emp_id, d.dept_name
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
)
SELECT
    h.first_name,
    h.last_name,
    h.salary,
    dn.dept_name
FROM high_earners h
LEFT JOIN dept_names dn ON h.emp_id = dn.emp_id;

-- CTE를 여러 번 참조
WITH emp_summary AS (
    SELECT
        dept_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY dept_id
)
SELECT
    'Total Departments' AS metric,
    COUNT(*) AS value
FROM emp_summary
UNION ALL
SELECT
    'Avg Employees per Dept',
    ROUND(AVG(emp_count), 2)
FROM emp_summary
UNION ALL
SELECT
    'Overall Avg Salary',
    ROUND(AVG(avg_salary), 2)
FROM emp_summary;

-- =============================================================================
-- 8. 재귀 CTE (Recursive CTE)
-- =============================================================================

-- 숫자 시퀀스 생성
WITH RECURSIVE numbers AS (
    -- Base case
    SELECT 1 AS n
    UNION ALL
    -- Recursive case
    SELECT n + 1
    FROM numbers
    WHERE n < 10
)
SELECT n FROM numbers;

-- 조직도 (직원-관리자 계층)
WITH RECURSIVE org_chart AS (
    -- Base case: 최상위 관리자 (manager_id가 NULL)
    SELECT
        emp_id,
        first_name || ' ' || last_name AS name,
        manager_id,
        1 AS level,
        ARRAY[emp_id] AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: 부하 직원
    SELECT
        e.emp_id,
        e.first_name || ' ' || e.last_name,
        e.manager_id,
        oc.level + 1,
        oc.path || e.emp_id
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT
    REPEAT('  ', level - 1) || name AS employee_hierarchy,
    level
FROM org_chart
ORDER BY path;

-- 날짜 시리즈 생성
WITH RECURSIVE date_series AS (
    SELECT DATE '2024-01-01' AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < DATE '2024-01-10'
)
SELECT date FROM date_series;

-- =============================================================================
-- 9. LATERAL JOIN (상관 서브쿼리의 대안)
-- =============================================================================

-- 각 부서의 상위 2명
SELECT d.dept_name, top_employees.*
FROM departments d
CROSS JOIN LATERAL (
    SELECT first_name, last_name, salary
    FROM employees e
    WHERE e.dept_id = d.dept_id
    ORDER BY salary DESC
    LIMIT 2
) AS top_employees;

-- =============================================================================
-- 10. 서브쿼리 vs CTE vs LATERAL 비교
-- =============================================================================

-- 방법 1: 서브쿼리 (인라인 뷰)
SELECT *
FROM (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
) sub
WHERE avg_sal > 50000;

-- 방법 2: CTE (더 읽기 쉬움)
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT * FROM dept_avg WHERE avg_sal > 50000;

-- 방법 3: LATERAL (행별로 서브쿼리 필요할 때)
SELECT d.dept_name, stats.avg_salary
FROM departments d
CROSS JOIN LATERAL (
    SELECT ROUND(AVG(salary), 2) AS avg_salary
    FROM employees e
    WHERE e.dept_id = d.dept_id
) stats
WHERE stats.avg_salary > 50000;

-- =============================================================================
-- 서브쿼리와 CTE 요약
-- =============================================================================
/*
서브쿼리 위치:
- SELECT: 스칼라 서브쿼리 (단일 값)
- FROM: 인라인 뷰 (테이블처럼 사용)
- WHERE: 조건에서 사용

서브쿼리 연산자:
- =, <, >: 스칼라 비교
- IN, NOT IN: 목록 포함 여부
- EXISTS, NOT EXISTS: 존재 여부
- ANY, ALL: 조건 비교

CTE 장점:
- 가독성 향상
- 재사용 가능
- 재귀 쿼리 지원
- 실행 계획 최적화 힌트 (MATERIALIZED)

LATERAL:
- 행별로 상관 서브쿼리 실행
- TOP-N 문제에 유용
*/
