-- =============================================================================
-- PostgreSQL 집계 함수 예제
-- Aggregation Functions and GROUP BY
-- =============================================================================

-- 먼저 01_basic_crud.sql과 02_joins.sql을 실행하여 테이블과 데이터를 생성하세요.

-- =============================================================================
-- 1. 기본 집계 함수
-- =============================================================================

-- COUNT - 행 개수
SELECT COUNT(*) AS total_employees FROM employees;

SELECT COUNT(email) AS employees_with_email FROM employees;  -- NULL 제외

SELECT COUNT(DISTINCT dept_id) AS unique_departments FROM employees;

-- SUM - 합계
SELECT SUM(salary) AS total_salary FROM employees;

-- AVG - 평균
SELECT AVG(salary) AS average_salary FROM employees;

SELECT ROUND(AVG(salary), 2) AS avg_salary_rounded FROM employees;

-- MIN, MAX - 최솟값, 최댓값
SELECT
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees;

SELECT
    MIN(hire_date) AS first_hire,
    MAX(hire_date) AS last_hire
FROM employees;

-- 모든 집계 함수 함께 사용
SELECT
    COUNT(*) AS employee_count,
    SUM(salary) AS total_salary,
    ROUND(AVG(salary), 2) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    MAX(salary) - MIN(salary) AS salary_range
FROM employees;

-- =============================================================================
-- 2. GROUP BY
-- =============================================================================

-- 부서별 집계
SELECT
    dept_id,
    COUNT(*) AS employee_count,
    ROUND(AVG(salary), 2) AS avg_salary
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id
ORDER BY employee_count DESC;

-- JOIN과 함께 사용
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    SUM(e.salary) AS total_salary,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
ORDER BY total_salary DESC NULLS LAST;

-- 여러 컬럼으로 그룹화
SELECT
    dept_id,
    is_active,
    COUNT(*) AS employee_count
FROM employees
GROUP BY dept_id, is_active
ORDER BY dept_id, is_active;

-- 표현식으로 그룹화
SELECT
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS hire_count
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date)
ORDER BY hire_year;

-- =============================================================================
-- 3. HAVING - 그룹 필터링
-- =============================================================================

-- 직원이 2명 이상인 부서
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(e.emp_id) >= 2
ORDER BY employee_count DESC;

-- 평균 급여가 50000 이상인 부서
SELECT
    d.dept_name,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING AVG(e.salary) >= 50000
ORDER BY avg_salary DESC;

-- WHERE와 HAVING 함께 사용
SELECT
    d.dept_name,
    COUNT(e.emp_id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
WHERE e.is_active = TRUE  -- 행 필터링 (그룹 전)
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(e.emp_id) >= 1  -- 그룹 필터링 (그룹 후)
ORDER BY avg_salary DESC;

-- =============================================================================
-- 4. 고급 집계 함수
-- =============================================================================

-- STRING_AGG - 문자열 연결
SELECT
    d.dept_name,
    STRING_AGG(e.first_name || ' ' || e.last_name, ', ' ORDER BY e.first_name) AS employee_names
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- ARRAY_AGG - 배열로 집계
SELECT
    d.dept_name,
    ARRAY_AGG(e.first_name ORDER BY e.first_name) AS employee_names_array
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- JSON_AGG - JSON 배열로 집계
SELECT
    d.dept_name,
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'name', e.first_name || ' ' || e.last_name,
            'salary', e.salary
        )
    ) AS employees_json
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- 5. ROLLUP, CUBE, GROUPING SETS
-- =============================================================================

-- ROLLUP - 계층적 소계
SELECT
    d.dept_name,
    EXTRACT(YEAR FROM e.hire_date) AS hire_year,
    COUNT(*) AS employee_count,
    SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY ROLLUP(d.dept_name, EXTRACT(YEAR FROM e.hire_date))
ORDER BY d.dept_name NULLS LAST, hire_year NULLS LAST;

-- CUBE - 모든 가능한 조합의 소계
SELECT
    d.dept_name,
    e.is_active,
    COUNT(*) AS employee_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY CUBE(d.dept_name, e.is_active)
ORDER BY d.dept_name NULLS LAST, e.is_active NULLS LAST;

-- GROUPING SETS - 특정 조합만 소계
SELECT
    d.dept_name,
    EXTRACT(YEAR FROM e.hire_date) AS hire_year,
    COUNT(*) AS employee_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY GROUPING SETS (
    (d.dept_name),
    (EXTRACT(YEAR FROM e.hire_date)),
    ()
)
ORDER BY d.dept_name NULLS LAST, hire_year NULLS LAST;

-- GROUPING() 함수로 소계 행 구분
SELECT
    CASE WHEN GROUPING(d.dept_name) = 1 THEN 'All Departments' ELSE d.dept_name END AS dept_name,
    COUNT(*) AS employee_count,
    SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY ROLLUP(d.dept_name)
ORDER BY GROUPING(d.dept_name), d.dept_name;

-- =============================================================================
-- 6. FILTER 절
-- =============================================================================

-- 조건별로 다른 집계
SELECT
    d.dept_name,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE e.salary > 50000) AS high_salary_count,
    COUNT(*) FILTER (WHERE e.salary <= 50000) AS low_salary_count,
    ROUND(AVG(e.salary) FILTER (WHERE e.is_active = TRUE), 2) AS active_avg_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- 7. 통계 함수
-- =============================================================================

-- 표준편차와 분산
SELECT
    d.dept_name,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    ROUND(STDDEV(e.salary), 2) AS stddev_salary,
    ROUND(VARIANCE(e.salary), 2) AS variance_salary
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name
HAVING COUNT(*) > 1;  -- 표준편차는 2개 이상 필요

-- 백분위수
SELECT
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS q1_salary,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS q3_salary
FROM employees;

-- 최빈값 (MODE)
SELECT
    MODE() WITHIN GROUP (ORDER BY dept_id) AS most_common_dept
FROM employees;

-- =============================================================================
-- 8. 조건부 집계 (CASE와 함께)
-- =============================================================================

SELECT
    d.dept_name,
    COUNT(*) AS total,
    SUM(CASE WHEN e.salary >= 55000 THEN 1 ELSE 0 END) AS high_earners,
    SUM(CASE WHEN e.salary < 55000 THEN 1 ELSE 0 END) AS others,
    ROUND(
        100.0 * SUM(CASE WHEN e.salary >= 55000 THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS high_earner_pct
FROM departments d
JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_id, d.dept_name;

-- =============================================================================
-- 집계 함수 요약
-- =============================================================================
/*
기본 집계:
- COUNT(*), COUNT(col), COUNT(DISTINCT col)
- SUM(col), AVG(col)
- MIN(col), MAX(col)

문자열/배열 집계:
- STRING_AGG(col, delimiter)
- ARRAY_AGG(col)
- JSON_AGG(value)

그룹화:
- GROUP BY: 기본 그룹화
- HAVING: 그룹 필터링 (집계 후)
- ROLLUP: 계층적 소계
- CUBE: 모든 조합 소계
- GROUPING SETS: 특정 조합 소계

고급:
- FILTER (WHERE ...): 조건부 집계
- PERCENTILE_CONT(): 백분위수
- STDDEV(), VARIANCE(): 통계

순서:
SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT
*/
