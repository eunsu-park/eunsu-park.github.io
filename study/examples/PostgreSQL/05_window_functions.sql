-- =============================================================================
-- PostgreSQL 윈도우 함수 예제
-- Window Functions (Analytic Functions)
-- =============================================================================

-- 먼저 이전 예제 파일들을 실행하여 테이블과 데이터를 생성하세요.

-- =============================================================================
-- 1. 기본 윈도우 함수 구조
-- =============================================================================

-- OVER() - 전체 테이블을 하나의 윈도우로
SELECT
    first_name,
    last_name,
    salary,
    SUM(salary) OVER() AS total_salary,
    ROUND(AVG(salary) OVER(), 2) AS avg_salary,
    COUNT(*) OVER() AS total_count
FROM employees;

-- =============================================================================
-- 2. PARTITION BY - 그룹별 윈도우
-- =============================================================================

-- 부서별로 파티션 나누기
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(PARTITION BY e.dept_id) AS dept_total,
    ROUND(AVG(e.salary) OVER(PARTITION BY e.dept_id), 2) AS dept_avg,
    COUNT(*) OVER(PARTITION BY e.dept_id) AS dept_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary DESC;

-- 부서별 급여 비율
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(PARTITION BY e.dept_id) AS dept_total,
    ROUND(100.0 * e.salary / SUM(e.salary) OVER(PARTITION BY e.dept_id), 2) AS salary_pct
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, salary_pct DESC;

-- =============================================================================
-- 3. 순위 함수 (Ranking Functions)
-- =============================================================================

-- ROW_NUMBER: 연속 번호 (동점자도 다른 번호)
-- RANK: 동점자는 같은 순위, 다음 순위 건너뜀
-- DENSE_RANK: 동점자는 같은 순위, 다음 순위 연속

SELECT
    first_name,
    last_name,
    salary,
    ROW_NUMBER() OVER(ORDER BY salary DESC) AS row_num,
    RANK() OVER(ORDER BY salary DESC) AS rank,
    DENSE_RANK() OVER(ORDER BY salary DESC) AS dense_rank
FROM employees;

-- 부서별 급여 순위
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    RANK() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS dept_salary_rank
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, dept_salary_rank;

-- 부서별 Top 2 급여자
SELECT * FROM (
    SELECT
        e.first_name,
        e.last_name,
        d.dept_name,
        e.salary,
        ROW_NUMBER() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS rn
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
) ranked
WHERE rn <= 2;

-- NTILE: N개의 버킷으로 분할
SELECT
    first_name,
    last_name,
    salary,
    NTILE(4) OVER(ORDER BY salary DESC) AS salary_quartile
FROM employees;

-- =============================================================================
-- 4. 오프셋 함수 (LAG, LEAD, FIRST_VALUE, LAST_VALUE)
-- =============================================================================

-- LAG: 이전 행 값
-- LEAD: 다음 행 값
SELECT
    first_name,
    hire_date,
    LAG(first_name, 1) OVER(ORDER BY hire_date) AS prev_hire,
    LEAD(first_name, 1) OVER(ORDER BY hire_date) AS next_hire
FROM employees
ORDER BY hire_date;

-- 급여 변화 분석
SELECT
    first_name,
    salary,
    LAG(salary) OVER(ORDER BY emp_id) AS prev_salary,
    salary - LAG(salary) OVER(ORDER BY emp_id) AS salary_diff
FROM employees;

-- 부서 내 이전/다음 직원
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    LAG(e.salary) OVER(PARTITION BY e.dept_id ORDER BY e.salary) AS lower_salary,
    LEAD(e.salary) OVER(PARTITION BY e.dept_id ORDER BY e.salary) AS higher_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary;

-- FIRST_VALUE, LAST_VALUE
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    FIRST_VALUE(e.first_name) OVER(
        PARTITION BY e.dept_id ORDER BY e.salary DESC
    ) AS highest_paid,
    LAST_VALUE(e.first_name) OVER(
        PARTITION BY e.dept_id ORDER BY e.salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- 5. 집계 윈도우 함수
-- =============================================================================

-- 누적 합계 (Running Total)
SELECT
    first_name,
    hire_date,
    salary,
    SUM(salary) OVER(ORDER BY hire_date) AS running_total
FROM employees
ORDER BY hire_date;

-- 부서별 누적 합계
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    SUM(e.salary) OVER(
        PARTITION BY e.dept_id
        ORDER BY e.emp_id
    ) AS dept_running_total
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.emp_id;

-- 이동 평균 (Moving Average)
SELECT
    first_name,
    salary,
    ROUND(AVG(salary) OVER(
        ORDER BY emp_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_3
FROM employees;

-- =============================================================================
-- 6. 프레임 절 (Frame Specification)
-- =============================================================================

-- ROWS: 물리적 행 기준
-- RANGE: 논리적 값 기준

-- 현재 행 기준 앞뒤 1행
SELECT
    first_name,
    salary,
    AVG(salary) OVER(
        ORDER BY salary
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS avg_neighbors
FROM employees;

-- 처음부터 현재 행까지
SELECT
    first_name,
    salary,
    MAX(salary) OVER(
        ORDER BY emp_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS max_so_far
FROM employees;

-- 현재 행부터 끝까지
SELECT
    first_name,
    salary,
    COUNT(*) OVER(
        ORDER BY emp_id
        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS remaining_count
FROM employees;

-- =============================================================================
-- 7. 백분위수와 분포 함수
-- =============================================================================

-- PERCENT_RANK: 0~1 사이의 상대적 순위
-- CUME_DIST: 누적 분포

SELECT
    first_name,
    salary,
    ROUND(PERCENT_RANK() OVER(ORDER BY salary)::numeric, 4) AS pct_rank,
    ROUND(CUME_DIST() OVER(ORDER BY salary)::numeric, 4) AS cumulative_dist
FROM employees;

-- 부서 내 백분위
SELECT
    e.first_name,
    d.dept_name,
    e.salary,
    ROUND(PERCENT_RANK() OVER(
        PARTITION BY e.dept_id ORDER BY e.salary
    )::numeric, 4) AS dept_percentile
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- =============================================================================
-- 8. 실전 예제
-- =============================================================================

-- 급여 통계 종합
SELECT
    e.first_name,
    e.last_name,
    d.dept_name,
    e.salary,
    -- 부서 통계
    ROUND(AVG(e.salary) OVER(PARTITION BY e.dept_id), 2) AS dept_avg,
    MIN(e.salary) OVER(PARTITION BY e.dept_id) AS dept_min,
    MAX(e.salary) OVER(PARTITION BY e.dept_id) AS dept_max,
    -- 전사 통계
    ROUND(AVG(e.salary) OVER(), 2) AS company_avg,
    -- 순위
    RANK() OVER(PARTITION BY e.dept_id ORDER BY e.salary DESC) AS dept_rank,
    RANK() OVER(ORDER BY e.salary DESC) AS company_rank,
    -- 전사 대비 비율
    ROUND(100.0 * e.salary / SUM(e.salary) OVER(), 2) AS company_pct
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_name, e.salary DESC;

-- 연속 고용 분석 (입사일 기준 간격)
SELECT
    first_name,
    hire_date,
    LAG(hire_date) OVER(ORDER BY hire_date) AS prev_hire_date,
    hire_date - LAG(hire_date) OVER(ORDER BY hire_date) AS days_since_last_hire
FROM employees
ORDER BY hire_date;

-- 급여 대역 분류
SELECT
    first_name,
    salary,
    CASE NTILE(3) OVER(ORDER BY salary)
        WHEN 1 THEN 'Low'
        WHEN 2 THEN 'Medium'
        WHEN 3 THEN 'High'
    END AS salary_band
FROM employees;

-- =============================================================================
-- 9. 윈도우 함수 별칭 (WINDOW 절)
-- =============================================================================

-- 같은 윈도우 정의 재사용
SELECT
    first_name,
    salary,
    SUM(salary) OVER w AS running_total,
    AVG(salary) OVER w AS running_avg,
    COUNT(*) OVER w AS running_count
FROM employees
WINDOW w AS (ORDER BY emp_id)
ORDER BY emp_id;

-- =============================================================================
-- 윈도우 함수 요약
-- =============================================================================
/*
순위 함수:
- ROW_NUMBER(): 연속 번호
- RANK(): 동점 시 같은 순위, 건너뜀
- DENSE_RANK(): 동점 시 같은 순위, 연속
- NTILE(n): n개 버킷으로 분할

오프셋 함수:
- LAG(col, n, default): n행 이전 값
- LEAD(col, n, default): n행 이후 값
- FIRST_VALUE(col): 프레임의 첫 값
- LAST_VALUE(col): 프레임의 마지막 값
- NTH_VALUE(col, n): 프레임의 n번째 값

집계 함수:
- SUM(), AVG(), COUNT(), MIN(), MAX() - OVER()와 함께

분포 함수:
- PERCENT_RANK(): 백분율 순위 (0~1)
- CUME_DIST(): 누적 분포

프레임 절:
- ROWS BETWEEN ... AND ...
- RANGE BETWEEN ... AND ...
- UNBOUNDED PRECEDING / FOLLOWING
- CURRENT ROW
- n PRECEDING / FOLLOWING

구문:
window_function() OVER(
    [PARTITION BY col, ...]
    [ORDER BY col [ASC|DESC], ...]
    [frame_clause]
)
*/
