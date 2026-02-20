# Software Engineering Examples

This directory contains 8 example files demonstrating key software engineering concepts, from requirements specification to CI/CD pipelines. Python examples use only the standard library (no external dependencies).

## Files Overview

### 1. `04_user_story_template.md` - User Stories and Acceptance Criteria
**Concepts:**
- User story format: "As a [role], I want [feature], so that [benefit]"
- Acceptance criteria in Given/When/Then (Gherkin) format
- INVEST criteria checklist
- Definition of Done

---

### 2. `05_uml_class_diagram.py` - UML Class Diagram Generator
**Concepts:**
- ASCII-based UML class diagram rendering
- Class attributes and methods with visibility (+/-/#)
- Relationships: inheritance, composition, aggregation, association, dependency
- E-commerce domain model example

**Run:** `python 05_uml_class_diagram.py`

---

### 3. `06_estimation_calculator.py` - Software Estimation Calculator
**Concepts:**
- COCOMO II basic model (effort, duration, team size)
- Three-point PERT estimation with confidence intervals
- Story point velocity projection
- Function Point Analysis (IFPUG)

**Run:** `python 06_estimation_calculator.py`

---

### 4. `07_code_metrics.py` - Code Quality Metrics
**Concepts:**
- Cyclomatic complexity calculation using Python `ast` module
- Lines of code analysis (total, blank, comment, logical)
- Halstead metrics (vocabulary, volume, difficulty, effort)
- Risk rating classification

**Run:** `python 07_code_metrics.py`

---

### 5. `10_gantt_chart.py` - Gantt Chart and Critical Path
**Concepts:**
- Critical Path Method (CPM): forward/backward pass
- Task dependency resolution
- Slack calculation and critical path identification
- ASCII Gantt chart rendering

**Run:** `python 10_gantt_chart.py`

---

### 6. `11_tech_debt_tracker.py` - Technical Debt Tracker
**Concepts:**
- Technical debt modeling (type, severity, interest rate)
- ROI-based prioritization for debt payoff
- Sprint simulation with greedy payoff strategy
- Debt report generation

**Run:** `python 11_tech_debt_tracker.py`

---

### 7. `13_ci_cd_pipeline.yml` - GitHub Actions CI/CD Pipeline
**Concepts:**
- Multi-stage pipeline: lint, test, build, deploy
- Matrix strategy for multiple Python versions
- Dependency caching and artifact upload
- Environment protection rules for production
- Rolling deployment strategy

---

### 8. `14_adr_template.md` - Architecture Decision Records
**Concepts:**
- ADR template (Nygard format)
- Status, Context, Decision, Consequences structure
- Two complete example ADRs (database selection, microservices extraction)
- Guidelines for writing effective ADRs
