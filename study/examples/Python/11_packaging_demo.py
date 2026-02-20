"""
Python Packaging and Project Structure

Demonstrates:
- pyproject.toml structure
- setup.py vs pyproject.toml
- Entry points
- Virtual environments
- Package structure
- Versioning
- Dependencies management
"""

from typing import List


def section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# Modern Python Package Structure
# =============================================================================

section("Modern Python Package Structure")

print("""
Recommended package structure:

my_package/
├── pyproject.toml          # Project metadata (PEP 518, 621)
├── README.md               # Project description
├── LICENSE                 # License file
├── .gitignore             # Git ignore patterns
├── src/                   # Source code directory
│   └── my_package/        # Actual package
│       ├── __init__.py    # Package initialization
│       ├── core.py        # Core functionality
│       ├── utils.py       # Utility functions
│       └── py.typed       # Type hints marker
├── tests/                 # Test directory
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/                  # Documentation
│   └── index.md
└── examples/              # Usage examples
    └── example.py

Why src/ layout?
- Prevents accidental imports from source
- Ensures package is installed before testing
- Better isolation during development
""")


# =============================================================================
# pyproject.toml Example
# =============================================================================

section("pyproject.toml Example")

pyproject_toml = '''
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-package"
version = "0.1.0"
description = "A short description of my package"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
maintainers = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["example", "package", "tutorial"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "requests>=2.28.0",
    "numpy>=1.22.0",
    "pandas>=1.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "mypy>=0.990",
    "ruff>=0.0.250",
]
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/username/my-package"
Documentation = "https://my-package.readthedocs.io"
Repository = "https://github.com/username/my-package"
"Bug Tracker" = "https://github.com/username/my-package/issues"

[project.scripts]
my-command = "my_package.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]

[tool.ruff]
line-length = 88
target-version = "py38"
'''

print("Modern pyproject.toml (PEP 621):")
print(pyproject_toml)


# =============================================================================
# Entry Points
# =============================================================================

section("Entry Points")

print("""
Entry points create command-line scripts from Python functions.

In pyproject.toml:
  [project.scripts]
  my-command = "my_package.cli:main"

This creates a 'my-command' executable that calls main() in my_package/cli.py

Example cli.py:
""")

cli_example = '''
# src/my_package/cli.py

def main():
    """Main entry point for CLI."""
    import sys
    print(f"Hello from my-package!")
    print(f"Arguments: {sys.argv[1:]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

print(cli_example)

print("""
After installation:
  $ my-command arg1 arg2
  Hello from my-package!
  Arguments: ['arg1', 'arg2']
""")


# =============================================================================
# Virtual Environments
# =============================================================================

section("Virtual Environments")

print("""
Virtual environments isolate project dependencies.

Creating virtual environment:
  # Using venv (built-in)
  $ python -m venv venv
  $ python -m venv .venv  # Hidden directory

  # Using virtualenv (third-party, more features)
  $ pip install virtualenv
  $ virtualenv venv

Activating:
  # Linux/Mac
  $ source venv/bin/activate
  $ source .venv/bin/activate

  # Windows
  $ venv\\Scripts\\activate

  # With activated venv, prompt shows:
  (venv) $ python --version

Deactivating:
  (venv) $ deactivate

Modern alternatives:
  # Poetry - dependency management + virtual environments
  $ pip install poetry
  $ poetry init
  $ poetry add requests
  $ poetry install
  $ poetry run python script.py

  # pipenv - combines pip and virtualenv
  $ pip install pipenv
  $ pipenv install requests
  $ pipenv shell

  # uv - faster pip/venv (Rust-based)
  $ pip install uv
  $ uv venv
  $ uv pip install requests
""")


# =============================================================================
# Installing Package
# =============================================================================

section("Installing Package")

print("""
Development installation (editable mode):
  # Changes to source code immediately available
  $ pip install -e .
  $ pip install -e ".[dev]"      # With dev dependencies
  $ pip install -e ".[dev,docs]" # Multiple extras

Regular installation:
  $ pip install .
  $ pip install .[dev]

From PyPI (after publishing):
  $ pip install my-package
  $ pip install my-package==0.1.0
  $ pip install my-package>=0.1.0

From git:
  $ pip install git+https://github.com/username/my-package.git
  $ pip install git+https://github.com/username/my-package.git@v0.1.0

From local wheel:
  $ pip install dist/my_package-0.1.0-py3-none-any.whl
""")


# =============================================================================
# Building and Publishing
# =============================================================================

section("Building and Publishing")

print("""
Build package:
  # Install build tools
  $ pip install build twine

  # Build distributions
  $ python -m build
  # Creates:
  #   dist/my_package-0.1.0-py3-none-any.whl
  #   dist/my_package-0.1.0.tar.gz

Check package:
  $ twine check dist/*

Upload to TestPyPI (testing):
  $ twine upload --repository testpypi dist/*

Upload to PyPI (production):
  $ twine upload dist/*

Test installation from TestPyPI:
  $ pip install --index-url https://test.pypi.org/simple/ my-package
""")


# =============================================================================
# Requirements Files
# =============================================================================

section("Requirements Files")

print("""
requirements.txt (production dependencies):
  requests>=2.28.0
  numpy>=1.22.0,<2.0.0
  pandas==1.5.3

requirements-dev.txt (development dependencies):
  -r requirements.txt  # Include production deps
  pytest>=7.0.0
  black>=22.0.0
  mypy>=0.990

requirements-lock.txt (exact versions, for reproducibility):
  requests==2.28.2
  numpy==1.24.3
  pandas==1.5.3
  # ... with all transitive dependencies

Installing:
  $ pip install -r requirements.txt
  $ pip install -r requirements-dev.txt

Freezing current environment:
  $ pip freeze > requirements-lock.txt

Generating requirements from pyproject.toml:
  $ pip-compile pyproject.toml  # Using pip-tools
""")


# =============================================================================
# Package Versioning
# =============================================================================

section("Package Versioning")

print("""
Semantic Versioning (SemVer): MAJOR.MINOR.PATCH

  MAJOR: Incompatible API changes (1.0.0 -> 2.0.0)
  MINOR: Add functionality (backward-compatible) (1.0.0 -> 1.1.0)
  PATCH: Bug fixes (backward-compatible) (1.0.0 -> 1.0.1)

Examples:
  0.1.0  - Initial development
  1.0.0  - First stable release
  1.1.0  - New feature added
  1.1.1  - Bug fix
  2.0.0  - Breaking change

Pre-release versions:
  1.0.0a1  - Alpha 1
  1.0.0b1  - Beta 1
  1.0.0rc1 - Release candidate 1

Version specifiers in dependencies:
  requests>=2.28.0        # At least 2.28.0
  requests>=2.28.0,<3.0.0 # At least 2.28.0, but less than 3.0.0
  requests==2.28.2        # Exactly 2.28.2
  requests~=2.28.0        # Compatible (>=2.28.0, <2.29.0)
  requests!=2.28.1        # Not 2.28.1

Single-sourcing version:
  # In src/my_package/__init__.py
  __version__ = "0.1.0"

  # In pyproject.toml
  [project]
  dynamic = ["version"]

  [tool.setuptools.dynamic]
  version = {attr = "my_package.__version__"}
""")


# =============================================================================
# Package __init__.py
# =============================================================================

section("Package __init__.py")

init_example = '''
# src/my_package/__init__.py

"""My Package - A short description."""

from my_package.core import main_function, AnotherClass
from my_package.utils import helper_function

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Define public API
__all__ = [
    "main_function",
    "AnotherClass",
    "helper_function",
]

# Package-level initialization (if needed)
def _initialize():
    """Initialize package."""
    pass

_initialize()
'''

print("Example __init__.py:")
print(init_example)


# =============================================================================
# Manifest and Data Files
# =============================================================================

section("Including Data Files")

print("""
MANIFEST.in (for sdist, source distribution):
  include README.md
  include LICENSE
  include pyproject.toml
  recursive-include src/my_package *.py
  recursive-include src/my_package *.typed
  recursive-include tests *.py
  include src/my_package/data/*.json
  exclude *.pyc
  exclude __pycache__
  prune .git

pyproject.toml configuration:
  [tool.setuptools]
  include-package-data = true

  [tool.setuptools.package-data]
  my_package = ["data/*.json", "py.typed"]

Accessing data files in code:
  # Using importlib.resources (Python 3.9+)
  from importlib.resources import files
  data_path = files("my_package").joinpath("data/config.json")

  # Using importlib.resources (Python 3.7-3.8)
  from importlib.resources import read_text
  content = read_text("my_package.data", "config.json")

  # Using pkg_resources (legacy)
  from pkg_resources import resource_filename
  path = resource_filename("my_package", "data/config.json")
""")


# =============================================================================
# Development Workflow
# =============================================================================

section("Development Workflow")

print("""
1. Create project structure:
   $ mkdir my-package
   $ cd my-package
   $ mkdir -p src/my_package tests docs

2. Initialize git:
   $ git init
   $ git add .
   $ git commit -m "Initial commit"

3. Create virtual environment:
   $ python -m venv .venv
   $ source .venv/bin/activate

4. Install in editable mode:
   $ pip install -e ".[dev]"

5. Run tests:
   $ pytest

6. Format code:
   $ black src/ tests/

7. Type checking:
   $ mypy src/

8. Build distribution:
   $ python -m build

9. Publish to PyPI:
   $ twine upload dist/*

Pre-commit hooks (optional):
  $ pip install pre-commit
  $ cat > .pre-commit-config.yaml << EOF
  repos:
    - repo: https://github.com/psf/black
      rev: 23.0.0
      hooks:
        - id: black
    - repo: https://github.com/pycqa/isort
      rev: 5.12.0
      hooks:
        - id: isort
  EOF
  $ pre-commit install
""")


# =============================================================================
# Summary
# =============================================================================

section("Summary")

print("""
Python packaging essentials:
1. pyproject.toml - Modern project metadata (PEP 621)
2. src/ layout - Better isolation and testing
3. Entry points - Create CLI commands
4. Virtual environments - Isolate dependencies
5. Editable installs - Development workflow
6. build + twine - Build and publish packages
7. requirements.txt - Pin dependencies
8. Semantic versioning - Version management
9. __init__.py - Define package API
10. MANIFEST.in - Include non-Python files

Modern tools:
- Poetry - All-in-one dependency + package management
- pipenv - Combines pip + virtualenv
- uv - Faster pip/venv alternative (Rust-based)
- build - Standard build tool
- twine - Secure PyPI uploads

Best practices:
- Use pyproject.toml over setup.py
- Use src/ layout for packages
- Pin exact versions in production
- Use semantic versioning
- Include type hints (py.typed)
- Write comprehensive README
- Test before publishing
""")
