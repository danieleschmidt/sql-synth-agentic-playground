# Contributing to SQL Synth Agentic Playground

Thank you for your interest in contributing to the SQL Synth Agentic Playground! This document provides guidelines and information for contributors.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized development)
- Visual Studio Code (recommended)

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 📚 **Documentation**: Improve or add documentation
- 🧪 **Testing**: Add or improve test coverage
- 🎨 **Design**: UI/UX improvements
- 🌐 **Translations**: Help make the project accessible globally

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/sql-synth-agentic-playground.git
cd sql-synth-agentic-playground

# Add the original repository as upstream
git remote add upstream https://github.com/danieleschmidt/sql-synth-agentic-playground.git
```

### 2. Environment Setup

#### Option A: Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

#### Option B: Dev Container (Recommended)
```bash
# Open in VS Code with Dev Containers extension
code .
# Command Palette (Ctrl+Shift+P) -> "Dev Containers: Reopen in Container"
```

#### Option C: Docker Development
```bash
# Build development image
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# See .env.example for all available options
```

### 4. Verify Setup

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/

# Start the application
streamlit run app.py
```

## Contributing Process

### 1. Create an Issue (for new features)

Before starting work on a new feature:
1. Check existing issues to avoid duplication
2. Create a new issue describing the feature
3. Wait for maintainer feedback before starting implementation

### 2. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow our [coding standards](#coding-standards)
- Write or update tests for your changes
- Update documentation as needed
- Ensure pre-commit hooks pass

### 4. Test Your Changes

```bash
# Run the full test suite
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run linting and formatting
ruff check .
black .
mypy src/
```

### 5. Commit Your Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Examples of good commit messages
git commit -m "feat: add support for PostgreSQL array types"
git commit -m "fix: handle edge case in SQL parser"
git commit -m "docs: update API documentation"
git commit -m "test: add unit tests for database module"
git commit -m "refactor: simplify query generation logic"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Build/tooling changes

### 6. Push and Create Pull Request

```bash
# Push your branch
git push origin your-branch-name

# Create a Pull Request on GitHub
# Use the PR template and provide detailed description
```

## Coding Standards

### Python Code Style

We use the following tools for code quality:

- **Black**: Code formatting (line length: 88)
- **Ruff**: Fast Python linter
- **isort**: Import sorting
- **mypy**: Type checking
- **Bandit**: Security linting

### Style Guidelines

```python
# Use type hints
def process_query(query: str, dialect: str = "postgresql") -> Dict[str, Any]:
    """Process natural language query and return SQL.
    
    Args:
        query: Natural language query string
        dialect: SQL dialect to use
        
    Returns:
        Dictionary containing generated SQL and metadata
        
    Raises:
        ValueError: If query is empty or invalid
    """
    pass

# Use dataclasses for structured data
@dataclass
class QueryResult:
    sql: str
    execution_time: float
    row_count: int
    
# Use context managers for resources
def get_database_connection():
    with DatabaseManager() as db:
        yield db.connection
```

### Documentation Strings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """Example function with types documented in the docstring.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    Raises:
        ValueError: If param1 is empty.
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with external dependencies
├── performance/    # Performance and load tests
├── fixtures/       # Test data and utilities
└── conftest.py     # Pytest configuration
```

### Writing Tests

```python
# Unit test example
def test_query_parser_basic():
    """Test basic query parsing functionality."""
    parser = QueryParser()
    result = parser.parse("Show me all users")
    
    assert result.table == "users"
    assert "SELECT" in result.sql
    assert result.confidence > 0.8

# Integration test example
@pytest.mark.integration
def test_database_connection():
    """Test actual database connectivity."""
    with DatabaseManager() as db:
        result = db.execute("SELECT 1")
        assert result is not None

# Performance test example
@pytest.mark.performance
def test_query_generation_performance():
    """Test query generation performance."""
    start_time = time.time()
    
    # Generate 100 queries
    for _ in range(100):
        generate_sql("Show me users")
    
    elapsed = time.time() - start_time
    assert elapsed < 5.0  # Should complete in under 5 seconds
```

### Test Coverage

- Aim for 80%+ code coverage
- Focus on critical paths and edge cases
- Test both success and failure scenarios
- Include performance tests for critical features

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and comments
2. **API Documentation**: Auto-generated from docstrings
3. **User Documentation**: Guides and tutorials
4. **Architecture Documentation**: System design and decisions

### Documentation Standards

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include code examples where helpful
- Add diagrams for complex concepts

### Building Documentation

```bash
# Generate API documentation
pdoc src/ --output-dir docs/api/

# Check documentation links
markdown-link-check README.md
```

## Security

### Security Guidelines

- Never commit secrets or API keys
- Use parameterized queries to prevent SQL injection
- Validate all user inputs
- Follow secure coding practices
- Report security issues privately to maintainers

### Security Testing

```bash
# Run security scans
bandit -r src/

# Check for known vulnerabilities
safety check

# Scan for secrets
git-secrets --scan
```

## Community

### Getting Help

- 📖 **Documentation**: Check our docs first
- 💬 **Discussions**: GitHub Discussions for questions
- 🐛 **Issues**: GitHub Issues for bugs and features
- 📧 **Email**: Contact maintainers for security issues

### Communication Guidelines

- Be respectful and inclusive
- Provide context and details in issues
- Use clear, descriptive titles
- Follow up on your contributions
- Help others when possible

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Annual contributor appreciation posts

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Security scan passed
- [ ] Performance benchmarks run

---

## Questions?

If you have questions not covered in this guide:

1. Check existing [GitHub Discussions](https://github.com/danieleschmidt/sql-synth-agentic-playground/discussions)
2. Create a new discussion
3. Tag maintainers if urgent

Thank you for contributing to SQL Synth Agentic Playground! 🚀