# Development Guide

## Quick Start

### Prerequisites
- Python 3.9+
- Git
- Docker (optional)

### Setup Commands

```bash
# Clone and setup
git clone <repo-url>
cd sql-synth-agentic-playground

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Run application
streamlit run app.py
```

### Testing

```bash
pytest                    # Run all tests
pytest tests/unit/        # Unit tests only
pytest --cov=src          # With coverage
```

### Code Quality

```bash
ruff check .              # Linting
black .                   # Formatting
mypy src/                 # Type checking
```

For detailed information, see [CONTRIBUTING.md](../CONTRIBUTING.md).