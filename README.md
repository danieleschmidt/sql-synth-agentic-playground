# SQL Synth Agentic Playground

Natural language to SQL translation with a benchmark evaluation framework and Streamlit UI. No LLM required — pure rule-based translation engine.

## Features

- **NL→SQL Translation**: Regex pattern-based translator covering SELECT, WHERE, ORDER BY, LIMIT, GROUP BY, aggregates (COUNT/SUM/AVG/MAX/MIN)
- **Benchmark Framework**: Evaluate against Spider-style examples, compute exact-match and structural-match rates
- **Streamlit UI**: Interactive playground with translation, benchmark running, and SQL execution
- **Demo Database**: Pre-loaded SQLite with employee, product, and customer tables
- **No external deps** for core translation — just Python stdlib

## Install

```bash
pip install -e ".[ui]"
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

### Python API

```python
from sql_synth.nl2sql import translate
from sql_synth.benchmark import run_benchmark

# Translate a query
result = translate("Show all employees where salary > 80000")
print(result.sql)  # SELECT * FROM employee WHERE salary > 80000

# Run benchmark
report = run_benchmark()
print(f"Structural match rate: {report.structural_match_rate:.0%}")
```

## Supported Query Patterns

| Pattern | Example | Generated SQL |
|---|---|---|
| Select all | "List all employees" | `SELECT * FROM employee` |
| Count | "How many customers" | `SELECT COUNT(*) FROM customer` |
| Where equals | "Find users where country is USA" | `SELECT * FROM user WHERE country = 'USA'` |
| Where > | "Find products where price > 50" | `SELECT * FROM product WHERE price > 50` |
| Order by | "Show employees ordered by salary desc" | `SELECT * FROM employee ORDER BY salary DESC` |
| Limit | "Get top 5 products" | `SELECT * FROM product LIMIT 5` |
| Avg | "Average salary of employees" | `SELECT AVG(salary) FROM employee` |
| Group by | "Count orders per status" | `SELECT status, COUNT(*) FROM order GROUP BY status` |

## Benchmark

The built-in benchmark includes 10 Spider-style examples. Run it via the UI or:

```python
from sql_synth.benchmark import run_benchmark
report = run_benchmark()
print(f"Exact match: {report.exact_match_rate:.0%}")
print(f"Structural match: {report.structural_match_rate:.0%}")
```

You can also load custom examples:
```python
from sql_synth.benchmark import load_custom_benchmark, run_benchmark
examples = load_custom_benchmark("my_benchmark.json")
report = run_benchmark(examples)
```

## Development

```bash
pip install pytest
pytest tests/
```
