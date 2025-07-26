# sql-synth-agentic-playground

[![Build Status](https://img.shields.io/github/actions/workflow/status/danieleschmidt/sql-synth-agentic-playground/ci.yml?branch=main)](https://github.com/danieleschmidt/sql-synth-agentic-playground/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/danieleschmidt/sql-synth-agentic-playground)](https://coveralls.io/github/danieleschmidt/sql-synth-agentic-playground)
[![License](https://img.shields.io/github/license/danieleschmidt/sql-synth-agentic-playground)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

An interactive playground for an agent that translates natural language queries into SQL. This project adds a comprehensive evaluation framework against standard benchmarks like Spider and WikiSQL, served via a Streamlit UI.

## ✨ Key Features

*   **Natural Language to SQL**: An agent capable of generating complex SQL queries from natural language prompts.
*   **Comprehensive Evaluation**: Evaluates accuracy against the Spider and WikiSQL benchmarks.
*   **Interactive UI**: A Streamlit interface for demonstrating and interacting with the agent.
*   **SQL Dialect Support**: The agent can select connectors for different dialects. Test cases should account for quirks (e.g., Snowflake's double-quoted identifiers and `ILIKE`).
*   **Fast CI Runs**: Caches the Spider/WikiSQL databases in a Docker volume to ensure benchmark runs are consistently fast.

## 🔐 Security

**SQL Injection Guard**: All generated queries use parameterized bindings to prevent SQL injection vulnerabilities. Direct string formatting of user input into SQL queries is strictly forbidden. For reporting vulnerabilities, please refer to our organization's `SECURITY.md` file.

## ⚡ Quick Start

1.  Clone the repository and install dependencies: `pip install -r requirements.txt`.
2.  Set up your database connection in a `.env` file.
3.  Cache the benchmark databases: `docker compose up -d benchmark-db`.
4.  Run the Streamlit app: `streamlit run app.py`.

## 📈 Roadmap

*   **v0.1.0**: Core SQL generation agent and Streamlit UI.
*   **v0.2.0**: Integration of the Spider and WikiSQL evaluation benchmarks.
*   **v0.3.0**: Support for more complex SQL features and database dialects.

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained.

## 📝 License

This project is licensed under the MIT License. The Spider dataset is licensed under CC BY-SA 4.0.

## 📚 References

*   **Spider Dataset**: [arXiv:1809.08887](https://arxiv.org/abs/1809.08887)
*   **LangChain SQL Tool API**: [SQL Agent Toolkit Reference](https://api.python.langchain.com/en/latest/agents/langchain_community.agent_toolkits.sql.base.create_sql_agent.html)```
