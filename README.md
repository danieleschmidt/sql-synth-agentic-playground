# agentic-startup-studio-boilerplate

<!-- IMPORTANT: Replace 'your-github-username-or-org' with your actual GitHub details -->
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-github-username-or-org/agentic-startup-studio-boilerplate/ci.yml?branch=main)](https://github.com/your-github-username-or-org/agentic-startup-studio-boilerplate/actions)
[![Coverage Status](https://img.shields.io/coveralls/github/your-github-username-or-org/agentic-startup-studio-boilerplate)](https://coveralls.io/github/your-github-username-or-org/agentic-startup-studio-boilerplate)
[![License](https://img.shields.io/github/license/your-github-username-or-org/agentic-startup-studio-boilerplate)](LICENSE)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://semver.org)

A Cookiecutter template for rapidly building agentic startups. It provides a reusable skeleton that wires together CrewAI, FastAPI, and a React frontend with Shadcn UI components.

## ✨ Key Features

*   **Cookiecutter Template**: Quickly scaffold a new project with a standardized structure.
*   **Integrated Tech Stack**: Wires together CrewAI, FastAPI, and React with Shadcn.
*   **Pluggable Auth**: Includes an optional `docker-compose.keycloak.yml` stub for user authentication.
*   **Infrastructure-as-Code**: Provides a `/iac` folder with Terraform scripts to provision the backend, database, and frontend bucket.
*   **DX Polish**: A single `dev up` script spins up the entire development environment using Docker Compose.

## ⚠️ Important Notes

*   **Terraform State**: For multi-developer environments, you MUST configure remote state for Terraform (e.g., using an S3 bucket and DynamoDB lock table) to prevent conflicts. The default local state is for single-developer use only.
*   **Authentication**: To enable auth, uncomment the Keycloak service in the main `docker-compose.yml` and follow the [Keycloak quick-start guide](https://www.keycloak.org/getting-started/getting-started-docker) for setup.

## ⚡ Quick Start

1.  Install Cookiecutter: `pip install cookiecutter`
2.  Generate a new project: `cookiecutter gh:your-github-username-or-org/agentic-startup-studio-boilerplate`
3.  `cd my-new-app`
4.  Start the development environment: `./dev up`

## 📈 Roadmap

*   **v0.1.0**: Basic template with CrewAI, FastAPI, React, IaC, and dev script.
*   **v0.2.0**: Addition of more sophisticated venture scoring logic examples.
*   **v0.3.0**: Integration with other startup tools like Stripe for payments.

## 🤝 Contributing

We welcome contributions! Please see our organization-wide `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. A `CHANGELOG.md` is maintained.

## 📝 License

This project is licensed under the Apache-2.0 License.
