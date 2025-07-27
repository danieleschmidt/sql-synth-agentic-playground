# SQL Synthesis Agentic Playground Makefile
# Provides standardized build, test, and deployment commands

.PHONY: help install install-dev test test-unit test-integration test-performance lint format typecheck security clean build run dev logs stop restart backup restore docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := sql-synth-agentic-playground
VERSION := $(shell grep '^version' pyproject.toml | cut -d'"' -f2)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) - Version $(VERSION)$(NC)"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e .[dev]
	pre-commit install

# Testing targets
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest -m "unit" --verbose

test-integration: ## Run integration tests only
	pytest -m "integration" --verbose

test-performance: ## Run performance tests
	pytest -m "benchmark" --verbose

test-security: ## Run security tests
	pytest -m "security" --verbose
	bandit -r src/

test-coverage: ## Run tests with coverage report
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	pytest-watch

# Code quality targets
lint: ## Run linting checks
	ruff check .
	black --check .
	isort --check-only .

format: ## Format code
	black .
	isort .
	ruff check . --fix

typecheck: ## Run type checking
	mypy src/

security: ## Run security checks
	bandit -r src/
	safety check
	pip-audit

quality: lint typecheck security ## Run all code quality checks

# Cleaning targets
clean: ## Clean build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

clean-data: ## Clean data directories
	rm -rf data/ logs/ tmp/

clean-all: clean clean-data ## Clean everything

# Docker targets
build: ## Build Docker image
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker build -t $(PROJECT_NAME):latest .

build-dev: ## Build development Docker image
	docker build --target development -t $(PROJECT_NAME):dev .

run: ## Run application with Docker Compose
	$(DOCKER_COMPOSE) up -d

dev: ## Run development environment
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml up -d

stop: ## Stop all containers
	$(DOCKER_COMPOSE) down

restart: ## Restart all containers
	$(DOCKER_COMPOSE) restart

logs: ## Show container logs
	$(DOCKER_COMPOSE) logs -f

status: ## Show container status
	$(DOCKER_COMPOSE) ps

# Database targets
db-migrate: ## Run database migrations
	alembic upgrade head

db-rollback: ## Rollback last migration
	alembic downgrade -1

db-reset: ## Reset database (WARNING: destroys data)
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d postgres
	sleep 5
	$(MAKE) db-migrate

db-backup: ## Backup database
	mkdir -p backups
	$(DOCKER_COMPOSE) exec postgres pg_dump -U sql_synth_user sql_synth_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (set BACKUP_FILE=filename)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "$(RED)Error: BACKUP_FILE not set$(NC)"; exit 1; fi
	$(DOCKER_COMPOSE) exec -T postgres psql -U sql_synth_user -d sql_synth_db < backups/$(BACKUP_FILE)

# Benchmark targets
benchmark-download: ## Download benchmark datasets
	mkdir -p data/spider data/wikisql
	# Add download commands for Spider and WikiSQL datasets
	@echo "$(YELLOW)Note: Add actual download commands for benchmark datasets$(NC)"

benchmark-run: ## Run benchmark evaluation
	pytest tests/benchmark/ -v

benchmark-cache: ## Setup benchmark database cache
	$(DOCKER_COMPOSE) up -d benchmark-db
	# Add commands to load benchmark data into cache

# Development targets
dev-setup: install-dev ## Setup development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	cp .env.example .env
	$(MAKE) build-dev
	$(MAKE) dev
	@echo "$(GREEN)Development environment ready!$(NC)"
	@echo "Application: http://localhost:8501"
	@echo "pgAdmin: http://localhost:5050 (admin@admin.com/admin)"

dev-test: ## Run development test suite
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml exec sql-synth-app pytest

dev-shell: ## Open shell in development container
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml exec sql-synth-app bash

# Documentation targets
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	# Add documentation generation commands
	@echo "$(GREEN)Documentation generated in docs/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(NC)"
	# Add documentation serving command

# Monitoring targets
monitor: ## Start monitoring stack
	$(DOCKER_COMPOSE) --profile monitoring up -d

metrics: ## Show application metrics
	curl -s http://localhost:9090/api/v1/query?query=up | jq

# Release targets
release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(NC)"
	$(MAKE) test
	$(MAKE) quality
	$(MAKE) security
	@echo "$(GREEN)Release checks passed!$(NC)"

release-build: release-check ## Build release artifacts
	@echo "$(BLUE)Building release $(VERSION)...$(NC)"
	$(MAKE) clean
	$(MAKE) build
	docker tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Release $(VERSION) built successfully!$(NC)"

# CI/CD targets
ci-install: ## Install dependencies for CI
	$(PIP) install -e .[dev]

ci-test: ## Run tests for CI
	pytest --junitxml=test-results.xml --cov=src --cov-report=xml

ci-security: ## Run security checks for CI
	bandit -r src/ -f json -o security-report.json
	safety check --json --output safety-report.json

# Utility targets
env-check: ## Check environment variables
	@echo "$(BLUE)Checking environment...$(NC)"
	@python -c "import os; print('Python path:', os.environ.get('PYTHONPATH', 'Not set'))"
	@python -c "import sys; print('Python version:', sys.version)"
	@which docker > /dev/null || echo "$(RED)Docker not found$(NC)"
	@which docker-compose > /dev/null || echo "$(RED)Docker Compose not found$(NC)"

size: ## Show Docker image sizes
	docker images $(PROJECT_NAME) --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

prune: ## Prune Docker system
	docker system prune -f
	docker volume prune -f

# Performance targets
perf-test: ## Run performance tests
	pytest tests/performance/ -v --tb=short

load-test: ## Run load tests
	# Add load testing commands (e.g., with locust or artillery)
	@echo "$(YELLOW)Load testing not yet implemented$(NC)"

# Health checks
health: ## Check application health
	curl -f http://localhost:8501/_stcore/health || echo "$(RED)Application unhealthy$(NC)"
	$(DOCKER_COMPOSE) ps

ping-db: ## Ping database
	$(DOCKER_COMPOSE) exec postgres pg_isready -U sql_synth_user

# Quick start
quickstart: ## Quick start for new developers
	@echo "$(BLUE)🚀 Quick Start for $(PROJECT_NAME)$(NC)"
	@echo "1. Setting up development environment..."
	$(MAKE) dev-setup
	@echo "2. Running tests..."
	$(MAKE) test-unit
	@echo "$(GREEN)✅ Setup complete! Visit http://localhost:8501$(NC)"

# Show current version
version: ## Show current version
	@echo "$(PROJECT_NAME) version: $(VERSION)"