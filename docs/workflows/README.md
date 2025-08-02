# GitHub Workflows Documentation & Templates

## Overview

This directory contains comprehensive GitHub Actions workflow templates and documentation for the SQL Synth Agentic Playground project. Due to GitHub App permission limitations, these workflows must be manually created by repository maintainers.

## 📝 Available Workflow Templates

### Core Workflows

| Workflow | File | Purpose | Priority |
|----------|------|---------|----------|
| **CI/CD Pipeline** | `examples/ci.yml` | Comprehensive testing, building, and deployment | 🔴 Critical |
| **Security Scanning** | `examples/security-scan.yml` | Multi-tool security analysis and vulnerability detection | 🔴 Critical |
| **Dependency Management** | `examples/dependency-update.yml` | Automated dependency updates and security patches | 🟡 High |

### Workflow Features

#### CI/CD Pipeline (`ci.yml`)
- **Multi-stage Testing**: Unit, integration, security, and performance tests
- **Code Quality**: Automated linting, type checking, and formatting validation
- **Security Integration**: SAST, dependency scanning, and container security
- **Docker Building**: Multi-platform container builds with security scanning
- **SBOM Generation**: Software Bill of Materials for supply chain security
- **Deployment Automation**: Staging and production deployment with smoke tests

#### Security Scanning (`security-scan.yml`)
- **SAST Tools**: Bandit, Semgrep, and CodeQL integration
- **Dependency Scanning**: Safety, Snyk, and OSV-Scanner for vulnerability detection
- **Container Security**: Trivy, Grype, and Docker Scout scanning
- **Secret Detection**: Gitleaks and TruffleHog for exposed secrets
- **IaC Security**: Checkov for infrastructure as code validation
- **Reporting**: Automated security reports and PR comments

#### Dependency Management (`dependency-update.yml`)
- **Automated Updates**: Patch, minor, and major version updates
- **Security Prioritization**: Immediate updates for vulnerable packages
- **License Compliance**: Automated license compatibility checking
- **Testing Integration**: Full test suite validation after updates
- **PR Automation**: Automated pull request creation with detailed summaries

## 🚀 Manual Setup Required

### 1. Create Workflow Files

Copy the template files to your `.github/workflows/` directory:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy templates (repository maintainers must do this manually)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add required secrets in GitHub repository settings:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password or access token
- `CODECOV_TOKEN` - Codecov integration token
- `SNYK_TOKEN` - Snyk vulnerability scanning (optional)
- `SLACK_WEBHOOK_URL` - Slack notifications (optional)

### 3. Configure Branch Protection

Set up branch protection rules for `main` branch:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to protected branches

### 4. Repository Settings

Configure the following:
- **Topics**: `python`, `sql`, `ai`, `langchain`, `streamlit`, `nlp`, `database`
- **Security Features**: Enable all available security features
- **Actions Permissions**: Allow actions and reusable workflows
- **Dependabot**: Enable security updates

## 📄 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Workflow Security](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

---

📝 **Note**: Repository maintainers must manually create these workflow files due to GitHub App permission limitations.