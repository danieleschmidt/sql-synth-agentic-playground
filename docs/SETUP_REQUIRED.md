# Manual Setup Requirements

## Repository Configuration

### 1. Branch Protection Rules
- Navigate to Settings > Branches
- Add rule for `main` branch:
  - Require pull request reviews
  - Require status checks to pass
  - Require branches to be up to date
  - Restrict pushes

### 2. GitHub Actions Workflows
Due to permission restrictions, manually create:
- `.github/workflows/ci.yml` - CI/CD pipeline
- `.github/workflows/security.yml` - Security scanning
- `.github/workflows/release.yml` - Release automation

### 3. Repository Settings
- Add topics: python, sql, ai, langchain, streamlit
- Set description and homepage URL
- Enable vulnerability alerts
- Configure security advisories

### 4. Integrations
- Enable Dependabot alerts
- Configure CodeQL analysis
- Set up container scanning

## External Services

### Monitoring Setup
- Configure application monitoring
- Set up error tracking
- Enable performance monitoring

### Security Services
- Register with security scanning tools
- Configure SAST integration
- Set up dependency monitoring

For detailed workflow configurations, see `docs/workflows/README.md`.