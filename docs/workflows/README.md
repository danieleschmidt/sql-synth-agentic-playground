# Workflow Requirements

## Manual Setup Required

Due to permission limitations, the following GitHub Actions workflows require manual setup:

### Essential Workflows

1. **CI/CD Pipeline** (`.github/workflows/ci.yml`)
   - Automated testing on pull requests
   - Code quality checks (ruff, black, mypy)
   - Security scanning (bandit, safety)

2. **Release Automation** (`.github/workflows/release.yml`)
   - Automated versioning and releases
   - Changelog generation
   - Package publishing

3. **Security Scanning** (`.github/workflows/security.yml`)
   - Dependency vulnerability scanning
   - Container security checks
   - SAST scanning

### Branch Protection

Configure branch protection rules for `main`:
- Require pull request reviews
- Require status checks
- Require branches to be up to date
- Restrict pushes to protected branches

### Repository Settings

- Add repository topics: python, sql, ai, langchain, streamlit
- Set homepage URL
- Enable security features in Settings > Security

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)