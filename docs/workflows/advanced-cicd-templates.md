# Advanced CI/CD Workflow Templates

## Overview

This document provides comprehensive CI/CD workflow templates for advanced automation, security, and deployment strategies. These templates implement enterprise-grade practices with SLSA compliance and comprehensive quality gates.

## Core CI Pipeline Template

### `.github/workflows/ci.yml`
```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.7.1"

permissions:
  contents: read
  security-events: write
  pull-requests: write
  id-token: write  # For SLSA attestation

jobs:
  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comprehensive scanning

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install bandit[toml] safety pip-audit

      - name: Run Bandit Security Scan
        run: |
          bandit -r src/ -f json -o bandit-report.json || true
          bandit -r src/ -f txt

      - name: Run Safety Check
        run: safety check --json --output safety-report.json || true

      - name: Run Pip Audit
        run: pip-audit --format=json --output=pip-audit-report.json || true

      - name: Upload security reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            pip-audit-report.json

  code-quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run Ruff Linter
        run: ruff check . --output-format=github

      - name: Run Ruff Formatter
        run: ruff format --check .

      - name: Run MyPy Type Checking
        run: mypy src/

      - name: Check Import Sorting
        run: isort --check-only --diff .

      - name: Run Pydocstyle
        run: pydocstyle src/

      - name: Run Interrogate (Documentation Coverage)
        run: interrogate -v --ignore-init-method --ignore-module --fail-under=80 src/

  testing:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]

      - name: Run Unit Tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html

      - name: Run Integration Tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        run: pytest tests/integration/ -v

      - name: Run Performance Tests
        run: pytest tests/performance/ -v --benchmark-json=benchmark.json

      - name: Upload Coverage Reports
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

      - name: Upload Test Artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            htmlcov/
            benchmark.json
            pytest-report.xml

  build-and-test:
    name: Build and Container Test
    runs-on: ubuntu-latest
    needs: [security-scan, code-quality, testing]
    outputs:
      image: ${{ steps.image.outputs.image }}
      digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Container Image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: sql-synth:test
          cache-from: type=gha
          cache-to: type=gha,mode=max
          outputs: type=docker,dest=/tmp/image.tar

      - name: Load and Test Container
        run: |
          docker load --input /tmp/image.tar
          docker run --rm sql-synth:test python -c "import src.sql_synth; print('Container build successful')"

      - name: Run Container Security Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: sql-synth:test
          format: sarif
          output: trivy-results.sarif

      - name: Upload Trivy Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: sql-synth:test
          format: spdx-json
          output-file: sbom.spdx.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json

  attestation:
    name: Generate SLSA Attestation
    runs-on: ubuntu-latest
    needs: [build-and-test]
    if: github.ref == 'refs/heads/main'
    permissions:
      id-token: write
      contents: read
      attestations: write
    
    steps:
      - name: Generate SLSA Attestation
        uses: actions/attest-build-provenance@v1
        with:
          subject-name: sql-synth-agentic-playground
          subject-digest: ${{ needs.build-and-test.outputs.digest }}
```

## Advanced Release Pipeline

### `.github/workflows/release.yml`
```yaml
name: Release Pipeline

on:
  push:
    tags: ['v*']
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v1.0.0)'
        required: true
        type: string

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  validate-release:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Validate Version Tag
        id: version
        run: |
          if [[ "${GITHUB_REF}" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            VERSION=${GITHUB_REF#refs/tags/}
          elif [[ "${{ github.event.inputs.version }}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            VERSION=${{ github.event.inputs.version }}
          else
            echo "Invalid version format"
            exit 1
          fi
          echo "version=${VERSION}" >> $GITHUB_OUTPUT

      - name: Verify Changelog
        run: |
          if ! grep -q "${{ steps.version.outputs.version }}" CHANGELOG.md; then
            echo "Version not found in CHANGELOG.md"
            exit 1
          fi

  security-review:
    name: Pre-Release Security Review
    runs-on: ubuntu-latest
    needs: validate-release
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Comprehensive Security Scan
        run: |
          python scripts/security_scan.py --comprehensive --output security-review.json

      - name: Verify Security Thresholds
        run: |
          # Custom script to verify security metrics meet release thresholds
          python -c "
          import json
          with open('security-review.json') as f:
              data = json.load(f)
          
          if data['critical_vulnerabilities'] > 0:
              raise Exception('Critical vulnerabilities found')
          if data['high_vulnerabilities'] > 2:
              raise Exception('Too many high vulnerabilities')
          "

  build-release:
    name: Build Release Artifacts
    runs-on: ubuntu-latest
    needs: [validate-release, security-review]
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Build Python Package
        run: |
          pip install build
          python -m build

      - name: Build Container Images
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: false
          tags: |
            ghcr.io/${{ github.repository }}:${{ needs.validate-release.outputs.version }}
            ghcr.io/${{ github.repository }}:latest
          outputs: type=oci,dest=container-image.tar

      - name: Generate Artifact Hashes
        id: hash
        run: |
          cd dist/
          sha256sum * > ../checksums.txt
          echo "hashes=$(base64 -w0 < ../checksums.txt)" >> $GITHUB_OUTPUT

      - name: Upload Release Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: release-artifacts
          path: |
            dist/
            container-image.tar
            checksums.txt

  generate-provenance:
    name: Generate SLSA Provenance
    needs: [build-release]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.10.0
    with:
      base64-subjects: "${{ needs.build-release.outputs.hashes }}"
      upload-assets: true

  publish-release:
    name: Publish Release
    runs-on: ubuntu-latest
    needs: [validate-release, build-release, generate-provenance]
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Download Artifacts
        uses: actions/download-artifact@v4
        with:
          name: release-artifacts
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Publish Container Images
        run: |
          docker load < container-image.tar
          docker push ghcr.io/${{ github.repository }}:${{ needs.validate-release.outputs.version }}
          docker push ghcr.io/${{ github.repository }}:latest

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ needs.validate-release.outputs.version }}
          name: Release ${{ needs.validate-release.outputs.version }}
          body_path: CHANGELOG.md
          files: |
            dist/*
            checksums.txt
          draft: false
          prerelease: false

  post-release:
    name: Post-Release Tasks
    runs-on: ubuntu-latest
    needs: [publish-release]
    
    steps:
      - name: Update Documentation
        run: |
          # Trigger documentation updates
          curl -X POST \
            -H "Authorization: token ${{ secrets.GITHUB_TOKEN }}" \
            -H "Accept: application/vnd.github.v3+json" \
            https://api.github.com/repositories/${{ github.repository }}/dispatches \
            -d '{"event_type":"docs-update"}'

      - name: Notify Stakeholders
        run: |
          echo "Release ${{ needs.validate-release.outputs.version }} published successfully"
          # Add notification logic (Slack, email, etc.)
```

## Infrastructure as Code Pipeline

### `.github/workflows/infrastructure.yml`
```yaml
name: Infrastructure Pipeline

on:
  push:
    paths: ['infrastructure/**']
    branches: [main]
  pull_request:
    paths: ['infrastructure/**']
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  plan:
    name: Terraform Plan
    runs-on: ubuntu-latest
    outputs:
      tfplan: ${{ steps.plan.outputs.tfplan }}
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
          cli_config_credentials_token: ${{ secrets.TF_API_TOKEN }}

      - name: Terraform Format Check
        run: terraform fmt -check -recursive infrastructure/

      - name: Terraform Init
        run: terraform init
        working-directory: infrastructure/

      - name: Terraform Validate
        run: terraform validate
        working-directory: infrastructure/

      - name: Terraform Plan
        id: plan
        run: |
          terraform plan -detailed-exitcode -out=tfplan
          echo "tfplan=$(base64 -w0 < tfplan)" >> $GITHUB_OUTPUT
        working-directory: infrastructure/

      - name: Upload Terraform Plan
        uses: actions/upload-artifact@v4
        with:
          name: terraform-plan
          path: infrastructure/tfplan

  security-scan:
    name: Infrastructure Security Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: infrastructure/
          framework: terraform
          output_format: sarif
          output_file_path: checkov-results.sarif

      - name: Upload Checkov Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: checkov-results.sarif

  apply:
    name: Terraform Apply
    runs-on: ubuntu-latest
    needs: [plan, security-scan]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
          cli_config_credentials_token: ${{ secrets.TF_API_TOKEN }}

      - name: Download Terraform Plan
        uses: actions/download-artifact@v4
        with:
          name: terraform-plan
          path: infrastructure/

      - name: Terraform Apply
        run: terraform apply tfplan
        working-directory: infrastructure/
```

## Security-First Pipeline

### `.github/workflows/security.yml`
```yaml
name: Security Pipeline

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install pip-audit safety

      - name: Run Pip Audit
        run: |
          pip-audit --format=json --output=pip-audit.json
          pip-audit --format=sarif --output=pip-audit.sarif

      - name: Run Safety Check
        run: |
          safety check --json --output safety.json
          safety check --output safety.txt

      - name: Upload SARIF Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: pip-audit.sarif

  sast-scan:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: python

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          publishToken: ${{ secrets.SEMGREP_APP_TOKEN }}

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Container
        run: docker build -t security-scan:latest .

      - name: Run Trivy Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: security-scan:latest
          format: sarif
          output: trivy-results.sarif

      - name: Upload Trivy Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif

  compliance-check:
    name: Compliance Verification
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: SLSA Compliance Check
        run: |
          python scripts/slsa_compliance_check.py

      - name: License Compliance
        uses: fossa-contrib/fossa-action@v2
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
```

## Performance Testing Pipeline

### `.github/workflows/performance.yml`
```yaml
name: Performance Testing

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM
  workflow_dispatch:
  pull_request:
    paths: ['src/**']

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest-benchmark

      - name: Run Benchmarks
        run: |
          pytest tests/performance/ --benchmark-json=benchmark.json

      - name: Performance Regression Check
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: pytest
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '120%'
          comment-on-alert: true
          fail-on-alert: true

  load-test:
    name: Load Testing
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install locust

      - name: Start Application
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        run: |
          streamlit run app.py --server.port 8501 &
          sleep 10

      - name: Run Load Tests
        run: |
          locust -f tests/load/locustfile.py --host=http://localhost:8501 \
                 --users=50 --spawn-rate=5 --run-time=300s --html=load-test-report.html

      - name: Upload Load Test Report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: load-test-report
          path: load-test-report.html
```

## Deployment Strategies

### Blue-Green Deployment
```yaml
name: Blue-Green Deployment

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options: ['staging', 'production']

jobs:
  deploy:
    name: Blue-Green Deploy
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}
    
    steps:
      - name: Deploy to Green Environment
        run: |
          # Deploy new version to green environment
          kubectl apply -f k8s/green-deployment.yaml

      - name: Health Check Green Environment
        run: |
          # Wait for green environment to be healthy
          kubectl wait --for=condition=ready pod -l app=sql-synth,version=green --timeout=300s

      - name: Run Smoke Tests
        run: |
          # Run smoke tests against green environment
          python tests/smoke/test_green_deployment.py

      - name: Switch Traffic to Green
        run: |
          # Update service to point to green environment
          kubectl patch service sql-synth-service -p '{"spec":{"selector":{"version":"green"}}}'

      - name: Monitor Traffic Switch
        run: |
          # Monitor traffic and error rates
          python scripts/monitor_deployment.py --duration=300

      - name: Cleanup Blue Environment
        if: success()
        run: |
          # Scale down blue environment after successful deployment
          kubectl scale deployment sql-synth-blue --replicas=0
```

## Usage Instructions

### Quick Start
1. Copy the desired workflow template to `.github/workflows/`
2. Customize environment variables and secrets
3. Configure branch protection rules
4. Set up required secrets in repository settings

### Required Secrets
```yaml
secrets:
  PYPI_API_TOKEN: # For package publishing
  GITHUB_TOKEN: # Automatically provided
  TF_API_TOKEN: # For Terraform Cloud
  FOSSA_API_KEY: # For license compliance
  SEMGREP_APP_TOKEN: # For SAST scanning
```

### Branch Protection Configuration
```yaml
branch_protection:
  main:
    required_status_checks:
      - security-scan
      - code-quality
      - testing
    required_reviews: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
```

---

*These advanced CI/CD templates provide enterprise-grade automation with security, compliance, and performance built-in from day one.*