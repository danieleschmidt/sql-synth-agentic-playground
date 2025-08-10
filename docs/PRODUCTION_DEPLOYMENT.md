# 🚀 Production Deployment Guide

This guide covers the complete deployment process for the SQL Synthesis Agentic Playground in production environments.

## Prerequisites

### System Requirements
- **Kubernetes cluster** (v1.24+)
- **Docker** (v20.10+)
- **kubectl** (v1.24+)
- **Helm** (v3.8+)
- **4 CPU cores** and **8GB RAM** minimum per node

### Required Credentials
- Container registry access (GitHub Container Registry, Docker Hub, etc.)
- Kubernetes cluster access (kubeconfig file)
- OpenAI API key for SQL generation
- Database connection credentials

## Quick Start

### 1. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export DATABASE_URL="postgresql://user:pass@host:port/db"
export DOCKER_REGISTRY="ghcr.io/your-org"
export NAMESPACE="sql-synth-prod"
```

### 2. Run Deployment Script

```bash
chmod +x scripts/deploy_production.sh
./scripts/deploy_production.sh
```

The script will automatically:
- Build and push Docker images
- Deploy to Kubernetes
- Run health checks
- Perform smoke tests
- Handle rollbacks on failure

## Manual Deployment Steps

### Step 1: Build Docker Image

```bash
# Build production image
docker build \
  --file Dockerfile.production \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  --build-arg VERSION="$(git rev-parse --short HEAD)" \
  --tag ghcr.io/sql-synth-agentic-playground:latest \
  .

# Push to registry
docker push ghcr.io/sql-synth-agentic-playground:latest
```

### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace sql-synth-prod

# Deploy secrets
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: sql-synth-secrets
  namespace: sql-synth-prod
data:
  openai-api-key: $(echo -n "$OPENAI_API_KEY" | base64 -w 0)
  database-url: $(echo -n "$DATABASE_URL" | base64 -w 0)
EOF

# Deploy application
kubectl apply -f deployment/kubernetes/production-deployment.yaml

# Wait for rollout
kubectl rollout status deployment/sql-synth-app -n sql-synth-prod
```

### Step 3: Verify Deployment

```bash
# Check pod status
kubectl get pods -n sql-synth-prod

# Check service status
kubectl get services -n sql-synth-prod

# View logs
kubectl logs -f deployment/sql-synth-app -n sql-synth-prod
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for SQL generation | - | ✅ |
| `DATABASE_URL` | Primary database connection | `sqlite:///:memory:` | ❌ |
| `REDIS_URL` | Redis cache connection | `redis://redis-service:6379` | ❌ |
| `LOG_LEVEL` | Application log level | `INFO` | ❌ |
| `ENVIRONMENT` | Environment name | `production` | ❌ |

### Resource Configuration

```yaml
resources:
  requests:
    cpu: "1000m"
    memory: "2Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### Auto-scaling Configuration

```yaml
hpa:
  minReplicas: 3
  maxReplicas: 20
  targetCPU: 70
  targetMemory: 80
```

## Monitoring & Observability

### Health Checks

The application provides multiple health check endpoints:

- `GET /health` - Basic health status
- `GET /health/ready` - Readiness check
- `GET /health/live` - Liveness check

### Metrics Collection

Prometheus metrics are available at:
- `GET /metrics` on port 9090

Key metrics include:
- Request latency (p95, p99)
- Error rates
- SQL generation success rate
- Database connection pool status
- Cache hit rates

### Log Aggregation

Logs are structured JSON format with fields:
- `timestamp`
- `level`
- `message`
- `correlation_id`
- `user_id`
- `tenant_id`

## Security Configuration

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sql-synth-network-policy
spec:
  podSelector:
    matchLabels:
      app: sql-synth-agentic-playground
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8501
```

### RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sql-synth-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
```

## Backup & Disaster Recovery

### Database Backups

```bash
# Automated daily backups
kubectl create cronjob backup-database \
  --image=postgres:13 \
  --schedule="0 2 * * *" \
  -- pg_dump $DATABASE_URL > /backups/backup-$(date +%Y%m%d).sql
```

### Configuration Backups

```bash
# Backup Kubernetes resources
kubectl get all,secrets,configmaps -n sql-synth-prod -o yaml > backup-k8s.yaml
```

## Scaling Guidelines

### Horizontal Scaling

The application supports horizontal scaling with:
- Stateless architecture
- Session affinity for user experience
- Shared Redis cache for consistency

### Vertical Scaling

Resource requirements scale with:
- Concurrent users: +100MB RAM per 100 users
- Database connections: +50MB RAM per connection
- Cache size: Configure based on query volume

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures

```bash
# Check pod events
kubectl describe pod <pod-name> -n sql-synth-prod

# Check logs
kubectl logs <pod-name> -n sql-synth-prod
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it <pod-name> -n sql-synth-prod -- \
  python -c "from src.sql_synth.database import get_database_manager; get_database_manager().test_connection()"
```

#### 3. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n sql-synth-prod

# Scale up resources
kubectl patch deployment sql-synth-app -n sql-synth-prod -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"sql-synth-app","resources":{"limits":{"memory":"6Gi"}}}]}}}}'
```

### Log Analysis

Common log patterns to monitor:

```bash
# Error patterns
kubectl logs -f deployment/sql-synth-app -n sql-synth-prod | grep "ERROR"

# Performance issues
kubectl logs -f deployment/sql-synth-app -n sql-synth-prod | grep "generation_time.*[5-9]\."

# Security events
kubectl logs -f deployment/sql-synth-app -n sql-synth-prod | grep "security_violation"
```

## Performance Tuning

### Database Optimization

```python
# Connection pool tuning
database:
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
```

### Cache Optimization

```python
# Redis configuration
redis:
  max_connections: 50
  connection_pool_class: "BlockingConnectionPool"
  health_check_interval: 30
```

### Application Tuning

```python
# Streamlit configuration
streamlit:
  server.maxUploadSize: 200
  server.enableCORS: true
  server.enableXsrfProtection: true
```

## Maintenance Procedures

### Rolling Updates

```bash
# Update image
kubectl set image deployment/sql-synth-app \
  sql-synth-app=ghcr.io/sql-synth-agentic-playground:v2.0.0 \
  -n sql-synth-prod

# Monitor rollout
kubectl rollout status deployment/sql-synth-app -n sql-synth-prod
```

### Rollback Procedures

```bash
# Rollback to previous version
kubectl rollout undo deployment/sql-synth-app -n sql-synth-prod

# Rollback to specific revision
kubectl rollout undo deployment/sql-synth-app --to-revision=2 -n sql-synth-prod
```

### Database Migrations

```bash
# Run migrations
kubectl exec -it deployment/sql-synth-app -n sql-synth-prod -- \
  python manage.py migrate
```

## Cost Optimization

### Resource Right-sizing

Monitor and adjust resource requests/limits based on actual usage:

```bash
# Resource usage analysis
kubectl top pods -n sql-synth-prod --sort-by=memory
kubectl top pods -n sql-synth-prod --sort-by=cpu
```

### Auto-scaling Tuning

```yaml
# Cost-optimized HPA settings
hpa:
  minReplicas: 2  # Reduce minimum replicas during low traffic
  maxReplicas: 15  # Cap maximum replicas
  targetCPU: 80   # Higher threshold to reduce scaling frequency
  scaleDownStabilization: 300s  # Slower scale-down
```

## Compliance & Security

### Audit Logging

All security events are logged with structured data:

```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "event_type": "security_violation",
  "user_id": "user123",
  "tenant_id": "tenant456",
  "risk_score": 0.8,
  "details": "SQL injection attempt detected"
}
```

### Data Privacy

GDPR compliance features:
- PII detection and masking
- Data retention policies
- User data export capabilities
- Right to erasure implementation

### Security Scanning

Regular security scans:

```bash
# Container vulnerability scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ghcr.io/sql-synth-agentic-playground:latest

# Kubernetes security scanning
kubectl run kube-bench --rm -i --tty --restart=Never --image=aquasec/kube-bench:latest
```

---

For additional support, please refer to the troubleshooting guide or contact the development team.