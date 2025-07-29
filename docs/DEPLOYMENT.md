# Deployment Guide

## Overview

This guide covers deployment strategies, environments, and operational procedures for the SQL Synthesis Agentic Playground.

## Deployment Environments

### Development Environment

**Purpose**: Local development and testing

```bash
# Clone repository
git clone https://github.com/danieleschmidt/sql-synth-agentic-playground.git
cd sql-synth-agentic-playground

# Install dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with local configuration

# Start development server
streamlit run app.py
```

**Development Stack**:
- Python 3.9+ with virtual environment
- SQLite for local database
- Streamlit development server
- Local benchmark data cache

### Staging Environment

**Purpose**: Pre-production testing and validation

**Infrastructure**:
- Docker containers
- PostgreSQL database
- Redis for caching
- Basic monitoring

**Deployment**:
```bash
# Build and deploy staging
docker-compose -f docker-compose.staging.yml up -d

# Verify deployment
curl -f http://staging.sqlsynth.local/health

# Run smoke tests
pytest tests/integration/ -v
```

### Production Environment

**Purpose**: Live application serving users

**Infrastructure Requirements**:
- Container orchestration (Kubernetes/Docker Swarm)
- Load balancer (NGINX/HAProxy)
- PostgreSQL cluster with read replicas
- Redis cluster for caching
- Monitoring stack (Prometheus/Grafana)
- Log aggregation (ELK/CloudWatch)

## Container Deployment

### Docker Configuration

**Dockerfile** (production-ready):
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash sqlsynth
RUN chown -R sqlsynth:sqlsynth /app
USER sqlsynth

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Compose Production**:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/sqlsynth
      - REDIS_URL=redis://redis:6379/0
      - ENV=production
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data:ro
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: sqlsynth
      POSTGRES_USER: user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment

### Namespace and Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sql-synthesis
  labels:
    name: sql-synthesis

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: sql-synthesis
data:
  ENV: "production"
  LOG_LEVEL: "INFO"
  STREAMLIT_SERVER_PORT: "8501"
  STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
```

### Application Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sql-synthesis-app
  namespace: sql-synthesis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sql-synthesis
  template:
    metadata:
      labels:
        app: sql-synthesis
    spec:
      containers:
      - name: app
        image: sql-synthesis:latest
        ports:
        - containerPort: 8501
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        envFrom:
        - configMapRef:
            name: app-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sql-synthesis-service
  namespace: sql-synthesis
spec:
  selector:
    app: sql-synthesis
  ports:
  - port: 80
    targetPort: 8501
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sql-synthesis-ingress
  namespace: sql-synthesis
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - sqlsynth.example.com
    secretName: sql-synthesis-tls
  rules:
  - host: sqlsynth.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sql-synthesis-service
            port:
              number: 80
```

## Database Deployment

### PostgreSQL Configuration

**Production PostgreSQL Setup**:
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: sql-synthesis
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: sqlsynth
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

### Database Migrations

```bash
# Database migration script
#!/bin/bash
set -e

echo "Running database migrations..."

# Apply schema migrations
alembic upgrade head

# Load benchmark data if needed
python scripts/load_benchmark_data.py

# Verify database health
python scripts/verify_db_health.py

echo "Database migration completed successfully"
```

## Monitoring Deployment

### Prometheus Configuration

```yaml
# k8s/monitoring.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: sql-synthesis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-data
          mountPath: /prometheus
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
      volumes:
      - name: prometheus-config
        configMap:
          name: prometheus-config
      - name: prometheus-data
        persistentVolumeClaim:
          claimName: prometheus-data
```

## Deployment Automation

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
    
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      run: |
        # Update image tag in deployment
        sed -i "s|image: sql-synthesis:latest|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}|" k8s/deployment.yaml
        
        # Apply Kubernetes manifests
        kubectl apply -f k8s/
        
        # Wait for rollout to complete
        kubectl rollout status deployment/sql-synthesis-app -n sql-synthesis
```

### Blue-Green Deployment

```bash
#!/bin/bash
# blue-green-deploy.sh

set -e

NAMESPACE="sql-synthesis"
NEW_VERSION=$1
CURRENT_SLOT=$(kubectl get service sql-synthesis-service -n $NAMESPACE -o jsonpath='{.spec.selector.slot}')

if [ "$CURRENT_SLOT" = "blue" ]; then
    NEW_SLOT="green"
else
    NEW_SLOT="blue"
fi

echo "Current slot: $CURRENT_SLOT"
echo "Deploying to slot: $NEW_SLOT"

# Deploy to new slot
kubectl set image deployment/sql-synthesis-app-$NEW_SLOT app=sql-synthesis:$NEW_VERSION -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/sql-synthesis-app-$NEW_SLOT -n $NAMESPACE

# Health check
kubectl exec -n $NAMESPACE deployment/sql-synthesis-app-$NEW_SLOT -- curl -f http://localhost:8501/health

# Switch traffic
kubectl patch service sql-synthesis-service -n $NAMESPACE -p '{"spec":{"selector":{"slot":"'$NEW_SLOT'"}}}'

echo "Deployment completed successfully"
echo "New active slot: $NEW_SLOT"
```

## Security Considerations

### Container Security

1. **Base Image Security**
   - Use official Python slim images
   - Regular security updates
   - Vulnerability scanning

2. **Runtime Security**
   - Non-root user
   - Read-only root filesystem
   - Security contexts

3. **Secrets Management**
   - Kubernetes secrets
   - External secret management (Vault)
   - Environment variable encryption

### Network Security

1. **TLS/SSL**
   - HTTPS termination at load balancer
   - Internal TLS for sensitive data
   - Certificate management

2. **Network Policies**
   - Kubernetes network policies
   - Firewall rules
   - Service mesh (optional)

## Troubleshooting

### Common Deployment Issues

1. **Container Won't Start**
   ```bash
   # Check pod status
   kubectl get pods -n sql-synthesis
   
   # View logs
   kubectl logs deployment/sql-synthesis-app -n sql-synthesis
   
   # Describe pod for events
   kubectl describe pod <pod-name> -n sql-synthesis
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   kubectl exec -it deployment/sql-synthesis-app -n sql-synthesis -- python -c "
   from src.sql_synth.database import get_database_manager
   db = get_database_manager()
   print('Database connection:', db.test_connection())
   "
   ```

3. **Performance Issues**
   ```bash
   # Check resource usage
   kubectl top pods -n sql-synthesis
   
   # View detailed metrics
   kubectl describe node <node-name>
   ```

### Rollback Procedures

```bash
# Rollback to previous version
kubectl rollout undo deployment/sql-synthesis-app -n sql-synthesis

# Rollback to specific revision
kubectl rollout undo deployment/sql-synthesis-app --to-revision=2 -n sql-synthesis

# Check rollout history
kubectl rollout history deployment/sql-synthesis-app -n sql-synthesis
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Create database backup
kubectl exec -n sql-synthesis postgres-0 -- pg_dump -U $DB_USER $DB_NAME > $BACKUP_DIR/database.sql

# Upload to S3 (or your backup storage)
aws s3 cp $BACKUP_DIR s3://backups/sql-synthesis/ --recursive

echo "Backup completed: $BACKUP_DIR"
```

### Disaster Recovery

1. **RTO**: 1 hour (Recovery Time Objective)
2. **RPO**: 15 minutes (Recovery Point Objective)
3. **Backup Strategy**: Daily full backups, continuous WAL archiving
4. **Recovery Testing**: Monthly disaster recovery drills

## References

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Docker Security](https://docs.docker.com/engine/security/security/)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [PostgreSQL High Availability](https://www.postgresql.org/docs/current/high-availability.html)