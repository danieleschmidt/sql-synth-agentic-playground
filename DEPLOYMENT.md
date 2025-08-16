# SQL Synthesis Agent - Production Deployment Guide

## 🚀 Overview

This guide provides comprehensive instructions for deploying the SQL Synthesis Agent in production environments with enterprise-grade features including high availability, security, monitoring, and auto-scaling.

## 📋 Prerequisites

### System Requirements
- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB RAM, Recommended 8GB+ RAM
- **Storage**: Minimum 20GB, Recommended 50GB+ SSD
- **Network**: HTTPS/TLS termination, Load balancer support

### Dependencies
- Docker 20.10+
- Docker Compose 2.0+ (for Docker deployment)
- Kubernetes 1.24+ (for K8s deployment)
- Nginx (for reverse proxy)
- Redis 7+ (for distributed caching)
- PostgreSQL 13+ (recommended database)

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | - | Database connection string |
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for SQL generation |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `LOG_DIR` | No | /app/logs | Directory for log files |
| `LOG_JSON` | No | true | Enable JSON structured logging |
| `ENABLE_CACHING` | No | true | Enable intelligent caching |
| `CACHE_TTL` | No | 3600 | Cache TTL in seconds |
| `MAX_CONNECTIONS` | No | 20 | Maximum database connections |
| `HEALTH_CHECK_INTERVAL` | No | 30 | Health check interval in seconds |
| `REDIS_URL` | No | - | Redis URL for distributed caching |
| `MONITORING_STATE_FILE` | No | /app/data/monitoring.json | Monitoring state persistence |

### Security Configuration

```bash
# Generate secure secrets
export DATABASE_URL="postgresql://user:$(openssl rand -base64 32)@db.example.com/dbname"
export OPENAI_API_KEY="sk-your-openai-api-key"
export GRAFANA_PASSWORD="$(openssl rand -base64 24)"
```

## 🐳 Docker Deployment

### Quick Start

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd sql-synthesis-agent
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Deploy with Docker Compose:**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

3. **Verify deployment:**
   ```bash
   curl -f http://localhost:8501/health
   ```

### Production Docker Deployment

1. **Build production image:**
   ```bash
   docker build -f Dockerfile.production -t sql-synth-agent:latest .
   ```

2. **Deploy full stack:**
   ```bash
   # Start all services
   docker-compose -f docker-compose.production.yml up -d
   
   # Check status
   docker-compose -f docker-compose.production.yml ps
   
   # View logs
   docker-compose -f docker-compose.production.yml logs -f
   ```

3. **Monitor services:**
   - **Application**: http://localhost:8501
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000
   - **Redis**: localhost:6379

## ☸️ Kubernetes Deployment

### Prerequisites

1. **Install required tools:**
   ```bash
   kubectl version --client
   helm version
   ```

2. **Configure cluster access:**
   ```bash
   kubectl config current-context
   kubectl cluster-info
   ```

### Deployment Steps

1. **Create namespace and secrets:**
   ```bash
   kubectl create namespace sql-synth-agent
   
   # Create secrets
   kubectl create secret generic sql-synth-secrets \
     --from-literal=DATABASE_URL="your-database-url" \
     --from-literal=OPENAI_API_KEY="your-openai-key" \
     -n sql-synth-agent
   ```

2. **Deploy application:**
   ```bash
   kubectl apply -f k8s-deployment.yaml
   ```

3. **Verify deployment:**
   ```bash
   kubectl get all -n sql-synth-agent
   kubectl get ing -n sql-synth-agent
   ```

4. **Check pod status:**
   ```bash
   kubectl describe pods -n sql-synth-agent
   kubectl logs -f deployment/sql-synth-agent -n sql-synth-agent
   ```

### Auto-scaling Configuration

The deployment includes HorizontalPodAutoscaler (HPA) with:
- **Min replicas**: 3
- **Max replicas**: 10
- **CPU target**: 70%
- **Memory target**: 80%

Monitor scaling:
```bash
kubectl get hpa -n sql-synth-agent --watch
```

## 🔍 Monitoring & Observability

### Health Checks

The application provides multiple health check endpoints:

- `GET /health` - Basic health status
- `GET /health?ready` - Readiness probe
- `GET /health?live` - Liveness probe
- `GET /health?detailed` - Comprehensive health info
- `GET /metrics` - Prometheus metrics

### Metrics Collection

Key metrics monitored:

- **Performance**: Response times, cache hit rates, query success rates
- **System**: CPU, memory, disk usage
- **Security**: Failed validations, injection attempts
- **Application**: Active connections, queue depths, error rates

### Grafana Dashboards

Pre-configured dashboards include:

1. **Application Overview**: Key performance indicators
2. **System Resources**: Infrastructure monitoring
3. **Security Events**: Security violations and threats
4. **Cache Performance**: Cache hit rates and efficiency
5. **Database Connections**: Connection pool utilization

## 🛡️ Security Best Practices

### Network Security

1. **Use HTTPS/TLS:**
   ```nginx
   server {
       listen 443 ssl http2;
       ssl_certificate /etc/nginx/ssl/cert.pem;
       ssl_certificate_key /etc/nginx/ssl/key.pem;
   }
   ```

2. **Configure firewalls:**
   ```bash
   # Allow only necessary ports
   ufw allow 80,443/tcp
   ufw allow 22/tcp
   ufw deny incoming
   ```

### Application Security

1. **Input validation**: All user inputs are validated and sanitized
2. **SQL injection protection**: Multi-layered detection and prevention
3. **Rate limiting**: Configurable rate limits per IP/user
4. **Audit logging**: Comprehensive security event logging

### Container Security

1. **Non-root user**: Application runs as non-root user
2. **Read-only filesystem**: Root filesystem is read-only
3. **Security contexts**: Kubernetes security contexts configured
4. **Network policies**: Restrict pod-to-pod communication

## 📊 Performance Optimization

### Cache Configuration

```yaml
# Redis configuration for optimal performance
maxmemory: 1gb
maxmemory-policy: allkeys-lru
save: 900 1 300 10 60 10000
```

### Database Optimization

```sql
-- PostgreSQL optimization
CREATE INDEX CONCURRENTLY idx_query_hash ON queries USING hash(query_hash);
CREATE INDEX idx_created_at ON queries(created_at);

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
```

### Application Tuning

```bash
# Environment variables for performance
export MAX_CONNECTIONS=20
export CACHE_TTL=3600
export WORKER_THREADS=4
export ASYNC_POOL_SIZE=100
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage:**
   ```bash
   # Check cache utilization
   docker exec sql-synth-agent python -c "
   from src.sql_synth.advanced_cache import get_cache_statistics
   print(get_cache_statistics())
   "
   ```

2. **Database Connection Issues:**
   ```bash
   # Test database connectivity
   docker exec sql-synth-agent python -c "
   from src.sql_synth.database import get_database_manager
   db = get_database_manager()
   print(db.test_connection())
   "
   ```

3. **Performance Issues:**
   ```bash
   # Check performance metrics
   curl http://localhost:8501/health?detailed
   ```

### Log Analysis

```bash
# Application logs
docker logs sql-synth-agent -f

# System logs
journalctl -u docker -f

# Kubernetes logs
kubectl logs -f deployment/sql-synth-agent -n sql-synth-agent
```

### Health Check Debugging

```bash
# Manual health checks
curl -v http://localhost:8501/health
curl -v http://localhost:8501/health?ready
curl -v http://localhost:8501/health?detailed
```

## 📈 Scaling Guidelines

### Horizontal Scaling

- **Load balancer**: Use sticky sessions for WebSocket connections
- **Database**: Implement read replicas for query execution
- **Cache**: Use Redis cluster for distributed caching
- **Storage**: Use shared storage for logs and data

### Vertical Scaling

- **Memory**: Increase for larger cache sizes
- **CPU**: Increase for concurrent request handling
- **Storage**: Increase for log retention and data storage

### Auto-scaling Triggers

```yaml
# Custom metrics for scaling
- type: External
  external:
    metric:
      name: sql_synth_queue_depth
    target:
      type: AverageValue
      averageValue: "10"
```

## 🔄 Backup & Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U user dbname | gzip > backup_$(date +%Y%m%d).sql.gz

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DB_NAME="sql_synth"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump $DB_NAME | gzip > "$BACKUP_DIR/db_backup_$DATE.sql.gz"
```

### Application State Backup

```bash
# Backup monitoring state
docker cp sql-synth-agent:/app/data/monitoring.json ./backup/

# Backup cache state (if persisted)
docker exec redis redis-cli BGSAVE
```

### Disaster Recovery

1. **Database restoration:**
   ```bash
   gunzip -c backup.sql.gz | psql -h localhost -U user dbname
   ```

2. **Application redeployment:**
   ```bash
   docker-compose -f docker-compose.production.yml down
   docker-compose -f docker-compose.production.yml up -d
   ```

## 🎯 Production Checklist

### Pre-deployment

- [ ] Environment variables configured
- [ ] Secrets properly secured
- [ ] Database schema deployed
- [ ] SSL certificates installed
- [ ] Backup strategy implemented
- [ ] Monitoring configured

### Post-deployment

- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs flowing correctly
- [ ] Security scanning completed
- [ ] Performance testing done
- [ ] Documentation updated

### Ongoing Operations

- [ ] Regular security updates
- [ ] Performance monitoring
- [ ] Backup verification
- [ ] Capacity planning
- [ ] Incident response plan
- [ ] Change management process

## 📞 Support

For production support and enterprise features:

- **Documentation**: See README.md and inline code documentation
- **Monitoring**: Check Grafana dashboards and Prometheus metrics
- **Logging**: Review structured logs in JSON format
- **Health**: Monitor health check endpoints continuously

---

**Security Note**: This deployment includes enterprise-grade security features including input validation, SQL injection protection, rate limiting, and comprehensive audit logging. Regular security reviews and updates are recommended.