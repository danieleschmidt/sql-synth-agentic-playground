#!/bin/bash
# Monitoring and Observability Setup Script
# Sets up comprehensive monitoring stack for SQL Synth application

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MONITORING_NAMESPACE="monitoring"
APP_NAMESPACE="sql-synth"
KUBERNETES_DEPLOYMENT=${KUBERNETES_DEPLOYMENT:-false}
DOCKER_DEPLOYMENT=${DOCKER_DEPLOYMENT:-true}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    if [[ "$DOCKER_DEPLOYMENT" == "true" ]]; then
        if ! command -v docker >/dev/null 2>&1; then
            missing_deps+=("docker")
        fi
        
        if ! command -v docker-compose >/dev/null 2>&1; then
            missing_deps+=("docker-compose")
        fi
    fi
    
    if [[ "$KUBERNETES_DEPLOYMENT" == "true" ]]; then
        if ! command -v kubectl >/dev/null 2>&1; then
            missing_deps+=("kubectl")
        fi
        
        if ! command -v helm >/dev/null 2>&1; then
            missing_deps+=("helm")
        fi
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Function to setup Docker monitoring stack
setup_docker_monitoring() {
    print_status "Setting up Docker monitoring stack..."
    
    # Create monitoring directory structure
    mkdir -p config/grafana/{dashboards,provisioning/{dashboards,datasources}}
    mkdir -p config/prometheus/rules
    mkdir -p data/{prometheus,grafana,jaeger}
    
    # Create Grafana provisioning configuration
    cat <<EOF > config/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      queryTimeout: "60s"
      httpMethod: "POST"
      exemplarTraceIdDestinations:
        - name: traceID
          datasourceUid: jaeger
          
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    uid: jaeger
    editable: true
    jsonData:
      tracesToLogs:
        datasourceUid: loki
        tags: ['job', 'instance', 'pod', 'namespace']
        mappedTags: [{ key: 'service.name', value: 'service' }]
        mapTagNamesEnabled: false
        spanStartTimeShift: '1h'
        spanEndTimeShift: '1h'
        filterByTraceID: false
        filterBySpanID: false
        
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    uid: loki
    editable: true
    jsonData:
      maxLines: 1000
      derivedFields:
        - datasourceUid: jaeger
          matcherRegex: "traceID=(\\w+)"
          name: TraceID
          url: '$${__value.raw}'
EOF
    
    # Create dashboard provisioning configuration
    cat <<EOF > config/grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1

providers:
  - name: 'SQL Synth Dashboards'
    orgId: 1
    folder: 'SQL Synth'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    # Create enhanced docker-compose file for monitoring
    cat <<EOF > docker-compose.monitoring.yml
version: '3.8'

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: sql-synth-otel-collector
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./config/otel-collector.yml:/etc/otel-collector-config.yml:ro
      - ./logs:/app/logs:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics
      - "13133:13133" # Health check
      - "1777:1777"   # pprof
      - "55679:55679" # zpages
    depends_on:
      - jaeger
      - prometheus
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Jaeger for distributed tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: sql-synth-jaeger
    ports:
      - "16686:16686" # Jaeger UI
      - "14250:14250" # gRPC collector
      - "14268:14268" # HTTP collector
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    volumes:
      - jaeger_data:/badger
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Loki for log aggregation
  loki:
    image: grafana/loki:latest
    container_name: sql-synth-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - loki_data:/loki
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Vector for log collection
  vector:
    image: timberio/vector:latest-alpine
    container_name: sql-synth-vector
    volumes:
      - ./config/vector.toml:/etc/vector/vector.toml:ro
      - ./logs:/app/logs:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - loki
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # AlertManager for alert handling
  alertmanager:
    image: prom/alertmanager:latest
    container_name: sql-synth-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./config/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--cluster.advertise-address=0.0.0.0:9093'
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # cAdvisor for container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: sql-synth-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Node Exporter for system metrics
  node-exporter:
    image: prom/node-exporter:latest
    container_name: sql-synth-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Postgres Exporter for database metrics
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: sql-synth-postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://sql_synth_user:sql_synth_password@postgres:5432/sql_synth_db?sslmode=disable
    depends_on:
      - postgres
    networks:
      - sql-synth-network
    restart: unless-stopped
    
  # Redis Exporter for cache metrics
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: sql-synth-redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    depends_on:
      - redis
    networks:
      - sql-synth-network
    restart: unless-stopped

volumes:
  jaeger_data:
    driver: local
  loki_data:
    driver: local
  alertmanager_data:
    driver: local
    
networks:
  sql-synth-network:
    external: true
EOF
    
    print_success "Docker monitoring configuration created"
}

# Function to create alerting configuration
create_alerting_config() {
    print_status "Creating alerting configuration..."
    
    cat <<EOF > config/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@sql-synth.local'
  smtp_auth_username: 'alertmanager@sql-synth.local'
  smtp_auth_password: 'password'
  
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 5s
    repeat_interval: 30m
  - match:
      component: security
    receiver: 'security-alerts'
    group_wait: 0s
    repeat_interval: 15m

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true
    
- name: 'critical-alerts'
  email_configs:
  - to: 'admin@sql-synth.local'
    subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Instance: {{ .Labels.instance }}
      Severity: {{ .Labels.severity }}
      {{ end }}
  slack_configs:
  - api_url: '${SLACK_WEBHOOK_URL}'
    channel: '#alerts'
    title: 'Critical Alert: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    
- name: 'security-alerts'
  email_configs:
  - to: 'security@sql-synth.local'
    subject: 'SECURITY ALERT: {{ .GroupLabels.alertname }}'
    body: |
      SECURITY INCIDENT DETECTED
      
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Instance: {{ .Labels.instance }}
      Time: {{ .StartsAt }}
      {{ end }}
      
      Immediate investigation required.
  pagerduty_configs:
  - routing_key: '${PAGERDUTY_ROUTING_KEY}'
    description: 'Security Alert: {{ .GroupLabels.alertname }}'
    
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
EOF
    
    print_success "Alerting configuration created"
}

# Function to create log collection configuration
create_log_config() {
    print_status "Creating log collection configuration..."
    
    cat <<EOF > config/vector.toml
# Vector configuration for log collection
[api]
enabled = true
address = "0.0.0.0:8686"

# Data directory
data_dir = "/vector-data-dir"

# Sources
[sources.app_logs]
type = "file"
includes = ["/app/logs/*.log"]
read_from = "beginning"
remove_after_secs = 86400

[sources.docker_logs]
type = "docker_logs"
include_containers = ["sql-synth-app", "sql-synth-postgres", "sql-synth-redis"]
exclude_containers = ["vector", "grafana", "prometheus"]
auto_partial_merge = true

# Transforms
[transforms.app_logs_parsed]
type = "remap"
inputs = ["app_logs"]
source = '''
. = parse_json!(.message)
.timestamp = parse_timestamp!(.timestamp, format: "%Y-%m-%dT%H:%M:%S%.fZ")
.level = downcase(.level)
.service = "sql-synth-app"
'''

[transforms.docker_logs_enriched]
type = "remap"
inputs = ["docker_logs"]
source = '''
.service = .container_name
.environment = "production"
.log_type = "container"
if .container_name == "sql-synth-postgres" {
  .component = "database"
} else if .container_name == "sql-synth-redis" {
  .component = "cache"
} else {
  .component = "application"
}
'''

# Sinks
[sinks.loki]
type = "loki"
inputs = ["app_logs_parsed", "docker_logs_enriched"]
endpoint = "http://loki:3100"
encoding.codec = "json"
labels.service = "{{ service }}"
labels.level = "{{ level }}"
labels.component = "{{ component }}"

[sinks.console]
type = "console"
inputs = ["app_logs_parsed"]
encoding.codec = "json"

# Health checks
[sources.internal_metrics]
type = "internal_metrics"

[sinks.prometheus_metrics]
type = "prometheus_exporter"
inputs = ["internal_metrics"]
address = "0.0.0.0:9598"
EOF
    
    print_success "Log collection configuration created"
}

# Function to setup Kubernetes monitoring
setup_kubernetes_monitoring() {
    print_status "Setting up Kubernetes monitoring..."
    
    # Create namespace
    kubectl create namespace $MONITORING_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace $MONITORING_NAMESPACE \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword=admin123 \
        --wait
    
    # Install Jaeger
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace $MONITORING_NAMESPACE \
        --set provisionDataStore.cassandra=false \
        --set allInOne.enabled=true \
        --set storage.type=memory \
        --set agent.enabled=false \
        --set collector.enabled=false \
        --set query.enabled=false \
        --wait
    
    # Install Loki
    helm upgrade --install loki grafana/loki \
        --namespace $MONITORING_NAMESPACE \
        --set persistence.enabled=true \
        --set persistence.size=10Gi \
        --wait
    
    print_success "Kubernetes monitoring stack installed"
}

# Function to create monitoring scripts
create_monitoring_scripts() {
    print_status "Creating monitoring utility scripts..."
    
    # Create monitoring health check script
    cat <<'EOF' > scripts/check-monitoring-health.sh
#!/bin/bash
# Monitoring Stack Health Check

echo "🔍 Monitoring Stack Health Check"
echo "================================"

# Check Prometheus
echo "\n📊 Prometheus:"
if curl -s http://localhost:9090/-/healthy >/dev/null; then
    echo "✅ Prometheus is healthy"
    targets=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length')
    echo "📊 Active targets: $targets"
else
    echo "❌ Prometheus is unhealthy"
fi

# Check Grafana
echo "\n📈 Grafana:"
if curl -s http://localhost:3000/api/health >/dev/null; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is unhealthy"
fi

# Check Jaeger
echo "\n🔍 Jaeger:"
if curl -s http://localhost:16686/api/services >/dev/null; then
    echo "✅ Jaeger is healthy"
    services=$(curl -s http://localhost:16686/api/services | jq -r '.data | length')
    echo "🔍 Traced services: $services"
else
    echo "❌ Jaeger is unhealthy"
fi

# Check AlertManager
echo "\n🚨 AlertManager:"
if curl -s http://localhost:9093/-/healthy >/dev/null; then
    echo "✅ AlertManager is healthy"
    alerts=$(curl -s http://localhost:9093/api/v1/alerts | jq -r '.data | length')
    echo "🚨 Active alerts: $alerts"
else
    echo "❌ AlertManager is unhealthy"
fi

# Check OpenTelemetry Collector
echo "\n📡 OpenTelemetry Collector:"
if curl -s http://localhost:13133 >/dev/null; then
    echo "✅ OTel Collector is healthy"
else
    echo "❌ OTel Collector is unhealthy"
fi

echo "\n🏁 Health check completed"
EOF
    
    chmod +x scripts/check-monitoring-health.sh
    
    # Create monitoring startup script
    cat <<'EOF' > scripts/start-monitoring.sh
#!/bin/bash
# Start Monitoring Stack

echo "🚀 Starting SQL Synth Monitoring Stack"
echo "====================================="

# Start main application first
echo "📊 Starting application..."
docker-compose up -d

# Wait for application to be ready
echo "⏳ Waiting for application to be ready..."
sleep 30

# Start monitoring stack
echo "📈 Starting monitoring services..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for monitoring services
echo "⏳ Waiting for monitoring services to be ready..."
sleep 60

# Run health check
echo "🔍 Running health check..."
./scripts/check-monitoring-health.sh

echo "\n🎉 Monitoring stack started successfully!"
echo "\n🔗 Access URLs:"
echo "  📊 Prometheus: http://localhost:9090"
echo "  📈 Grafana: http://localhost:3000 (admin/admin)"
echo "  🔍 Jaeger: http://localhost:16686"
echo "  🚨 AlertManager: http://localhost:9093"
echo "  📡 OTel Collector: http://localhost:55679"
echo "  🖥️  cAdvisor: http://localhost:8080"
echo "  💾 Logs (Loki): http://localhost:3100"
EOF
    
    chmod +x scripts/start-monitoring.sh
    
    print_success "Monitoring utility scripts created"
}

# Function to run monitoring setup
main() {
    print_status "Starting monitoring and observability setup..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --kubernetes)
                KUBERNETES_DEPLOYMENT=true
                DOCKER_DEPLOYMENT=false
                shift
                ;;
            --docker)
                DOCKER_DEPLOYMENT=true
                KUBERNETES_DEPLOYMENT=false
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --kubernetes    Setup for Kubernetes deployment"
                echo "  --docker        Setup for Docker deployment (default)"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    
    if [[ "$DOCKER_DEPLOYMENT" == "true" ]]; then
        setup_docker_monitoring
        create_alerting_config
        create_log_config
        create_monitoring_scripts
    fi
    
    if [[ "$KUBERNETES_DEPLOYMENT" == "true" ]]; then
        setup_kubernetes_monitoring
    fi
    
    print_success "Monitoring and observability setup completed!"
    
    if [[ "$DOCKER_DEPLOYMENT" == "true" ]]; then
        print_status "To start monitoring stack: ./scripts/start-monitoring.sh"
        print_status "To check health: ./scripts/check-monitoring-health.sh"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi