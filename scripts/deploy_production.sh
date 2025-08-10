#!/bin/bash
# Production Deployment Script for SQL Synthesis Agentic Playground
# This script handles the complete deployment process including:
# - Building and pushing Docker images
# - Deploying to Kubernetes
# - Running health checks and validation
# - Rolling back on failure

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-production}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
VCS_REF="$(git rev-parse HEAD)"

# Docker configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
DOCKER_IMAGE="${DOCKER_IMAGE:-sql-synth-agentic-playground}"
FULL_IMAGE_TAG="${DOCKER_REGISTRY}/${DOCKER_IMAGE}:${VERSION}"
LATEST_TAG="${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest"

# Kubernetes configuration
NAMESPACE="${NAMESPACE:-sql-synth-prod}"
DEPLOYMENT_NAME="sql-synth-app"
KUBE_CONFIG="${KUBE_CONFIG:-}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
handle_error() {
    log_error "Deployment failed on line $1. Rolling back..."
    rollback_deployment
    exit 1
}

trap 'handle_error $LINENO' ERR

# Pre-deployment validation
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check required tools
    command -v docker >/dev/null 2>&1 || { log_error "Docker not found"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl not found"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm not found"; exit 1; }
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if [ -n "$KUBE_CONFIG" ]; then
        export KUBECONFIG="$KUBE_CONFIG"
    fi
    
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check environment variables
    if [ -z "${DATABASE_URL:-}" ]; then
        log_warning "DATABASE_URL not set - using default configuration"
    fi
    
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        log_error "OPENAI_API_KEY must be set for production deployment"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    docker build \
        --file Dockerfile.production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        --tag "$FULL_IMAGE_TAG" \
        --tag "$LATEST_TAG" \
        .
    
    log_success "Docker image built successfully"
    
    # Push to registry
    log_info "Pushing Docker image to registry..."
    docker push "$FULL_IMAGE_TAG"
    docker push "$LATEST_TAG"
    
    log_success "Docker image pushed successfully"
}

# Create Kubernetes namespace if it doesn't exist
create_namespace() {
    log_info "Creating Kubernetes namespace if needed..."
    
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        kubectl create namespace "$NAMESPACE"
        
        # Add labels
        kubectl label namespace "$NAMESPACE" \
            app=sql-synth-agentic-playground \
            environment="$ENVIRONMENT" \
            version="$VERSION"
        
        log_success "Namespace $NAMESPACE created"
    else
        log_info "Namespace $NAMESPACE already exists"
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Create secret manifest
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: sql-synth-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  database-url: $(echo -n "${DATABASE_URL:-sqlite:///:memory:}" | base64 -w 0)
  openai-api-key: $(echo -n "${OPENAI_API_KEY}" | base64 -w 0)
  jwt-secret: $(echo -n "${JWT_SECRET:-$(openssl rand -hex 32)}" | base64 -w 0)
  encryption-key: $(echo -n "${ENCRYPTION_KEY:-$(openssl rand -hex 32)}" | base64 -w 0)
  redis-password: $(echo -n "${REDIS_PASSWORD:-$(openssl rand -hex 16)}" | base64 -w 0)
EOF
    
    log_success "Secrets deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application to Kubernetes..."
    
    # Update image tag in deployment manifest
    sed -e "s|sql-synth-agentic-playground:v1.0.0|${FULL_IMAGE_TAG}|g" \
        "$PROJECT_ROOT/deployment/kubernetes/production-deployment.yaml" | \
        kubectl apply -f -
    
    log_success "Application deployed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for rollout to complete
    kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app=sql-synth-agentic-playground \
        -n "$NAMESPACE" \
        --timeout=300s
    
    log_success "Deployment is ready"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service sql-synth-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [ -z "$SERVICE_IP" ]; then
        # Fallback to port-forward for testing
        log_warning "LoadBalancer IP not available, using port-forward for health check"
        kubectl port-forward svc/sql-synth-service 8501:80 -n "$NAMESPACE" &
        PORT_FORWARD_PID=$!
        sleep 10
        HEALTH_URL="http://localhost:8501/health"
    else
        HEALTH_URL="http://$SERVICE_IP/health"
    fi
    
    # Perform health check
    for i in {1..30}; do
        if curl -f -s "$HEALTH_URL" >/dev/null 2>&1; then
            log_success "Health check passed"
            if [ -n "${PORT_FORWARD_PID:-}" ]; then
                kill $PORT_FORWARD_PID
            fi
            return 0
        fi
        log_info "Health check attempt $i/30 failed, retrying in 10s..."
        sleep 10
    done
    
    if [ -n "${PORT_FORWARD_PID:-}" ]; then
        kill $PORT_FORWARD_PID
    fi
    
    log_error "Health check failed after 30 attempts"
    return 1
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    kubectl rollout undo deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE"
    kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Port forward to test service
    kubectl port-forward svc/sql-synth-service 8501:80 -n "$NAMESPACE" &
    PORT_FORWARD_PID=$!
    sleep 10
    
    # Test basic functionality
    python3 << 'EOF'
import requests
import json
import time

def test_basic_functionality():
    base_url = "http://localhost:8501"
    
    # Test health endpoint
    health_response = requests.get(f"{base_url}/health", timeout=10)
    assert health_response.status_code == 200, f"Health check failed: {health_response.status_code}"
    
    # Test main page
    main_response = requests.get(base_url, timeout=10)
    assert main_response.status_code == 200, f"Main page failed: {main_response.status_code}"
    
    print("✅ Smoke tests passed")

if __name__ == "__main__":
    test_basic_functionality()
EOF
    
    # Clean up port forward
    kill $PORT_FORWARD_PID
    
    log_success "Smoke tests completed"
}

# Post-deployment tasks
post_deployment() {
    log_info "Performing post-deployment tasks..."
    
    # Update deployment metadata
    kubectl annotate deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        deployment.kubernetes.io/revision="$(date +%s)" \
        build.version="$VERSION" \
        build.date="$BUILD_DATE" \
        --overwrite
    
    # Create deployment record
    cat > "$PROJECT_ROOT/deployment_record.json" << EOF
{
  "timestamp": "$BUILD_DATE",
  "version": "$VERSION",
  "vcs_ref": "$VCS_REF",
  "environment": "$ENVIRONMENT",
  "image": "$FULL_IMAGE_TAG",
  "namespace": "$NAMESPACE",
  "status": "success"
}
EOF
    
    log_success "Post-deployment tasks completed"
}

# Main deployment function
main() {
    log_info "Starting production deployment..."
    log_info "Version: $VERSION"
    log_info "Environment: $ENVIRONMENT"
    log_info "Image: $FULL_IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    
    validate_prerequisites
    build_and_push_image
    create_namespace
    deploy_secrets
    deploy_application
    wait_for_deployment
    health_check
    run_smoke_tests
    post_deployment
    
    log_success "🎉 Production deployment completed successfully!"
    log_info "Application is available at: http://$(kubectl get service sql-synth-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi