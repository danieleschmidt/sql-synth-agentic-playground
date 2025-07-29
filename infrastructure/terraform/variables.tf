# Terraform variables for SQL Synthesis Agentic Playground infrastructure

# General Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "sql-synthesis-agentic-playground"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

# Terraform State Configuration
variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
}

variable "terraform_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  type        = string
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_node_instance_types" {
  description = "Instance types for EKS worker nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "eks_node_group_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 1
}

variable "eks_node_group_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 2
}

variable "eks_node_group_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 10
}

# Database Configuration
variable "postgres_version" {
  description = "PostgreSQL version for RDS"
  type        = string
  default     = "15.4"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage for RDS (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for RDS autoscaling (GB)"
  type        = number
  default     = 100
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "sql_synth"
}

variable "database_username" {
  description = "Database master username"
  type        = string
  default     = "postgres"
}

variable "db_backup_retention_period" {
  description = "Database backup retention period in days"
  type        = number
  default     = 7
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

# Application Configuration
variable "app_image_tag" {
  description = "Docker image tag for the application"
  type        = string
  default     = "latest"
}

variable "app_replica_count" {
  description = "Number of application replicas"
  type        = number
  default     = 2
}

variable "app_cpu_request" {
  description = "CPU request for application pods"
  type        = string
  default     = "100m"
}

variable "app_cpu_limit" {
  description = "CPU limit for application pods"
  type        = string
  default     = "500m"
}

variable "app_memory_request" {
  description = "Memory request for application pods"
  type        = string
  default     = "128Mi"
}

variable "app_memory_limit" {
  description = "Memory limit for application pods"
  type        = string
  default     = "512Mi"
}

# Monitoring Configuration
variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "prometheus_storage_size" {
  description = "Storage size for Prometheus"
  type        = string
  default     = "50Gi"
}

variable "grafana_storage_size" {
  description = "Storage size for Grafana"
  type        = string
  default     = "10Gi"
}

# Security Configuration
variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Network Policy"
  type        = bool
  default     = true
}

variable "enable_service_mesh" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = false
}

# SSL/TLS Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

# Auto-scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "hpa_min_replicas" {
  description = "Minimum replicas for HPA"
  type        = number
  default     = 2
}

variable "hpa_max_replicas" {
  description = "Maximum replicas for HPA"
  type        = number
  default     = 20
}

variable "hpa_target_cpu_utilization" {
  description = "Target CPU utilization for HPA"
  type        = number
  default     = 70
}

# Backup Configuration
variable "enable_database_backup" {
  description = "Enable automated database backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# Environment-specific configurations
variable "environment_configs" {
  description = "Environment-specific configurations"
  type = map(object({
    db_instance_class       = string
    eks_node_instance_types = list(string)
    min_nodes              = number
    max_nodes              = number
    desired_nodes          = number
    app_replicas           = number
    enable_deletion_protection = bool
  }))
  
  default = {
    dev = {
      db_instance_class       = "db.t3.micro"
      eks_node_instance_types = ["t3.small"]
      min_nodes              = 1
      max_nodes              = 3
      desired_nodes          = 1
      app_replicas           = 1
      enable_deletion_protection = false
    }
    
    staging = {
      db_instance_class       = "db.t3.small"
      eks_node_instance_types = ["t3.medium"]
      min_nodes              = 1
      max_nodes              = 5
      desired_nodes          = 2
      app_replicas           = 2
      enable_deletion_protection = false
    }
    
    production = {
      db_instance_class       = "db.t3.medium"
      eks_node_instance_types = ["t3.large", "t3.xlarge"]
      min_nodes              = 2
      max_nodes              = 20
      desired_nodes          = 3
      app_replicas           = 3
      enable_deletion_protection = true
    }
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for optional components"
  type = object({
    enable_monitoring    = bool
    enable_logging      = bool
    enable_tracing      = bool
    enable_service_mesh = bool
    enable_backup       = bool
    enable_autoscaling  = bool
  })
  
  default = {
    enable_monitoring    = true
    enable_logging      = true
    enable_tracing      = false
    enable_service_mesh = false
    enable_backup       = true
    enable_autoscaling  = true
  }
}

# Cost Optimization
variable "cost_optimization" {
  description = "Cost optimization settings"
  type = object({
    use_spot_instances      = bool
    spot_instance_percentage = number
    enable_hibernation      = bool
    enable_scheduled_scaling = bool
  })
  
  default = {
    use_spot_instances      = false
    spot_instance_percentage = 50
    enable_hibernation      = false
    enable_scheduled_scaling = false
  }
}

# Compliance Settings
variable "compliance_requirements" {
  description = "Compliance requirements"
  type = object({
    encrypt_at_rest     = bool
    encrypt_in_transit  = bool
    enable_audit_logs   = bool
    enable_vpc_flow_logs = bool
    require_mfa         = bool
  })
  
  default = {
    encrypt_at_rest     = true
    encrypt_in_transit  = true
    enable_audit_logs   = true
    enable_vpc_flow_logs = true
    require_mfa         = false
  }
}

# Disaster Recovery
variable "disaster_recovery" {
  description = "Disaster recovery configuration"
  type = object({
    enable_cross_region_backup = bool
    backup_region             = string
    rto_minutes               = number
    rpo_minutes               = number
  })
  
  default = {
    enable_cross_region_backup = false
    backup_region             = "us-east-1"
    rto_minutes               = 60
    rpo_minutes               = 15
  }
}

# Notification Configuration
variable "notification_settings" {
  description = "Notification settings for alerts"
  type = object({
    email_addresses = list(string)
    slack_webhook   = string
    sns_topic_arn   = string
  })
  
  default = {
    email_addresses = []
    slack_webhook   = ""
    sns_topic_arn   = ""
  }
}