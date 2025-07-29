# Main Terraform configuration for SQL Synthesis Agentic Playground
# This file defines the core infrastructure components for production deployment

terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "sql-synth/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = var.terraform_lock_table
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "sql-synthesis-agentic-playground"
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "sql-synth-agentic-playground"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local values
locals {
  name_suffix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Repository  = "sql-synth-agentic-playground"
  }
  
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${local.name_suffix}-vpc"
  cidr = var.vpc_cidr
  
  azs             = local.azs
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  enable_nat_gateway   = true
  enable_vpn_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # Enable VPC Flow Logs for security monitoring
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_destination_type            = "cloud-watch-logs"
  
  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "${local.name_suffix}-cluster"
  cluster_version = var.kubernetes_version
  
  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  
  # OIDC Identity provider
  cluster_identity_providers = {
    sts = {
      client_id = "sts.amazonaws.com"
    }
  }
  
  # Cluster encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }
  
  # Managed node groups
  eks_managed_node_groups = {
    application = {
      name            = "application-nodes"
      instance_types  = var.eks_node_instance_types
      capacity_type   = "ON_DEMAND"
      
      min_size     = var.eks_node_group_min_size
      max_size     = var.eks_node_group_max_size
      desired_size = var.eks_node_group_desired_size
      
      disk_size = 50
      
      # Use the latest EKS optimized AMI
      ami_type = "AL2_x86_64"
      
      # Security groups
      vpc_security_group_ids = [aws_security_group.eks_nodes.id]
      
      labels = {
        Environment = var.environment
        NodeGroup   = "application"
      }
      
      taints = {}
      
      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled"                   = "true"
        "k8s.io/cluster-autoscaler/${local.name_suffix}-cluster" = "owned"
      })
    }
  }
  
  # Fargate profiles for serverless workloads
  fargate_profiles = {
    default = {
      name = "default-profile"
      selectors = [
        {
          namespace = "kube-system"
          labels = {
            "k8s-app" = "kube-dns"
          }
        },
        {
          namespace = "default"
          labels = {
            "compute-type" = "fargate"
          }
        }
      ]
      
      tags = local.common_tags
    }
  }
  
  tags = local.common_tags
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description         = "EKS Cluster Encryption Key"
  enable_key_rotation = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.name_suffix}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# Security Groups
resource "aws_security_group" "eks_nodes" {
  name        = "${local.name_suffix}-eks-nodes"
  description = "Security group for EKS worker nodes"
  vpc_id      = module.vpc.vpc_id
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Allow inbound traffic from ALB
  ingress {
    from_port       = 30000
    to_port         = 32767
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-eks-nodes"
  })
}

resource "aws_security_group" "alb" {
  name        = "${local.name_suffix}-alb"
  description = "Security group for Application Load Balancer"
  vpc_id      = module.vpc.vpc_id
  
  # HTTP
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # HTTPS
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-alb"
  })
}

# RDS Database
resource "aws_db_subnet_group" "main" {
  name       = "${local.name_suffix}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name        = "${local.name_suffix}-rds"
  description = "Security group for RDS database"
  vpc_id      = module.vpc.vpc_id
  
  # PostgreSQL port from EKS nodes
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-rds"
  })
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_db_instance" "main" {
  identifier = "${local.name_suffix}-db"
  
  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.db_instance_class
  
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.rds.arn
  
  db_name  = var.database_name
  username = var.database_username
  password = random_password.db_password.result
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = var.db_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  monitoring_interval             = 60
  monitoring_role_arn            = aws_iam_role.rds_enhanced_monitoring.arn
  
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-database"
  })
}

# KMS Key for RDS encryption
resource "aws_kms_key" "rds" {
  description         = "RDS Database Encryption Key"
  enable_key_rotation = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "rds" {
  name          = "alias/${local.name_suffix}-rds"
  target_key_id = aws_kms_key.rds.key_id
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${local.name_suffix}-rds-monitoring-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name_suffix}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
  
  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name        = "${local.name_suffix}-redis"
  description = "Security group for Redis cache"
  vpc_id      = module.vpc.vpc_id
  
  # Redis port from EKS nodes
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }
  
  tags = merge(local.common_tags, {
    Name = "${local.name_suffix}-redis"
  })
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${local.name_suffix}-redis"
  description               = "Redis cluster for SQL synthesis caching"
  
  node_type                 = var.redis_node_type
  port                      = 6379
  parameter_group_name      = "default.redis7"
  
  num_cache_clusters        = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = local.common_tags
}

# S3 Bucket for application data and backups
resource "aws_s3_bucket" "app_data" {
  bucket = "${local.name_suffix}-app-data"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.s3.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# KMS Key for S3 encryption
resource "aws_kms_key" "s3" {
  description         = "S3 Bucket Encryption Key"
  enable_key_rotation = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${local.name_suffix}-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "app_logs" {
  name              = "/aws/eks/${local.name_suffix}/application"
  retention_in_days = 30
  kms_key_id       = aws_kms_key.cloudwatch.arn
  
  tags = local.common_tags
}

# KMS Key for CloudWatch encryption
resource "aws_kms_key" "cloudwatch" {
  description         = "CloudWatch Logs Encryption Key"
  enable_key_rotation = true
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "logs.${var.aws_region}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_kms_alias" "cloudwatch" {
  name          = "alias/${local.name_suffix}-cloudwatch"
  target_key_id = aws_kms_key.cloudwatch.key_id
}