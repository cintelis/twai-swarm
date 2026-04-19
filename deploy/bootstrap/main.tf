# ─── State-backend bootstrap ───────────────────────────────────────────
# One-time apply. Creates the S3 bucket + DynamoDB lock table + KMS key
# that deploy/terraform/ uses for its remote state backend.
#
# This module intentionally uses LOCAL state — there is no chicken-and-egg
# way to create the state bucket with its own backend pointing at itself.
# The local state file lives next to this main.tf and is gitignored.
#
# Apply: cd deploy/bootstrap && terraform init && terraform apply
# Output: copy `terraform output backend_config_hcl` into
#         ../terraform/backend.conf, then `terraform init -reconfigure` there.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  account_id   = data.aws_caller_identity.current.account_id
  bucket_name  = "${var.project_name}-tfstate-${local.account_id}"
  table_name   = "${var.project_name}-tflock"
  kms_alias    = "alias/${var.project_name}-tfstate"
  common_tags = {
    Project   = var.project_name
    ManagedBy = "terraform-bootstrap"
    Purpose   = "remote-state"
  }
}

# ─── KMS key for bucket encryption ─────────────────────────────────────
# Customer-managed so the key policy is auditable; bucket rotates data
# keys automatically (S3 handles per-object DEK).

resource "aws_kms_key" "state" {
  description             = "Encrypts Terraform remote state for ${var.project_name}"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = local.common_tags
}

resource "aws_kms_alias" "state" {
  name          = local.kms_alias
  target_key_id = aws_kms_key.state.key_id
}

# ─── State bucket ──────────────────────────────────────────────────────

resource "aws_s3_bucket" "state" {
  bucket = local.bucket_name
  tags   = local.common_tags

  # Guardrail: refuse `terraform destroy` on the bucket unless emptied manually.
  # Prevents a stray destroy of this module from torching all future state.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "state" {
  bucket = aws_s3_bucket.state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.state.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "state" {
  bucket = aws_s3_bucket.state.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# ─── Lock table ────────────────────────────────────────────────────────
# Single hash key `LockID`; Terraform uses one row per state file while
# a plan/apply is in flight. PAY_PER_REQUEST so you don't pay for idle.

resource "aws_dynamodb_table" "lock" {
  name         = local.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"
  tags         = local.common_tags

  attribute {
    name = "LockID"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.state.arn
  }
}
