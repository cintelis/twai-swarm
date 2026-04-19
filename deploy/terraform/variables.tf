variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-2" # Sydney -- closest to Cintelis customers
}

variable "project_name" {
  description = "Used as prefix for resource names"
  type        = string
  default     = "lean-agent"
}

variable "image_tag" {
  description = "ECR image tag to deploy"
  type        = string
  default     = "latest"
}

variable "temporal_host" {
  description = "Temporal Cloud endpoint, e.g. your-namespace.xxxxx.tmprl.cloud:7233"
  type        = string
}

variable "temporal_namespace" {
  description = "Temporal Cloud namespace"
  type        = string
}

# Secrets -- provide via TF_VAR_* env vars or a .tfvars file NOT committed.
variable "anthropic_api_key" {
  type      = string
  sensitive = true
}

variable "xai_api_key" {
  description = "xAI API key (for Grok models)"
  type        = string
  sensitive   = true
}

variable "temporal_api_key" {
  description = "Temporal Cloud API key (replaces mTLS cert/key auth)"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "vpc_id" {
  description = "VPC to deploy into. Leave empty to use the account's default VPC."
  type        = string
  default     = ""
}
