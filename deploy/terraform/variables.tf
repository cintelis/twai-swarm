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

variable "openai_api_key" {
  description = "OpenAI API key. Powers the gpt-5.4 fallback provider (app/providers/openai_provider.py). Leave empty to disable fallback."
  type        = string
  sensitive   = true
  default     = ""
}

# ─── GitHub App credentials ───────────────────────────────────────────────
# Register the App at github.com/settings/apps/new (or org-level), then paste:
#   - Numeric App ID (shown on the app's settings page)
#   - The full .pem private key contents
#   - The public install URL (https://github.com/apps/<slug>/installations/new)
# Leave all three empty to disable the GitHub push feature; the API endpoints
# return 503 in that case.

variable "github_app_id" {
  description = "GitHub App ID (numeric). Empty disables the GitHub push feature."
  type        = string
  default     = ""
}

variable "github_app_private_key" {
  description = "GitHub App private key (.pem contents). Sensitive."
  type        = string
  sensitive   = true
  default     = ""
}

variable "github_app_install_url" {
  description = "Public install URL for the App. UI Connect button opens this."
  type        = string
  default     = ""
}

# ─── Langfuse self-hosted ─────────────────────────────────────────────────
# Langfuse is deployed as a separate ECS Express service against a new
# database in the existing RDS. Two-step bootstrap:
#   1. terraform apply → creates service with placeholder keys
#   2. Sign up + create a project on the Langfuse UI → copy public/secret keys
#   3. Set them in tfvars below → terraform apply again

variable "langfuse_public_url" {
  description = "Langfuse's own on.aws URL (set AFTER first apply). Used as NEXTAUTH_URL."
  type        = string
  default     = ""
}

variable "langfuse_public_key" {
  description = "Langfuse project public key (starts pk-lf-). Leave empty on first apply; populate after UI signup."
  type        = string
  default     = ""
}

variable "langfuse_secret_key" {
  description = "Langfuse project secret key (starts sk-lf-). Leave empty on first apply; populate after UI signup."
  type        = string
  sensitive   = true
  default     = ""
}

variable "langfuse_disable_signup" {
  description = "Flip to true after creating the first admin user so strangers can't register."
  type        = bool
  default     = false
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

# ─── HA / scaling knobs ────────────────────────────────────────────────────
# Defaults match what's currently deployed in dev. Flip a tfvar + apply when
# you're ready to harden for prod — every change here is a tfvars edit, no
# Terraform code change needed.

variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro for dev (~$15/mo), db.r6g.large+ for prod."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "RDS storage in GB. gp3 — easy to grow."
  type        = number
  default     = 20
}

variable "db_multi_az" {
  description = "Multi-AZ failover for RDS. Roughly doubles DB cost; required for prod."
  type        = bool
  default     = false
}

variable "db_backup_retention_days" {
  description = "RDS automated-backup retention. 1 day for dev, 7-35 for prod."
  type        = number
  default     = 1
}

variable "db_deletion_protection" {
  description = "Block accidental terraform destroy of RDS. Always true for prod."
  type        = bool
  default     = false
}

variable "api_min_tasks" {
  description = "Minimum API task count for the Express service autoscaler."
  type        = number
  default     = 1
}

variable "api_max_tasks" {
  description = "Maximum API task count for the Express service autoscaler."
  type        = number
  default     = 5
}

variable "api_cpu_target" {
  description = "Target average CPU utilisation that triggers API autoscaling (percent)."
  type        = number
  default     = 70
}

variable "api_cpu" {
  description = "API task CPU units. 256/512/1024/2048/4096."
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "API task memory MB. Must be a valid Fargate CPU/memory pair."
  type        = number
  default     = 1024
}

variable "worker_cpu" {
  description = "Worker task CPU units."
  type        = number
  default     = 1024
}

variable "worker_memory" {
  description = "Worker task memory MB."
  type        = number
  default     = 2048
}

variable "worker_desired_count" {
  description = "Number of worker tasks. Scale by adding queue-narrowed worker services for parallelism, not by raising this."
  type        = number
  default     = 1
}
