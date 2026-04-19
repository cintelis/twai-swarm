variable "project_name" {
  type        = string
  description = "Project short name — used as prefix for state bucket, lock table, and KMS alias. Keep short (3–20 chars, lowercase + hyphens)."
  default     = "twai-swarm"
}

variable "aws_region" {
  type        = string
  description = "AWS region for state infrastructure. Must match the region in deploy/terraform/."
  default     = "ap-southeast-2"
}
