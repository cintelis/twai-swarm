output "state_bucket_name" {
  value       = aws_s3_bucket.state.id
  description = "S3 bucket holding Terraform remote state."
}

output "state_bucket_arn" {
  value       = aws_s3_bucket.state.arn
  description = "ARN — wire into the GitHub deploy role's IAM policy."
}

output "lock_table_name" {
  value       = aws_dynamodb_table.lock.name
  description = "DynamoDB state-lock table."
}

output "lock_table_arn" {
  value       = aws_dynamodb_table.lock.arn
  description = "ARN — wire into the GitHub deploy role's IAM policy."
}

output "kms_key_arn" {
  value       = aws_kms_key.state.arn
  description = "KMS key for state encryption."
}

output "kms_key_id" {
  value       = aws_kms_key.state.id
  description = "Key ID — for backend.conf."
}

# Ready-to-paste config for deploy/terraform/backend.conf.
# Sprint 2026-04: switched from DynamoDB-table locking to S3-native lockfile
# (Terraform 1.11+). The DynamoDB `lock` table below is left in state as a
# fallback — drop it once `use_lockfile` has run cleanly for a few weeks.
output "backend_config_hcl" {
  value       = <<-EOT
    bucket       = "${aws_s3_bucket.state.id}"
    key          = "deploy/terraform.tfstate"
    region       = "${var.aws_region}"
    use_lockfile = true
    encrypt      = true
    kms_key_id   = "${aws_kms_key.state.arn}"
  EOT
  description = "Copy this into deploy/terraform/backend.conf (gitignored), then run `terraform init -reconfigure` in deploy/terraform/."
}
