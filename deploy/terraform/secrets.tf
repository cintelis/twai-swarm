#-----------------------------------------------------------------------------
# ECR repo for the single shared image
#-----------------------------------------------------------------------------
resource "aws_ecr_repository" "app" {
  name                 = "${local.name_prefix}-app"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

#-----------------------------------------------------------------------------
# Secrets Manager -- one secret per sensitive value. ECS task definitions
# reference these via `secrets` (not `environment`) so they never land in
# CloudTrail or task def JSON.
#-----------------------------------------------------------------------------
resource "aws_secretsmanager_secret" "anthropic" {
  name                    = "${local.name_prefix}/anthropic-api-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "anthropic" {
  secret_id     = aws_secretsmanager_secret.anthropic.id
  secret_string = var.anthropic_api_key
}

resource "aws_secretsmanager_secret" "xai" {
  name                    = "${local.name_prefix}/xai-api-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "xai" {
  secret_id     = aws_secretsmanager_secret.xai.id
  secret_string = var.xai_api_key
}

# OpenAI fallback provider. Optional: if var.openai_api_key is empty we still
# create the secret (with a placeholder string) so ECS task defs can reference
# it unconditionally. Fallback is disabled at the app layer when the value is
# the placeholder — see app/config.py + router.FALLBACK_CHAIN.
resource "aws_secretsmanager_secret" "openai" {
  name                    = "${local.name_prefix}/openai-api-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "openai" {
  secret_id     = aws_secretsmanager_secret.openai.id
  secret_string = var.openai_api_key != "" ? var.openai_api_key : "UNSET"
}

resource "aws_secretsmanager_secret" "temporal_api_key" {
  name                    = "${local.name_prefix}/temporal-api-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "temporal_api_key" {
  secret_id     = aws_secretsmanager_secret.temporal_api_key.id
  secret_string = var.temporal_api_key
}

resource "aws_secretsmanager_secret" "pg_dsn" {
  name                    = "${local.name_prefix}/pg-dsn"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "pg_dsn" {
  secret_id = aws_secretsmanager_secret.pg_dsn.id
  secret_string = format(
    "postgresql://%s:%s@%s:%d/%s",
    aws_db_instance.pg.username,
    var.db_password,
    aws_db_instance.pg.address,
    aws_db_instance.pg.port,
    aws_db_instance.pg.db_name,
  )
}

#-----------------------------------------------------------------------------
# CloudWatch log group -- shared by API + worker
#-----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 14
}
