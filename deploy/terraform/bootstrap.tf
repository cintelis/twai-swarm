#-----------------------------------------------------------------------------
# DB bootstrap task definition.
#
# Defines the one-shot ECS task that applies the schema (app/bootstrap_db.py).
# Terraform manages the task DEFINITION only — running it is a manual step
# triggered by `scripts/bootstrap_db.ps1` (Windows) or via aws ecs run-task
# directly. We removed the previous null_resource + local-exec pattern
# because it was Windows-bash-fragile (terraform's local-exec spawns bash
# without the full Windows PATH, so `aws.exe` wasn't found).
#
# When to run the bootstrap:
#   - First-time deploy (creates schema)
#   - Schema migration in app/bootstrap_db.py SCHEMA_SQL changed
#   - New tables added (e.g. tenant_id columns, langfuse DB)
#
# How to run:
#   PowerShell:  scripts/bootstrap_db.ps1
#   Or manually:
#     $task = aws ecs run-task --region ap-southeast-2 \
#       --cluster lean-agent-cluster \
#       --task-definition lean-agent-db-bootstrap \
#       --launch-type FARGATE \
#       --network-configuration "awsvpcConfiguration={subnets=[<3 subnet ids>],securityGroups=[sg-0b9bd1a13ffa5e437],assignPublicIp=ENABLED}" \
#       --query 'tasks[0].taskArn' --output text
#     aws ecs wait tasks-stopped --region ap-southeast-2 --cluster lean-agent-cluster --tasks $task
#     aws logs tail /ecs/lean-agent --log-stream-name-prefix bootstrap --since 2m
#
# Bootstrap is idempotent: CREATE TABLE IF NOT EXISTS + ALTER TABLE IF
# NOT EXISTS — safe to re-run.
#-----------------------------------------------------------------------------

resource "aws_ecs_task_definition" "db_bootstrap" {
  family                   = "${local.name_prefix}-db-bootstrap"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "bootstrap"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true
      command   = ["python", "-m", "app.bootstrap_db"]

      secrets = [
        { name = "PG_DSN", valueFrom = aws_secretsmanager_secret.pg_dsn.arn },
        # Bootstrap also creates the `langfuse` database + `langfuse_app` user
        # if this password is present (skips silently otherwise).
        { name = "LANGFUSE_DB_PASSWORD", valueFrom = aws_secretsmanager_secret.langfuse_db_password.arn },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "bootstrap"
        }
      }
    }
  ])
}
