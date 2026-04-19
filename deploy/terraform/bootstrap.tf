#-----------------------------------------------------------------------------
# One-shot DB bootstrap.
#
# Runs the init.sql against RDS from within the VPC using the same image.
# We override the command to run a short Python script that executes the SQL.
#
# Triggered by a null_resource that fires after RDS is up. Re-runs only if
# you taint it or change the SQL file hash.
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

# Trigger: re-run whenever init.sql changes
resource "null_resource" "db_bootstrap" {
  triggers = {
    sql_hash = filemd5("${path.module}/../../db/init.sql")
    task_def = aws_ecs_task_definition.db_bootstrap.arn
  }

  # Depends on RDS being healthy + the image being pushed.
  # If the image doesn't exist yet in ECR, this will fail -- that's expected
  # on the very first terraform apply. Re-run apply after the first push.
  depends_on = [
    aws_db_instance.pg,
    aws_ecs_cluster.main,
  ]

  provisioner "local-exec" {
    # Force bash on Windows (default would be cmd.exe, which can't run this script).
    # On Linux/macOS bash is already the natural choice. Requires Git Bash on
    # PATH on Windows — if `bash --version` works in PowerShell, you're set.
    interpreter = ["bash", "-c"]
    command     = <<-EOT
      set -e
      TASK_ARN=$(aws ecs run-task \
        --region ${var.aws_region} \
        --cluster ${aws_ecs_cluster.main.name} \
        --task-definition ${aws_ecs_task_definition.db_bootstrap.arn} \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[${join(",", data.aws_subnets.target.ids)}],securityGroups=[${aws_security_group.tasks.id}],assignPublicIp=ENABLED}" \
        --query 'tasks[0].taskArn' --output text)

      echo "Bootstrap task: $TASK_ARN"

      aws ecs wait tasks-stopped \
        --region ${var.aws_region} \
        --cluster ${aws_ecs_cluster.main.name} \
        --tasks "$TASK_ARN"

      EXIT_CODE=$(aws ecs describe-tasks \
        --region ${var.aws_region} \
        --cluster ${aws_ecs_cluster.main.name} \
        --tasks "$TASK_ARN" \
        --query 'tasks[0].containers[0].exitCode' --output text)

      if [ "$EXIT_CODE" != "0" ]; then
        echo "Bootstrap failed with exit code $EXIT_CODE"
        echo "Check logs in CloudWatch: /ecs/${local.name_prefix} (bootstrap stream)"
        exit 1
      fi

      echo "✅ DB bootstrap complete"
    EOT
  }
}
