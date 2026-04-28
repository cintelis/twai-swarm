#-----------------------------------------------------------------------------
# Cluster -- shared between the Express Mode API service and the regular
# Fargate worker service.
#-----------------------------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enhanced"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

#-----------------------------------------------------------------------------
# Security groups for tasks.
# - api_tasks: accepts traffic from the ALB (which Express Mode manages)
# - worker_tasks: no inbound needed (outbound-only to Temporal + Anthropic + RDS)
# Both named `tasks` in the SG resource because they both need DB access
# via the same rule on the RDS SG.
#-----------------------------------------------------------------------------
resource "aws_security_group" "tasks" {
  name   = "${local.name_prefix}-tasks-sg"
  vpc_id = data.aws_vpc.target.id

  # API ingress is added by Express Mode's managed SG; we trust the ALB here
  # by opening 8000 from the VPC CIDR. Tighten later by referencing the
  # Express-managed ALB SG once Terraform AWS provider exposes it cleanly.
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.target.cidr_block]
  }

  # Worker health check port -- only from within VPC
  ingress {
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.target.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

#-----------------------------------------------------------------------------
# Task definition: API container config is now owned by the Express module
# below (see module.api_express). Removing the separate aws_ecs_task_definition
# prevents drift and double-management.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Task definition: Worker
#-----------------------------------------------------------------------------
locals {
  # Container definitions defined once so both the task def AND the JSON
  # exported for the GitHub Actions deploy workflow stay byte-identical.
  # Reading from aws_ecs_task_definition.worker.container_definitions caused
  # "Provider produced inconsistent final plan" errors because the provider
  # normalises the JSON during apply (adds mountPoints:[], hostPort, etc.).
  worker_container_definitions = [
    {
      name      = "worker"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true
      command   = ["python", "-m", "app.worker"]

      portMappings = [
        { containerPort = 8001, protocol = "tcp" }
      ]

      environment = [
        { name = "TEMPORAL_HOST", value = var.temporal_host },
        { name = "TEMPORAL_NAMESPACE", value = var.temporal_namespace },
        { name = "TEMPORAL_TLS", value = "true" },
        { name = "TEMPORAL_QUEUES", value = "" },
        { name = "GITHUB_APP_INSTALL_URL", value = var.github_app_install_url },
        { name = "LANGFUSE_HOST", value = var.langfuse_public_url },
        { name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = var.otel_exporter_otlp_endpoint },
        { name = "DEPLOYMENT_ENV", value = "dev" },
      ]

      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic.arn },
        { name = "XAI_API_KEY", valueFrom = aws_secretsmanager_secret.xai.arn },
        { name = "OPENAI_API_KEY", valueFrom = aws_secretsmanager_secret.openai.arn },
        { name = "GITHUB_APP_ID", valueFrom = aws_secretsmanager_secret.github_app_id.arn },
        { name = "GITHUB_APP_PRIVATE_KEY", valueFrom = aws_secretsmanager_secret.github_app_private_key.arn },
        { name = "TEMPORAL_API_KEY", valueFrom = aws_secretsmanager_secret.temporal_api_key.arn },
        { name = "PG_DSN", valueFrom = aws_secretsmanager_secret.pg_dsn.arn },
        { name = "LANGFUSE_PUBLIC_KEY", valueFrom = aws_secretsmanager_secret.langfuse_public_key.arn },
        { name = "LANGFUSE_SECRET_KEY", valueFrom = aws_secretsmanager_secret.langfuse_secret_key.arn },
        { name = "LANGFUSE_DB_PASSWORD", valueFrom = aws_secretsmanager_secret.langfuse_db_password.arn },
        # Neo4j — repo-knowledge graph store. URL resolves via CloudMap;
        # password is the raw value (no "neo4j/" prefix — that's added on
        # the Neo4j server side, not here).
        { name = "NEO4J_URL", valueFrom = aws_secretsmanager_secret.neo4j_url.arn },
        { name = "NEO4J_PASSWORD", valueFrom = aws_secretsmanager_secret.neo4j_password.arn },
      ]

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 45
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "worker"
        }
      }
    }
  ]
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.worker_cpu)
  memory                   = tostring(var.worker_memory)
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode(local.worker_container_definitions)
}

#-----------------------------------------------------------------------------
# Service: Worker (regular Fargate -- no ALB needed)
#-----------------------------------------------------------------------------
resource "aws_ecs_service" "worker" {
  name            = "${local.name_prefix}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.target.ids
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = true # default VPC; for a private VPC with NAT, set false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # deploy.yml registers new task def revisions out-of-band via
  # aws ecs deploy-task-definition. Ignoring task_definition here prevents
  # `terraform apply` from rolling the service back to whatever revision
  # was current at the last apply.
  lifecycle {
    ignore_changes = [desired_count, task_definition]
  }
}

#-----------------------------------------------------------------------------
# Service: API (Express Mode)
#
# Uses the community module which wraps the aws_ecs_express_gateway_service
# resource (provider v6.23+, released alongside the Nov 2025 GA launch).
# Express Mode provisions the ALB, target group, listener rules, SSL cert
# on *.region.on.aws, and autoscaling policies automatically.
#
# Note: unlike the worker service, we pass container config directly to the
# module -- Express owns the task definition lifecycle.
#-----------------------------------------------------------------------------
#
# Known drift on every `terraform plan` (do not apply blindly):
#   - primary_container.image: `:<sha>` vs `:latest`. deploy.yml pushes
#     both tags to the same digest (deploy.yml:62-64), so this is a no-op
#     image-wise but TF treats the strings as different.
#   - network_configuration.security_groups: an Express-Mode-managed SG
#     gets attached live; TF wants it gone.
#   - ingress_paths / service_revision_arn: AWS-managed, change on every
#     deploy.
# We can't `ignore_changes` these because the resource is inside a registry
# module with no lifecycle hook exposed. Vendoring the module to add one
# isn't worth the maintenance burden — the diffs are cosmetic.
module "api_express" {
  # Exact pin. `~> 7.5` would allow 7.6/7.7 which have historically broken
  # this module's inputs. Bump manually after verifying a new release.
  source  = "terraform-aws-modules/ecs/aws//modules/express-service"
  version = "= 7.5.0"

  # -v2 suffix because the original lean-agent-api service is stuck in
  # INACTIVE for ~1h after a forced delete, and Express Mode blocks name
  # reuse during that window. Drop the suffix on a future apply once the
  # INACTIVE record has aged out (or just leave it).
  name    = "${local.name_prefix}-api-v2"
  cluster = aws_ecs_cluster.main.name

  cpu    = var.api_cpu
  memory = var.api_memory

  primary_container = {
    image          = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    container_port = 8000

    environment = [
      { name = "TEMPORAL_HOST", value = var.temporal_host },
      { name = "TEMPORAL_NAMESPACE", value = var.temporal_namespace },
      { name = "TEMPORAL_TLS", value = "true" },
      { name = "GITHUB_APP_INSTALL_URL", value = var.github_app_install_url },
      { name = "LANGFUSE_HOST", value = var.langfuse_public_url },
      { name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = var.otel_exporter_otlp_endpoint },
      { name = "DEPLOYMENT_ENV", value = "dev" },
    ]

    secret = [
      { name = "ANTHROPIC_API_KEY", value_from = aws_secretsmanager_secret.anthropic.arn },
      { name = "XAI_API_KEY", value_from = aws_secretsmanager_secret.xai.arn },
      { name = "OPENAI_API_KEY", value_from = aws_secretsmanager_secret.openai.arn },
      { name = "GITHUB_APP_ID", value_from = aws_secretsmanager_secret.github_app_id.arn },
      { name = "GITHUB_APP_PRIVATE_KEY", value_from = aws_secretsmanager_secret.github_app_private_key.arn },
      # Webhook HMAC secret — only the API needs this (worker doesn't handle webhooks).
      { name = "GITHUB_APP_WEBHOOK_SECRET", value_from = aws_secretsmanager_secret.github_app_webhook_secret.arn },
      { name = "TEMPORAL_API_KEY", value_from = aws_secretsmanager_secret.temporal_api_key.arn },
      { name = "PG_DSN", value_from = aws_secretsmanager_secret.pg_dsn.arn },
      { name = "LANGFUSE_PUBLIC_KEY", value_from = aws_secretsmanager_secret.langfuse_public_key.arn },
      { name = "LANGFUSE_SECRET_KEY", value_from = aws_secretsmanager_secret.langfuse_secret_key.arn },
    ]
  }

  # Reuse the IAM roles from iam.tf instead of letting the module create new
  # ones. Disabling create_*_iam_role makes the module honour the _arn vars.
  create_execution_iam_role      = false
  execution_iam_role_arn         = aws_iam_role.exec.arn
  create_infrastructure_iam_role = false
  infrastructure_iam_role_arn    = aws_iam_role.express_infra.arn

  health_check_path = "/health"

  network_configuration = {
    subnets = data.aws_subnets.target.ids
  }

  scaling_target = {
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = tostring(var.api_cpu_target)
    min_task_count            = var.api_min_tasks
    max_task_count            = var.api_max_tasks
  }

  vpc_id = data.aws_vpc.target.id

  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = {
    Project = var.project_name
  }
}
