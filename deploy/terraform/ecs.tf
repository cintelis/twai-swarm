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
resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true
      command   = ["python", "-m", "app.worker"]

      portMappings = [
        { containerPort = 8001, protocol = "tcp" }
      ]

      environment = [
        { name = "TEMPORAL_HOST",      value = var.temporal_host },
        { name = "TEMPORAL_NAMESPACE", value = var.temporal_namespace },
        { name = "TEMPORAL_TLS",       value = "true" },
        # Empty = handle all queues. Narrow this on specialised worker services.
        { name = "TEMPORAL_QUEUES",    value = "" },
      ]

      secrets = [
        { name = "ANTHROPIC_API_KEY",      valueFrom = aws_secretsmanager_secret.anthropic.arn },
        { name = "XAI_API_KEY",            valueFrom = aws_secretsmanager_secret.xai.arn },
        { name = "TEMPORAL_CLIENT_CERT",   valueFrom = aws_secretsmanager_secret.temporal_cert.arn },
        { name = "TEMPORAL_CLIENT_KEY",    valueFrom = aws_secretsmanager_secret.temporal_key.arn },
        { name = "PG_DSN",                 valueFrom = aws_secretsmanager_secret.pg_dsn.arn },
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
  ])
}

#-----------------------------------------------------------------------------
# Service: Worker (regular Fargate -- no ALB needed)
#-----------------------------------------------------------------------------
resource "aws_ecs_service" "worker" {
  name            = "${local.name_prefix}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 1
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

  # Don't re-deploy on every apply unless the task def actually changes.
  lifecycle {
    ignore_changes = [desired_count]
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
module "api_express" {
  # Exact pin. `~> 7.5` would allow 7.6/7.7 which have historically broken
  # this module's inputs. Bump manually after verifying a new release.
  source  = "terraform-aws-modules/ecs/aws//modules/express-service"
  version = "= 7.5.0"

  name    = "${local.name_prefix}-api"
  cluster = aws_ecs_cluster.main.name

  cpu    = 512
  memory = 1024

  primary_container = {
    image          = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    container_port = 8000

    environment = [
      { name = "TEMPORAL_HOST",      value = var.temporal_host },
      { name = "TEMPORAL_NAMESPACE", value = var.temporal_namespace },
      { name = "TEMPORAL_TLS",       value = "true" },
    ]

    secrets = [
      { name = "ANTHROPIC_API_KEY",    valueFrom = aws_secretsmanager_secret.anthropic.arn },
      { name = "XAI_API_KEY",          valueFrom = aws_secretsmanager_secret.xai.arn },
      { name = "TEMPORAL_CLIENT_CERT", valueFrom = aws_secretsmanager_secret.temporal_cert.arn },
      { name = "TEMPORAL_CLIENT_KEY",  valueFrom = aws_secretsmanager_secret.temporal_key.arn },
      { name = "PG_DSN",               valueFrom = aws_secretsmanager_secret.pg_dsn.arn },
    ]
  }

  execution_role_arn      = aws_iam_role.exec.arn
  infrastructure_role_arn = aws_iam_role.express_infra.arn

  health_check = {
    path = "/health"
  }

  network_configuration = {
    subnets = data.aws_subnets.target.ids
  }

  scaling_target = {
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = "70"
    min_task_count            = 1
    max_task_count            = 5
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
