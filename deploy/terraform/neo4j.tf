#-----------------------------------------------------------------------------
# Neo4j — code-knowledge graph store.
#
# Self-hosted Neo4j 5 Community on ECS Fargate. Used by the repo indexer
# (Sprint 10a) to store the AST-derived call graph of scanned repositories.
# The Architect / Coder agents query this graph to navigate existing code
# without cold-reading every file.
#
# Operational shape (intentionally minimal for dev validation):
#   - Single ECS task (community edition has no clustering anyway)
#   - No EFS — graph is derivable from git, re-index on task replacement
#   - No automated snapshots — same reason; source IS the backup
#   - Internal-only, reached via CloudMap service discovery from the worker
#   - Bolt 7687 + browser 7474; browser only reachable from inside the VPC
#
# Greenfield will swap to a managed cluster (Aura Enterprise or self-hosted
# multi-instance) — the schema, queries, and worker integration stay.
#
# How to query the graph from your laptop:
#   1. Start a Session Manager port-forward to the Neo4j task:
#        aws ssm start-session --target <task-id> \
#          --document-name AWS-StartPortForwardingSession \
#          --parameters portNumber=7474,localPortNumber=7474
#   2. Open http://localhost:7474, log in as neo4j with the password from
#      Secrets Manager (lean-agent/neo4j-password).
#-----------------------------------------------------------------------------

resource "random_password" "neo4j_password" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "neo4j_password" {
  name                    = "${local.name_prefix}/neo4j-password"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "neo4j_password" {
  secret_id     = aws_secretsmanager_secret.neo4j_password.id
  secret_string = random_password.neo4j_password.result
}

# Bolt connection URL the worker uses. Value is the CloudMap DNS name
# below; we store it in SM so the worker reads it like any other secret.
resource "aws_secretsmanager_secret" "neo4j_url" {
  name                    = "${local.name_prefix}/neo4j-url"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "neo4j_url" {
  secret_id     = aws_secretsmanager_secret.neo4j_url.id
  secret_string = "bolt://neo4j.${aws_service_discovery_private_dns_namespace.internal.name}:7687"
}

#-----------------------------------------------------------------------------
# CloudMap private DNS namespace — reused for any future internal services
# (e.g. an OTel collector or pgbouncer). The worker resolves
# `neo4j.internal.lean-agent.local` to the Neo4j task's private IP.
#-----------------------------------------------------------------------------
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name        = "internal.${local.name_prefix}.local"
  description = "Private DNS for internal services (Neo4j, future)"
  vpc         = data.aws_vpc.target.id
}

resource "aws_service_discovery_service" "neo4j" {
  name = "neo4j"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.internal.id
    routing_policy = "MULTIVALUE"
    dns_records {
      ttl  = 10
      type = "A"
    }
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

#-----------------------------------------------------------------------------
# Security group — Bolt + browser reachable from worker SG only.
#-----------------------------------------------------------------------------
resource "aws_security_group" "neo4j" {
  name   = "${local.name_prefix}-neo4j-sg"
  vpc_id = data.aws_vpc.target.id

  # Bolt — application protocol the driver uses
  ingress {
    from_port       = 7687
    to_port         = 7687
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
    description     = "Bolt from worker tasks"
  }

  # HTTP browser — Neo4j's web UI. Only the worker can reach it; for human
  # access use SSM port-forward (see header comment).
  ingress {
    from_port       = 7474
    to_port         = 7474
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
    description     = "Neo4j browser UI from worker tasks"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

#-----------------------------------------------------------------------------
# CloudWatch log group — dedicated so retention can be tuned independently.
#-----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "neo4j" {
  name              = "/ecs/${local.name_prefix}-neo4j"
  retention_in_days = 14
}

#-----------------------------------------------------------------------------
# Task definition.
#
# Memory tuning notes:
#   - Neo4j 5 wants ~1G heap + 512M page cache for trivial graphs.
#   - 2 vCPU / 4G memory keeps headroom for indexing twai-swarm itself
#     (~50K nodes after a full scan) without page-cache thrash.
#   - Bump to 4 vCPU / 8G if scanning monorepos.
#-----------------------------------------------------------------------------
resource "aws_ecs_task_definition" "neo4j" {
  family                   = "${local.name_prefix}-neo4j"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.exec.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "neo4j"
      image     = "neo4j:5-community"
      essential = true

      portMappings = [
        { containerPort = 7687, protocol = "tcp" },
        { containerPort = 7474, protocol = "tcp" },
      ]

      environment = [
        # Disable the Neo4j browser's auth-enabled banner / strict TLS in dev.
        { name = "NEO4J_server_default__listen__address", value = "0.0.0.0" },
        # Keep the heap modest; Fargate kills the task if we overshoot the
        # task-level memory limit.
        { name = "NEO4J_server_memory_heap_initial__size", value = "1G" },
        { name = "NEO4J_server_memory_heap_max__size",     value = "1G" },
        { name = "NEO4J_server_memory_pagecache_size",     value = "1G" },
        # Accept the LICENSE so the container starts non-interactively.
        { name = "NEO4J_ACCEPT_LICENSE_AGREEMENT", value = "yes" },
      ]

      secrets = [
        # Format Neo4j expects: "neo4j/<password>" via NEO4J_AUTH.
        # Wired via a small entrypoint trick in the readiness comment below.
        # For now: single password secret, formatted in a wrapper command.
        { name = "_NEO4J_PASSWORD", valueFrom = aws_secretsmanager_secret.neo4j_password.arn },
      ]

      # Format NEO4J_AUTH at runtime from the password secret. Neo4j refuses
      # to start without "neo4j/<pwd>" exactly, and we don't want to bake the
      # literal "neo4j/" prefix into the SM value (so it stays the raw password).
      command = ["bash", "-c", "export NEO4J_AUTH=\"neo4j/$_NEO4J_PASSWORD\" && exec /startup/docker-entrypoint.sh neo4j"]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.neo4j.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "neo4j"
        }
      }

      # Neo4j takes ~30s to fully boot. Don't mark it healthy until Bolt is up.
      healthCheck = {
        command     = ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:7474/ || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 5
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "neo4j" {
  name            = "${local.name_prefix}-neo4j"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.neo4j.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.target.ids
    security_groups  = [aws_security_group.neo4j.id]
    # Public IP only for ECR pulls + dockerhub pulls. The SG denies inbound
    # from anywhere outside the worker SG, so this isn't a real exposure.
    assign_public_ip = true
  }

  service_registries {
    registry_arn = aws_service_discovery_service.neo4j.arn
  }

  # Container terminates with the task; Service Discovery deregisters the IP
  # within ~10s (matches our DNS TTL), worker driver retries cover the gap.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  # Neo4j is a single-writer DB — never run more than one task concurrently.
  lifecycle {
    ignore_changes = [desired_count]
  }
}

output "neo4j_dns" {
  value       = "neo4j.${aws_service_discovery_private_dns_namespace.internal.name}"
  description = "Neo4j Bolt host (port 7687) reachable from worker tasks via CloudMap DNS."
}
