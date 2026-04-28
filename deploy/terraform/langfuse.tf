#-----------------------------------------------------------------------------
# Langfuse — self-hosted LLM observability.
#
# Single ECS Express service running langfuse/langfuse:2 against a new
# database + user in the existing RDS instance. Keeps ops surface minimal
# (one container, no ClickHouse, no Redis) while validating the self-host
# pattern the greenfield will use at scale via Helm.
#
# First-run:
#   1. terraform apply (creates service + DB + secrets)
#   2. Visit the on.aws URL output below
#   3. Sign up as the first admin user
#   4. Create a project, copy public_key + secret_key
#   5. Put those into tfvars as langfuse_public_key + langfuse_secret_key
#   6. terraform apply again (threads them into worker + API)
#   7. Next LLM call populates traces
#
# See _agent-instructions/twai-swarm/langfuse-setup.md for full walkthrough.
#-----------------------------------------------------------------------------

# Random secrets generated at apply-time (not tfvars) so they're never
# checked in or shared. Stored in SM; rotated by tainting these resources.
resource "random_password" "langfuse_nextauth_secret" {
  length  = 32
  special = false
}

resource "random_password" "langfuse_salt" {
  length  = 32
  special = false
}

resource "random_password" "langfuse_db_password" {
  length  = 32
  special = false
}

locals {
  # Connection string Langfuse uses to reach its own database. Points at the
  # existing RDS instance but a dedicated database + user with scoped perms.
  langfuse_db_url = format(
    "postgresql://langfuse_app:%s@%s:%d/langfuse",
    random_password.langfuse_db_password.result,
    aws_db_instance.pg.address,
    aws_db_instance.pg.port,
  )
}

resource "aws_secretsmanager_secret" "langfuse_nextauth_secret" {
  name                    = "${local.name_prefix}/langfuse-nextauth-secret"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_nextauth_secret" {
  secret_id     = aws_secretsmanager_secret.langfuse_nextauth_secret.id
  secret_string = random_password.langfuse_nextauth_secret.result
}

resource "aws_secretsmanager_secret" "langfuse_salt" {
  name                    = "${local.name_prefix}/langfuse-salt"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_salt" {
  secret_id     = aws_secretsmanager_secret.langfuse_salt.id
  secret_string = random_password.langfuse_salt.result
}

resource "aws_secretsmanager_secret" "langfuse_db_url" {
  name                    = "${local.name_prefix}/langfuse-db-url"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_db_url" {
  secret_id     = aws_secretsmanager_secret.langfuse_db_url.id
  secret_string = local.langfuse_db_url
}

# Also store the DB password on its own so bootstrap_db.py can CREATE USER
# without parsing it out of the URL.
resource "aws_secretsmanager_secret" "langfuse_db_password" {
  name                    = "${local.name_prefix}/langfuse-db-password"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_db_password" {
  secret_id     = aws_secretsmanager_secret.langfuse_db_password.id
  secret_string = random_password.langfuse_db_password.result
}

# Langfuse API keys — placeholders until the user signs in and creates a
# project. The swarm reads these from SM; when they're UNSET (first deploy),
# observability.py skips Langfuse tracing gracefully.
resource "aws_secretsmanager_secret" "langfuse_public_key" {
  name                    = "${local.name_prefix}/langfuse-public-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_public_key" {
  secret_id     = aws_secretsmanager_secret.langfuse_public_key.id
  secret_string = var.langfuse_public_key != "" ? var.langfuse_public_key : "UNSET"
}

resource "aws_secretsmanager_secret" "langfuse_secret_key" {
  name                    = "${local.name_prefix}/langfuse-secret-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "langfuse_secret_key" {
  secret_id     = aws_secretsmanager_secret.langfuse_secret_key.id
  secret_string = var.langfuse_secret_key != "" ? var.langfuse_secret_key : "UNSET"
}

#-----------------------------------------------------------------------------
# ECS Express service for Langfuse.
#
# Same Express Mode module the main API uses. Gets its own on.aws hostname,
# its own ACM cert, its own autoscaling. Single task for dev; bump api_min_tasks
# if trace volume justifies it.
#-----------------------------------------------------------------------------
# See ecs.tf::module.api_express for the known-drift rationale — same module,
# same cosmetic diffs (image tag, AWS-managed SG, ingress_paths). Don't apply
# this module's diffs without checking they're still in the known set.
module "langfuse_express" {
  source  = "terraform-aws-modules/ecs/aws//modules/express-service"
  version = "= 7.5.0"

  name    = "${local.name_prefix}-langfuse"
  cluster = aws_ecs_cluster.main.name

  cpu    = 512
  memory = 1024

  primary_container = {
    # v2 = single-container, Postgres-only. Greenfield will use v3 + ClickHouse
    # via the official Helm chart; v2 is fine for the dev testbed and lets us
    # prove the self-host pattern without the ClickHouse ops surface.
    image          = "langfuse/langfuse:2"
    container_port = 3000

    environment = [
      # NEXTAUTH_URL is read by next-auth for cookie scoping. Set after first
      # apply once the on.aws URL is known; until then Langfuse uses the
      # Host header (works for direct browser access).
      { name = "NEXTAUTH_URL", value = var.langfuse_public_url },
      { name = "TELEMETRY_ENABLED", value = "false" },
      { name = "LANGFUSE_CSP_ENFORCE_HTTPS", value = "true" },
      { name = "LANGFUSE_LOG_LEVEL", value = "info" },
      # Disable sign-up once the first admin user has been created. Toggle
      # via the tfvar, then terraform apply.
      { name = "AUTH_DISABLE_SIGNUP", value = var.langfuse_disable_signup ? "true" : "false" },
    ]

    secret = [
      { name = "DATABASE_URL", value_from = aws_secretsmanager_secret.langfuse_db_url.arn },
      { name = "NEXTAUTH_SECRET", value_from = aws_secretsmanager_secret.langfuse_nextauth_secret.arn },
      { name = "SALT", value_from = aws_secretsmanager_secret.langfuse_salt.arn },
    ]
  }

  create_execution_iam_role      = false
  execution_iam_role_arn         = aws_iam_role.exec.arn
  create_infrastructure_iam_role = false
  infrastructure_iam_role_arn    = aws_iam_role.express_infra.arn

  health_check_path = "/api/public/health"

  network_configuration = {
    subnets = data.aws_subnets.target.ids
  }

  scaling_target = {
    auto_scaling_metric       = "AVERAGE_CPU"
    auto_scaling_target_value = "70"
    min_task_count            = 1
    max_task_count            = 3
  }

  vpc_id = data.aws_vpc.target.id

  security_group_egress_rules = {
    all = {
      ip_protocol = "-1"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = {
    Project   = var.project_name
    Component = "langfuse"
  }
}

output "langfuse_url" {
  # The express-service module's service_url synthesises a friendly hostname
  # (https://<name>.ecs.<region>.on.aws/) but AWS doesn't auto-create that
  # DNS record — the real URL is the random `le-XXXX` endpoint allocated
  # under ingress_paths. Read it directly from the resource so the output
  # is always reachable.
  value       = "https://${module.langfuse_express.ingress_paths[0].endpoint}"
  description = "Langfuse UI URL. Sign up as admin on first visit, create a project, then set langfuse_public_key + langfuse_secret_key in tfvars."
}
