#-----------------------------------------------------------------------------
# Two roles per ECS convention:
#   execution role -- used by the ECS agent (pull image, fetch secrets, write logs)
#   task role      -- used by YOUR code inside the container (none needed here;
#                     your code calls Anthropic + Temporal Cloud, both external)
#-----------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role
resource "aws_iam_role" "exec" {
  name               = "${local.name_prefix}-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "exec_managed" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Grant execution role access to pull our secrets
data "aws_iam_policy_document" "secrets_access" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.anthropic.arn,
      aws_secretsmanager_secret.xai.arn,
      aws_secretsmanager_secret.temporal_api_key.arn,
      aws_secretsmanager_secret.pg_dsn.arn,
    ]
  }
}

resource "aws_iam_role_policy" "exec_secrets" {
  name   = "${local.name_prefix}-exec-secrets"
  role   = aws_iam_role.exec.id
  policy = data.aws_iam_policy_document.secrets_access.json
}

# Task role -- currently empty; add policies here if your code ever calls AWS APIs directly.
resource "aws_iam_role" "task" {
  name               = "${local.name_prefix}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

#-----------------------------------------------------------------------------
# Infrastructure role for ECS Express Mode -- lets Express manage ALB/target
# groups/security groups on your behalf. AWS-managed policy covers it.
#
# Trust policy: ecs.amazonaws.com (the SERVICE), not ecs-tasks.amazonaws.com
# (the TASKS). Express uses the service principal to provision ALB/TGs/SGs.
#-----------------------------------------------------------------------------
data "aws_iam_policy_document" "ecs_service_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "express_infra" {
  name               = "${local.name_prefix}-express-infra"
  assume_role_policy = data.aws_iam_policy_document.ecs_service_assume.json
}

resource "aws_iam_role_policy_attachment" "express_infra_managed" {
  role       = aws_iam_role.express_infra.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
}
