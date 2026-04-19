#-----------------------------------------------------------------------------
# GitHub Actions OIDC deploy role.
#
# Lets your GitHub Actions workflows get short-lived AWS credentials by
# exchanging a GitHub-signed JWT. No long-lived access keys. No rotation.
#
# Set `github_repo` to "owner/repo" (e.g. "cintelis/twai-swarm").
# The role will ONLY be assumable from workflows in that repo.
#-----------------------------------------------------------------------------

variable "github_repo" {
  description = "GitHub repo in owner/repo form. Only this repo can assume the deploy role."
  type        = string
}

# GitHub's OIDC provider. One per AWS account -- if you already have this
# resource from another project, import it instead of re-creating.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"] # GitHub's root cert thumbprint
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Only allow workflows in the specified repo. Covers any branch/tag/PR.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${local.name_prefix}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

# Permissions scoped to what the deploy workflow actually does:
# - push to our ECR repo
# - update the worker task def + service
# - update the Express service
# - describe cluster/services for smoke tests
data "aws_iam_policy_document" "github_deploy" {
  # ECR push
  statement {
    sid = "ECRAuth"
    actions = [
      "ecr:GetAuthorizationToken",
    ]
    resources = ["*"]
  }
  statement {
    sid = "ECRPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.app.arn]
  }

  # ECS deployments
  statement {
    sid = "ECSDeploy"
    actions = [
      "ecs:DescribeServices",
      "ecs:DescribeTaskDefinition",
      "ecs:DescribeTasks",
      "ecs:ListTasks",
      "ecs:RegisterTaskDefinition",
      "ecs:UpdateService",
      "ecs:DescribeClusters",
      # Express Mode APIs
      "ecs:DescribeExpressGatewayService",
      "ecs:UpdateExpressGatewayService",
      "ecs:ListExpressGatewayServices",
    ]
    resources = ["*"]
  }

  # PassRole so the new task definition can reference our exec/task roles
  statement {
    sid     = "PassRoles"
    actions = ["iam:PassRole"]
    resources = [
      aws_iam_role.exec.arn,
      aws_iam_role.task.arn,
      aws_iam_role.express_infra.arn,
    ]
  }

  # Remote-state access — CI runs `terraform apply -target=...` to regenerate
  # the worker task def before ECS render. Needs read/write on the state file,
  # lock acquisition on DynamoDB, and KMS for envelope decryption.
  statement {
    sid = "TFStateBucket"
    actions = [
      "s3:ListBucket",
      "s3:GetBucketVersioning",
    ]
    resources = [data.aws_s3_bucket.state.arn]
  }
  statement {
    sid = "TFStateObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${data.aws_s3_bucket.state.arn}/*"]
  }
  statement {
    sid = "TFStateLock"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
      "dynamodb:DescribeTable",
    ]
    resources = [data.aws_dynamodb_table.lock.arn]
  }
  statement {
    sid = "TFStateKMS"
    actions = [
      "kms:Encrypt",
      "kms:Decrypt",
      "kms:GenerateDataKey",
      "kms:DescribeKey",
    ]
    resources = [data.aws_kms_alias.state.target_key_arn]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "${local.name_prefix}-github-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}

# CI runs `terraform apply -target=local_file.worker_task_def_json` which
# triggers refresh of every dependency of the worker task definition: IAM
# roles, Secrets Manager secrets, the ECR repo, the log group, the RDS
# instance, the VPC/subnets/SGs. Enumerating each Describe/Get action gets
# unwieldy fast — granting the AWS-managed ReadOnlyAccess covers them all
# for a small security trade-off (the deploy role can read secret values,
# acceptable for dev). Narrow this in prod.
resource "aws_iam_role_policy_attachment" "github_deploy_readonly" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

output "github_deploy_role_arn" {
  value       = aws_iam_role.github_deploy.arn
  description = "Set this as the AWS_DEPLOY_ROLE_ARN secret in GitHub"
}
