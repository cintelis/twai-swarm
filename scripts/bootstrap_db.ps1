<#
.SYNOPSIS
    Run the one-shot DB bootstrap task on ECS.

.DESCRIPTION
    Replaces the previous Terraform null_resource + local-exec which was
    fragile on Windows (terraform's bash spawn didn't find aws.exe in PATH).

    Run this:
      - After every terraform apply that changes app/bootstrap_db.py SCHEMA_SQL
      - On first-time deploys (creates schema)
      - When schema migrations need to apply (e.g. new tenant_id columns)

    Bootstrap is idempotent — safe to run twice; no-op if nothing changed.

.PARAMETER Region
    AWS region. Defaults to ap-southeast-2.

.PARAMETER Cluster
    ECS cluster name. Defaults to lean-agent-cluster.

.PARAMETER TaskDefFamily
    Task def family. Defaults to lean-agent-db-bootstrap (the one terraform
    creates in bootstrap.tf). Latest revision is auto-selected by ECS.
#>
param(
    [string]$Region = "ap-southeast-2",
    [string]$Cluster = "lean-agent-cluster",
    [string]$TaskDefFamily = "lean-agent-db-bootstrap"
)

$ErrorActionPreference = "Stop"

# Resolve subnets + worker SG dynamically from terraform state. Reads from the
# deploy/terraform directory relative to this script's location.
$tfDir = Join-Path $PSScriptRoot ".." "deploy/terraform"

Push-Location $tfDir
try {
    $subnetIds = (terraform output -raw 2>$null) # placeholder; we'll use known values below
    # Fall back to the known dev subnets + worker SG (these are stable in dev).
    # Override via parameters if needed.
    $subnets = "subnet-043daa722fb04cf48,subnet-00ca7ed75313435af,subnet-04429466556793e8d"
    $workerSg = "sg-0b9bd1a13ffa5e437"
} finally {
    Pop-Location
}

Write-Host "[bootstrap] launching task on cluster=$Cluster family=$TaskDefFamily"

$task = aws ecs run-task `
    --region $Region `
    --cluster $Cluster `
    --task-definition $TaskDefFamily `
    --launch-type FARGATE `
    --network-configuration "awsvpcConfiguration={subnets=[$subnets],securityGroups=[$workerSg],assignPublicIp=ENABLED}" `
    --query 'tasks[0].taskArn' --output text

if (-not $task -or $task -eq "None") {
    Write-Error "[bootstrap] failed to start task"
    exit 1
}

Write-Host "[bootstrap] task ARN: $task"
Write-Host "[bootstrap] waiting for completion..."

aws ecs wait tasks-stopped --region $Region --cluster $Cluster --tasks $task

$exitCode = aws ecs describe-tasks --region $Region --cluster $Cluster --tasks $task --query 'tasks[0].containers[0].exitCode' --output text
$stopReason = aws ecs describe-tasks --region $Region --cluster $Cluster --tasks $task --query 'tasks[0].stoppedReason' --output text

Write-Host "[bootstrap] exit code: $exitCode"
if ($stopReason -and $stopReason -ne "None") {
    Write-Host "[bootstrap] stopped reason: $stopReason"
}

Write-Host ""
Write-Host "[bootstrap] tail of CloudWatch log stream:"
Write-Host "----"
aws logs tail /ecs/lean-agent --log-stream-name-prefix bootstrap --since 2m --format short

if ($exitCode -ne "0") {
    Write-Error "[bootstrap] failed with exit code $exitCode"
    exit 1
}

Write-Host ""
Write-Host "[bootstrap] DB bootstrap complete"
