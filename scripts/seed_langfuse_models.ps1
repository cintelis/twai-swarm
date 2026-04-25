<#
.SYNOPSIS
    Pull Langfuse keys from Secrets Manager and run seed_langfuse_models.py.

.DESCRIPTION
    Wraps the Python seeder so you don't have to copy keys around. Reads the
    same SM secrets the worker uses, exports them as env vars for the
    Python script, runs it.

    Run after:
      - First Langfuse deploy (creates the model entries)
      - Any pricing change in app/router.py MODELS

.PARAMETER Region
    AWS region. Defaults to ap-southeast-2.
#>
param(
    [string]$Region = "ap-southeast-2"
)

$ErrorActionPreference = "Stop"

Write-Host "[seed-models] reading Langfuse keys from Secrets Manager..."

$env:LANGFUSE_HOST = aws secretsmanager get-secret-value `
    --region $Region --secret-id lean-agent/langfuse-public-url `
    --query SecretString --output text 2>$null

# Fall back to terraform output if the URL secret doesn't exist (older deploys
# didn't store it in SM — only in tfvars).
if (-not $env:LANGFUSE_HOST) {
    Push-Location (Join-Path $PSScriptRoot ".." "deploy/terraform")
    try {
        $env:LANGFUSE_HOST = (terraform output -raw langfuse_url).Trim()
    } finally {
        Pop-Location
    }
}

$env:LANGFUSE_PUBLIC_KEY = aws secretsmanager get-secret-value `
    --region $Region --secret-id lean-agent/langfuse-public-key `
    --query SecretString --output text

$env:LANGFUSE_SECRET_KEY = aws secretsmanager get-secret-value `
    --region $Region --secret-id lean-agent/langfuse-secret-key `
    --query SecretString --output text

if ($env:LANGFUSE_PUBLIC_KEY -eq "UNSET" -or $env:LANGFUSE_SECRET_KEY -eq "UNSET") {
    Write-Error "[seed-models] Langfuse keys still 'UNSET' in Secrets Manager. Sign up + create project first, then update tfvars + apply."
    exit 1
}

Write-Host "[seed-models] target host: $env:LANGFUSE_HOST"
Write-Host "[seed-models] running seeder..."
Write-Host ""

$repoRoot = Join-Path $PSScriptRoot ".."
Push-Location $repoRoot
try {
    python scripts/seed_langfuse_models.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
