# Remote-state bootstrap

One-time setup for the Terraform remote state backend. Run this **once** before the first `terraform init` in `../terraform/`.

## What it creates

- **S3 bucket** — `twai-swarm-tfstate-<account-id>` — versioned, SSE-KMS, all public access blocked, `prevent_destroy` on
- **DynamoDB table** — `twai-swarm-tflock` — pay-per-request, PITR on, CMK-encrypted
- **KMS key** — customer-managed, 30-day deletion window, rotation on — alias `alias/twai-swarm-tfstate`

## Apply

```bash
cd deploy/bootstrap
terraform init      # local state, no backend
terraform apply     # ~1 min
terraform output backend_config_hcl > ../terraform/backend.conf
```

`backend.conf` is gitignored. Every operator runs the bootstrap output command after cloning (or pastes the content from a shared note).

## Initialise the main deploy config with the backend

```bash
cd ../terraform
terraform init -backend-config=backend.conf
```

On first init after migrating from local state: add `-migrate-state` to the init command.

## What not to do

- **Do not `terraform destroy` this module** unless you've already migrated every dependent config off the state bucket. `prevent_destroy` on the bucket will block you, but the KMS key + DynamoDB table would still go — and losing either breaks access to all state files.
- **Do not put this module's own state in S3.** Local state is intentional. Commit nothing from this directory except the `.tf` files.
- **Do not share the KMS key across unrelated projects.** If you add another project with Terraform state later, create a new bootstrap module for it.

## Day-2

- Bucket versioning is on — if a bad apply corrupts state, recover via `aws s3api list-object-versions` + restore.
- If the KMS key is ever scheduled for deletion, cancel it immediately — the 30-day window is a last-chance buffer but there is no recovery once the grace period ends.
- Costs: ~$1/mo (KMS key) + negligible S3 + DynamoDB on idle. Well worth it for the corruption protection.
