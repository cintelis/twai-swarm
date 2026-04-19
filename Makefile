# Lean Agent Framework -- dev shortcuts.
# `make help` lists targets.

.PHONY: help up down logs api worker tail test fmt lint \
        tf-init tf-plan tf-apply tf-destroy tf-fmt \
        build push deploy-local bootstrap-local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Local dev ──────────────────────────────────────────────────────────────

up: ## Start Temporal + Postgres (docker-compose)
	docker compose up -d
	@echo "→ Temporal UI:  http://localhost:8080"
	@echo "→ Postgres:     localhost:5432 (postgres/postgres)"

down: ## Stop local stack
	docker compose down

logs: ## Tail docker-compose logs
	docker compose logs -f

api: ## Run API locally (needs `make up` first)
	uvicorn app.api:app --reload --port 8000

worker: ## Run worker locally (needs `make up` first)
	python -m app.worker

bootstrap-local: ## Apply init.sql to local Postgres
	docker exec -i $$(docker compose ps -q postgres) psql -U postgres -d agentdb < db/init.sql

# ─── Quality ────────────────────────────────────────────────────────────────

fmt: ## Format Python code
	ruff format app/

lint: ## Lint Python code
	ruff check app/

test: ## Syntax + import check (add pytest later)
	python -m compileall -q app/
	python -c "from app import api, worker, workflows, activities, router, db, config"

# ─── Terraform ──────────────────────────────────────────────────────────────

TF_DIR        = deploy/terraform
TF_BOOT_DIR   = deploy/bootstrap
TF_BACKEND    = $(TF_DIR)/backend.conf

tf-bootstrap: ## ONE-TIME: create S3 state bucket + DynamoDB lock table + KMS key
	cd $(TF_BOOT_DIR) && terraform init && terraform apply
	@echo ""
	@echo "→ Writing backend.conf from bootstrap output..."
	@cd $(TF_BOOT_DIR) && terraform output -raw backend_config_hcl > ../terraform/backend.conf
	@echo "→ Done. Next: make tf-init"

tf-init: ## terraform init (uses backend.conf from bootstrap)
	@test -f $(TF_BACKEND) || (echo "ERROR: $(TF_BACKEND) not found. Run 'make tf-bootstrap' first." && exit 1)
	cd $(TF_DIR) && terraform init -backend-config=backend.conf

tf-init-migrate: ## terraform init -migrate-state (first run migrating off local state)
	@test -f $(TF_BACKEND) || (echo "ERROR: $(TF_BACKEND) not found. Run 'make tf-bootstrap' first." && exit 1)
	cd $(TF_DIR) && terraform init -backend-config=backend.conf -migrate-state

tf-fmt: ## terraform fmt
	cd $(TF_DIR) && terraform fmt -recursive
	cd $(TF_BOOT_DIR) && terraform fmt -recursive

tf-plan: ## terraform plan
	cd $(TF_DIR) && terraform plan

tf-apply: ## terraform apply
	cd $(TF_DIR) && terraform apply

tf-destroy: ## terraform destroy (careful! does NOT destroy the state bootstrap)
	cd $(TF_DIR) && terraform destroy

# ─── Manual image push (before CI is wired up) ──────────────────────────────

AWS_REGION  ?= ap-southeast-2
ECR_REPO    ?= lean-agent-app

ecr-login: ## Log docker into ECR
	aws ecr get-login-password --region $(AWS_REGION) | \
	  docker login --username AWS --password-stdin \
	  $$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$(AWS_REGION).amazonaws.com

build: ## Build image
	docker build -t $(ECR_REPO):latest .

push: build ecr-login ## Push image to ECR (use for first deploy before CI exists)
	@ACCOUNT=$$(aws sts get-caller-identity --query Account --output text); \
	REPO_URL=$${ACCOUNT}.dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO); \
	docker tag $(ECR_REPO):latest $${REPO_URL}:latest; \
	docker push $${REPO_URL}:latest; \
	echo "Pushed: $${REPO_URL}:latest"
