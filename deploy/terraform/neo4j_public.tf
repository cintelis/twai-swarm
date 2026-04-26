#-----------------------------------------------------------------------------
# Public TLS endpoint for Neo4j — NLB + ACM + Cloudflare DNS.
#
# Provides a stable, TLS-encrypted hostname for the Neo4j browser + Bolt
# connections from developer laptops. Replaces the IP-based access path
# (var.allowed_dev_ips → public-ip-of-the-task) which churns on task
# replacement.
#
# Architecture:
#
#   laptop  ─┐
#            │ TLS (cert from ACM)
#            ▼
#       NLB :443       (terminates TLS, forwards plain HTTP to Neo4j :7474)
#       NLB :7687      (terminates TLS, forwards plain Bolt to Neo4j :7687)
#            │
#            ▼
#       Neo4j Fargate task (CloudMap DNS for worker; this NLB for laptops)
#
# Why NLB and not ALB:
#   - ALB only speaks HTTP/HTTPS; Bolt is a custom TCP protocol over 7687.
#   - NLB handles both HTTP browser (with TLS termination) AND raw TCP
#     for Bolt (also with TLS termination via TLS listener type).
#
# Two-step apply:
#   1. Set var.neo4j_public_hostname in tfvars, run `terraform apply`.
#      Apply WILL pause/fail at aws_acm_certificate_validation — ACM
#      needs the validation CNAME in DNS before it issues the cert.
#      Outputs print the validation record to add to Cloudflare.
#   2. Add the validation CNAME in Cloudflare (DNS-only, gray cloud).
#      Also add the friendly CNAME `<hostname>` → NLB DNS (also gray;
#      Cloudflare proxy can't tunnel non-HTTP for Bolt).
#   3. Re-run `terraform apply` — ACM picks up the validation, cert
#      issues, NLB listener attaches, you're done.
#
# Cloudflare proxy note:
#   CF free plan can't proxy raw TCP (Bolt 7687), so both the friendly
#   hostname AND the ACM validation CNAME must be DNS-only (gray cloud).
#-----------------------------------------------------------------------------

locals {
  neo4j_public_enabled = var.neo4j_public_hostname != ""
}

#-----------------------------------------------------------------------------
# ACM certificate — DNS validation, manual via Cloudflare.
#-----------------------------------------------------------------------------
resource "aws_acm_certificate" "neo4j" {
  count             = local.neo4j_public_enabled ? 1 : 0
  domain_name       = var.neo4j_public_hostname
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Project   = var.project_name
    Component = "neo4j-public"
  }
}

#-----------------------------------------------------------------------------
# Network Load Balancer — internet-facing, in the public subnets.
# Security group attachment requires NLB-style SG (added late 2023);
# preserve_client_ip=false on target groups so SG ingress can be from
# the NLB SG instead of the rotating client IPs.
#-----------------------------------------------------------------------------
resource "aws_security_group" "neo4j_nlb" {
  count  = local.neo4j_public_enabled ? 1 : 0
  name   = "${local.name_prefix}-neo4j-nlb-sg"
  vpc_id = data.aws_vpc.target.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_dev_ips
    description = "HTTPS browser from dev IPs"
  }

  ingress {
    from_port   = 7687
    to_port     = 7687
    protocol    = "tcp"
    cidr_blocks = var.allowed_dev_ips
    description = "Bolt+TLS from dev IPs"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "neo4j" {
  count              = local.neo4j_public_enabled ? 1 : 0
  name               = "${local.name_prefix}-neo4j-nlb"
  load_balancer_type = "network"
  internal           = false
  subnets            = data.aws_subnets.target.ids
  security_groups    = [aws_security_group.neo4j_nlb[0].id]

  # Cross-zone helps when only one Neo4j task exists (single-AZ at any
  # moment) and the NLB has zonal IPs in multiple AZs.
  enable_cross_zone_load_balancing = true

  tags = {
    Project   = var.project_name
    Component = "neo4j-public"
  }
}

#-----------------------------------------------------------------------------
# Target groups — IP-targeted (Fargate uses awsvpc).
# preserve_client_ip=false so the Neo4j SG only needs to allow the NLB SG,
# not the dev IP range.
#-----------------------------------------------------------------------------
resource "aws_lb_target_group" "neo4j_browser" {
  count                = local.neo4j_public_enabled ? 1 : 0
  name                 = "${local.name_prefix}-neo4j-browser"
  port                 = 7474
  protocol             = "TCP"
  target_type          = "ip"
  vpc_id               = data.aws_vpc.target.id
  preserve_client_ip   = false
  deregistration_delay = 10

  health_check {
    protocol            = "HTTP"
    port                = "7474"
    path                = "/"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_target_group" "neo4j_bolt" {
  count                = local.neo4j_public_enabled ? 1 : 0
  name                 = "${local.name_prefix}-neo4j-bolt"
  port                 = 7687
  protocol             = "TCP"
  target_type          = "ip"
  vpc_id               = data.aws_vpc.target.id
  preserve_client_ip   = false
  deregistration_delay = 10

  health_check {
    protocol            = "TCP"
    port                = "7687"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

#-----------------------------------------------------------------------------
# Listeners — TLS termination at the NLB; plain TCP to the targets.
# Browser: HTTPS (443) → HTTP (7474).
# Bolt:    TLS  (7687) → plain Bolt (7687).
#-----------------------------------------------------------------------------
resource "aws_lb_listener" "neo4j_browser_tls" {
  count             = local.neo4j_public_enabled ? 1 : 0
  load_balancer_arn = aws_lb.neo4j[0].arn
  port              = 443
  protocol          = "TLS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.neo4j[0].arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.neo4j_browser[0].arn
  }
}

resource "aws_lb_listener" "neo4j_bolt_tls" {
  count             = local.neo4j_public_enabled ? 1 : 0
  load_balancer_arn = aws_lb.neo4j[0].arn
  port              = 7687
  protocol          = "TLS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.neo4j[0].arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.neo4j_bolt[0].arn
  }
}

#-----------------------------------------------------------------------------
# Outputs — feed the user the records they need to paste into Cloudflare.
#-----------------------------------------------------------------------------
output "neo4j_acm_validation_records" {
  description = <<-EOT
    DNS records to add in Cloudflare to validate the ACM certificate.
    Add as DNS-only (gray cloud) CNAME records. ACM polls until they
    resolve, then issues the cert. Re-run `terraform apply` after to
    finish wiring the NLB listener.
  EOT
  value = local.neo4j_public_enabled ? [
    for opt in aws_acm_certificate.neo4j[0].domain_validation_options : {
      name  = opt.resource_record_name
      type  = opt.resource_record_type
      value = opt.resource_record_value
    }
  ] : []
}

output "neo4j_nlb_dns" {
  description = "NLB hostname. Add the friendly Neo4j hostname as a DNS-only CNAME pointing at this value in Cloudflare."
  value       = local.neo4j_public_enabled ? aws_lb.neo4j[0].dns_name : ""
}

output "neo4j_public_url" {
  description = "Friendly Neo4j browser URL once Cloudflare DNS resolves."
  value       = local.neo4j_public_enabled ? "https://${var.neo4j_public_hostname}" : ""
}
