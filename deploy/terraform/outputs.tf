output "ecr_repository_url" {
  value       = aws_ecr_repository.app.repository_url
  description = "Push images to this URL"
}

output "api_url" {
  value       = module.api_express.service_url
  description = "Public URL of the API (AWS-provided *.on.aws domain)"
}

output "rds_endpoint" {
  value       = aws_db_instance.pg.address
  description = "RDS endpoint (VPC-internal)"
  sensitive   = true
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "worker_service_name" {
  value = aws_ecs_service.worker.name
}
