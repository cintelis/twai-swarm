#-----------------------------------------------------------------------------
# RDS Postgres. pgvector ships with Postgres 16 on RDS -- just enable the
# extension after the DB is up (done by init.sql in the first worker boot,
# or you can psql it manually).
#
# Lean defaults: db.t4g.micro, 20 GB, single-AZ. Upgrade when you care.
#-----------------------------------------------------------------------------

resource "aws_db_subnet_group" "pg" {
  name       = "${local.name_prefix}-pg"
  subnet_ids = data.aws_subnets.target.ids
}

resource "aws_security_group" "pg" {
  name   = "${local.name_prefix}-pg-sg"
  vpc_id = data.aws_vpc.target.id

  # Only ECS tasks can reach Postgres
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.tasks.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "pg" {
  identifier             = "${local.name_prefix}-pg"
  engine                 = "postgres"
  engine_version         = "16.13"
  instance_class         = var.db_instance_class
  allocated_storage      = var.db_allocated_storage
  storage_type           = "gp3"
  db_name                = "agentdb"
  username               = "postgres"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.pg.name
  vpc_security_group_ids = [aws_security_group.pg.id]
  publicly_accessible    = false
  apply_immediately      = true
  multi_az               = var.db_multi_az

  # Prod: flip db_deletion_protection=true and bump db_backup_retention_days.
  # Skip_final_snapshot mirrors deletion_protection — if you guard the DB,
  # also keep its goodbye snapshot.
  backup_retention_period = var.db_backup_retention_days
  deletion_protection     = var.db_deletion_protection
  skip_final_snapshot     = !var.db_deletion_protection
}
