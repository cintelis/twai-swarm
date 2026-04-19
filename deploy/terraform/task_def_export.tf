#-----------------------------------------------------------------------------
# Export the worker task definition as a JSON file that GitHub Actions reads.
#
# This closes the loop: Terraform is the source of truth for the task def,
# but the deploy workflow needs a file to pass to amazon-ecs-render-task-definition.
# We write the current container definitions out so the workflow picks up any
# infra changes (new secrets, resized memory, etc.) on the next deploy.
#-----------------------------------------------------------------------------
resource "local_file" "worker_task_def_json" {
  filename = "${path.module}/../ecs/worker-task-def.json"
  content = jsonencode({
    family                  = aws_ecs_task_definition.worker.family
    networkMode             = "awsvpc"
    requiresCompatibilities = ["FARGATE"]
    cpu                     = "1024"
    memory                  = "2048"
    executionRoleArn        = aws_iam_role.exec.arn
    taskRoleArn             = aws_iam_role.task.arn
    containerDefinitions    = jsondecode(aws_ecs_task_definition.worker.container_definitions)
  })
}
