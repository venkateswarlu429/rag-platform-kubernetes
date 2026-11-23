# ECR repository for the RAG API image.
# Build & push (from the repo root, after `aws ecr get-login-password`):
#   docker build -t <account>.dkr.ecr.<region>.amazonaws.com/rag-platform-api:latest .
#   docker push <account>.dkr.ecr.<region>.amazonaws.com/rag-platform-api:latest
resource "aws_ecr_repository" "rag_api" {
  name                 = "rag-platform-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project = "rag-platform"
  }
}

# Keep only the most recent images to avoid unbounded storage cost.
resource "aws_ecr_lifecycle_policy" "rag_api" {
  repository = aws_ecr_repository.rag_api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
