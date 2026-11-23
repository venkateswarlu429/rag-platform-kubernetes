output "cluster_name" {
  description = "Name of the EKS cluster."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for the EKS cluster API server."
  value       = module.eks.cluster_endpoint
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL (used for IRSA service accounts)."
  value       = module.eks.cluster_oidc_issuer_url
}

output "ecr_repository_url" {
  description = "URL of the ECR repository for the RAG API image."
  value       = aws_ecr_repository.rag_api.repository_url
}

output "vpc_id" {
  description = "ID of the VPC created for the cluster."
  value       = module.vpc.vpc_id
}
