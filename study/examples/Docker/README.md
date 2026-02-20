# Docker & Kubernetes Examples

This directory contains comprehensive, production-ready examples for Docker and Kubernetes, demonstrating best practices for containerization and orchestration.

## Directory Structure

```
Docker/
├── 01_multi_stage/         # Multi-stage Docker build
│   ├── Dockerfile          # Optimized multi-stage build
│   ├── app.py              # Flask application
│   ├── requirements.txt    # Python dependencies
│   └── .dockerignore       # Files to exclude from build
│
├── 02_compose/             # Docker Compose stack
│   ├── docker-compose.yml  # 3-tier application stack
│   ├── app.py              # Flask app with DB and cache
│   ├── Dockerfile          # Production-ready Dockerfile
│   ├── requirements.txt    # Dependencies
│   └── .env.example        # Environment variables template
│
├── 03_k8s/                 # Kubernetes manifests
│   ├── deployment.yaml     # Deployment with HPA
│   ├── service.yaml        # Service definitions
│   ├── ingress.yaml        # Ingress with TLS
│   └── configmap.yaml      # Configuration and secrets
│
└── 04_ci_cd/               # CI/CD pipeline
    └── .github/
        └── workflows/
            └── docker-ci.yml  # GitHub Actions workflow
```

## Examples Overview

### 1. Multi-stage Build (`01_multi_stage/`)

Demonstrates Docker multi-stage builds for creating slim, secure production images.

**Key Features:**
- Multi-stage build pattern (builder + runtime)
- Security best practices (non-root user, minimal base image)
- Health checks for container orchestration
- Proper signal handling for graceful shutdown
- Optimized layer caching

**Usage:**
```bash
cd 01_multi_stage

# Build the image
docker build -t flask-app:latest .

# Run the container
docker run -d -p 5000:5000 --name flask-app flask-app:latest

# Test the application
curl http://localhost:5000
curl http://localhost:5000/health

# View logs
docker logs flask-app

# Stop and remove
docker stop flask-app
docker rm flask-app
```

**Image size comparison:**
- Without multi-stage: ~800MB
- With multi-stage: ~150MB

---

### 2. Docker Compose (`02_compose/`)

Complete 3-tier web application stack with Flask, PostgreSQL, and Redis.

**Key Features:**
- Multi-container orchestration
- Service dependencies with health checks
- Named volumes for data persistence
- Environment-based configuration
- Internal networking
- Automatic restarts

**Stack:**
- **Web**: Flask application (Python 3.11)
- **Database**: PostgreSQL 16
- **Cache**: Redis 7

**Usage:**
```bash
cd 02_compose

# Create environment file
cp .env.example .env
# Edit .env with your own values

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps

# Test the application
curl http://localhost:5000
curl http://localhost:5000/stats

# Scale the web service
docker-compose up -d --scale web=3

# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

**Services:**
- Web: http://localhost:5000
- PostgreSQL: localhost:5432 (only if exposed)
- Redis: localhost:6379 (only if exposed)

---

### 3. Kubernetes Manifests (`03_k8s/`)

Production-ready Kubernetes deployment with all essential resources.

**Key Features:**
- Deployment with 3 replicas
- Horizontal Pod Autoscaler (HPA)
- Resource limits and requests
- Liveness, readiness, and startup probes
- Multiple service types (ClusterIP, NodePort, LoadBalancer)
- Ingress with TLS termination
- ConfigMap and Secret management
- RBAC with ServiceAccount
- Pod anti-affinity for high availability

**Resources:**
- Deployment + HPA
- Services (ClusterIP, headless)
- Ingress with NGINX controller
- ConfigMap for configuration
- Secret for sensitive data

**Usage:**
```bash
cd 03_k8s

# Create namespace (optional)
kubectl create namespace flask-app

# Apply ConfigMap and Secrets first
kubectl apply -f configmap.yaml

# Apply deployment
kubectl apply -f deployment.yaml

# Apply services
kubectl apply -f service.yaml

# Apply ingress (requires ingress controller)
kubectl apply -f ingress.yaml

# Check status
kubectl get all -l app=flask-app
kubectl get pods -l app=flask-app
kubectl get svc flask-app
kubectl get ingress flask-app-ingress

# View logs
kubectl logs -l app=flask-app -f

# Port forward for local testing
kubectl port-forward svc/flask-app 8080:80
curl http://localhost:8080/health

# Scale manually
kubectl scale deployment flask-app --replicas=5

# Update image
kubectl set image deployment/flask-app flask-app=new-image:tag
kubectl rollout status deployment/flask-app

# Rollback if needed
kubectl rollout undo deployment/flask-app

# Delete all resources
kubectl delete -f .
```

**Prerequisites:**
- Kubernetes cluster (minikube, kind, or cloud provider)
- kubectl configured
- NGINX Ingress Controller (for ingress)
- cert-manager (for TLS certificates)

---

### 4. CI/CD Pipeline (`04_ci_cd/`)

Complete GitHub Actions workflow for automated Docker builds and deployments.

**Key Features:**
- Multi-platform builds (amd64, arm64)
- Docker layer caching
- Security scanning with Trivy
- SBOM generation
- Automated testing
- Kubernetes deployment
- GitHub Container Registry integration

**Pipeline Stages:**
1. Build multi-platform Docker image
2. Run container tests
3. Security vulnerability scanning
4. Generate Software Bill of Materials (SBOM)
5. Push to GitHub Container Registry
6. Deploy to Kubernetes (production only)

**Usage:**
```bash
# 1. Copy workflow to your repository
cp -r 04_ci_cd/.github /path/to/your/repo/

# 2. Set up GitHub secrets
# Go to: Settings → Secrets and variables → Actions
# Add the following secrets:
#   - KUBE_CONFIG: Base64-encoded kubeconfig
#     cat ~/.kube/config | base64 -w 0

# 3. Push to trigger workflow
git add .github/workflows/docker-ci.yml
git commit -m "Add Docker CI/CD workflow"
git push

# 4. Monitor workflow
# Go to: Actions tab in GitHub

# 5. View built images
# Go to: Packages tab in GitHub
```

**Supported Triggers:**
- Push to main, develop, or release branches
- Pull requests to main
- Version tags (v1.0.0)
- Manual workflow dispatch

---

## Best Practices Demonstrated

### Security
- Non-root user in containers
- Read-only root filesystem
- Security context with dropped capabilities
- Secret management
- Image vulnerability scanning
- SBOM generation

### Performance
- Multi-stage builds for smaller images
- Layer caching optimization
- Resource limits and requests
- Horizontal pod autoscaling
- Redis caching

### Reliability
- Health checks (liveness, readiness, startup)
- Graceful shutdown
- Automatic restarts
- Rolling updates
- Pod anti-affinity

### Observability
- Structured logging
- Health check endpoints
- Prometheus annotations
- Container insights

## Prerequisites

### Docker
```bash
# macOS
brew install docker

# Ubuntu
sudo apt-get install docker.io docker-compose

# Verify
docker --version
docker-compose --version
```

### Kubernetes (choose one)
```bash
# minikube (local)
brew install minikube
minikube start

# kind (local)
brew install kind
kind create cluster

# Cloud providers
# - AWS EKS
# - Google GKE
# - Azure AKS
```

### Tools
```bash
# kubectl
brew install kubectl

# Helm
brew install helm

# NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# cert-manager (for TLS)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

## Testing the Examples

### Quick Test: Multi-stage Build
```bash
cd 01_multi_stage
docker build -t test-flask .
docker run -d -p 5000:5000 test-flask
curl http://localhost:5000/health
docker stop $(docker ps -q -f ancestor=test-flask)
```

### Quick Test: Docker Compose
```bash
cd 02_compose
cp .env.example .env
docker-compose up -d
sleep 10
curl http://localhost:5000/health
curl http://localhost:5000/stats
docker-compose down
```

### Quick Test: Kubernetes
```bash
cd 03_k8s
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl port-forward svc/flask-app 8080:80 &
sleep 5
curl http://localhost:8080/health
kubectl delete -f .
```

## Troubleshooting

### Docker Build Issues
```bash
# Clear build cache
docker builder prune -a

# Build without cache
docker build --no-cache -t flask-app .

# Check disk space
docker system df
```

### Docker Compose Issues
```bash
# View logs
docker-compose logs -f web

# Restart specific service
docker-compose restart web

# Rebuild and restart
docker-compose up -d --build
```

### Kubernetes Issues
```bash
# Check pod status
kubectl get pods -l app=flask-app
kubectl describe pod <pod-name>

# View events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check logs
kubectl logs <pod-name> -f

# Debug with ephemeral container
kubectl debug <pod-name> -it --image=busybox
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Compose Specification](https://docs.docker.com/compose/compose-file/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager](https://cert-manager.io/)

## Related Learning Materials

- [Docker Lessons](/opt/projects/01_Personal/03_Study/content/en/Docker/)
- [PostgreSQL Examples](/opt/projects/01_Personal/03_Study/examples/PostgreSQL/)
- [Git Workflows](/opt/projects/01_Personal/03_Study/content/en/Git/)

## License

These examples are provided under the MIT License. See the project root for details.
