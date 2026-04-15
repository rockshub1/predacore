# Local Kubernetes Setup Guide using Minikube

## Prerequisites
1. Install Docker Desktop for Mac (requires Intel or Apple Silicon chip)
2. Install Homebrew package manager
3. At least 4GB RAM and 2 CPU cores available

## Installation Steps

### 1. Install Minikube
```bash
brew install minikube
```

### 2. Start Kubernetes Cluster
```bash
minikube start \
  --driver=docker \
  --cpus=2 \
  --memory=4096 \
  --disk-size=20g \
  --addons=metrics-server,ingress,dashboard
```

### 3. Verify Installation
```bash
minikube status
kubectl cluster-info
```

### 4. Enable Ingress Controller
```bash
minikube addons enable ingress
```

### 5. Setup Kubernetes Dashboard
```bash
minikube dashboard
```

## Cluster Configuration

### Storage Setup
Create a persistent volume for Redis:
```yaml
# redis-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: redis-pv
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/data/redis"
```

### Service Mesh (Optional)
Install Istio for advanced traffic management:
```bash
brew install istioctl
istioctl install --set profile=demo -y
```

## Deployment Instructions

### 1. Build Docker Image
```bash
docker build -t project-prometheus/api-gateway:latest \
  -f project_prometheus/src/api_gateway/Dockerfile.api_gateway .
```

### 2. Deploy to Minikube
```bash
kubectl apply -f project_prometheus/deploy/kubernetes/api_gateway.yaml
kubectl apply -f project_prometheus/docs/architecture/sequence_diagrams/api_gateway_flow.md
```

### 3. Verify Deployment
```bash
kubectl get pods,svc,deployments
minikube service api-gateway --url
```

## Common Commands
- Check pod logs: `kubectl logs -f <pod-name>`
- Port forward: `kubectl port-forward svc/api-gateway 8080:8080`
- Delete cluster: `minikube delete --all`

## Troubleshooting
- If Docker Desktop is not running: Start Docker from Applications
- For image pull errors: `minikube image load project-prometheus/api-gateway:latest`
- Check ingress: `kubectl get ingress`

## Resource Management
Recommended resource limits for local development:
```yaml
resources:
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## Next Steps
1. Configure service mesh for advanced routing
2. Set up Prometheus monitoring stack
3. Implement CI/CD pipeline for automated deployments