# Deployment Guide

Complete guide for deploying NeuroQuant Trading System to production environments.

---

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Cloud Deployment](#cloud-deployment)
3. [Production Configuration](#production-configuration)
4. [Monitoring & Maintenance](#monitoring--maintenance)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Docker Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 10GB disk space

### Quick Start

**1. Build and Run**:

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

**2. Verify Deployment**:

```bash
# Check running containers
docker ps

# Health check
curl http://localhost:8000/health
```

**3. Access Application**:

- **Frontend**: http://localhost
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Docker Compose Configuration

**`docker-compose.yml`**:

```yaml
version: '3.8'

services:
  backend:
    build: .
    container_name: neuroquant-backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./data/trading.db
      - LOG_LEVEL=INFO
      - REDIS_ENABLED=false
    volumes:
      - ./database:/app/database
      - ./logs:/app/logs
      - ./models:/app/models
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: neuroquant-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend:/usr/share/nginx/html:ro
      - ./ssl:/etc/nginx/ssl:ro  # SSL certificates
    depends_on:
      - backend
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:alpine
    container_name: neuroquant-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: PostgreSQL for production
  postgres:
    image: postgres:15-alpine
    container_name: neuroquant-postgres
    environment:
      - POSTGRES_USER=neuroquant
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p logs models checkpoints database

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Build (Optimized)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y build-essential wget

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && make install

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

RUN mkdir -p logs models checkpoints database

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Commands

```bash
# Build specific service
docker-compose build backend

# Start specific service
docker-compose up -d backend

# View logs
docker-compose logs -f backend

# Execute command in container
docker-compose exec backend python -c "print('Hello')"

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart service
docker-compose restart backend

# Scale service (if configured)
docker-compose up -d --scale backend=3
```

---

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS (Elastic Container Service)

**1. Push Image to ECR**:

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t neuroquant .
docker tag neuroquant:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/neuroquant:latest

# Push
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/neuroquant:latest
```

**2. Create ECS Task Definition**:

```json
{
  "family": "neuroquant-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "neuroquant",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/neuroquant:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/trading"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/neuroquant",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**3. Create ECS Service**:

```bash
aws ecs create-service \
  --cluster neuroquant-cluster \
  --service-name neuroquant-service \
  --task-definition neuroquant-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

#### Using AWS EC2

**1. Launch EC2 Instance**:

- AMI: Amazon Linux 2023
- Instance type: t3.medium (2 vCPU, 4GB RAM)
- Security group: Open ports 22, 80, 443, 8000

**2. Install Docker**:

```bash
ssh ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**3. Deploy Application**:

```bash
# Clone repository
git clone https://github.com/your-repo/neuroquant.git
cd neuroquant

# Create .env file
nano .env

# Start services
docker-compose up -d
```

---

### Google Cloud Platform (GCP)

#### Using Cloud Run

**1. Build and Push to GCR**:

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/<project-id>/neuroquant .
docker push gcr.io/<project-id>/neuroquant
```

**2. Deploy to Cloud Run**:

```bash
gcloud run deploy neuroquant \
  --image gcr.io/<project-id>/neuroquant \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars DATABASE_URL=<db-url>,LOG_LEVEL=INFO \
  --allow-unauthenticated
```

---

### Microsoft Azure

#### Using Azure Container Instances

**1. Push to ACR**:

```bash
# Login to ACR
az acr login --name <registry-name>

# Build and push
docker build -t <registry-name>.azurecr.io/neuroquant .
docker push <registry-name>.azurecr.io/neuroquant
```

**2. Deploy to ACI**:

```bash
az container create \
  --resource-group neuroquant-rg \
  --name neuroquant \
  --image <registry-name>.azurecr.io/neuroquant \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --environment-variables DATABASE_URL=<db-url> LOG_LEVEL=INFO \
  --dns-name-label neuroquant
```

---

### DigitalOcean

#### Using App Platform

**1. Create `app.yaml`**:

```yaml
name: neuroquant
services:
  - name: backend
    image:
      registry_type: DOCKER_HUB
      repository: your-username/neuroquant
      tag: latest
    envs:
      - key: DATABASE_URL
        value: ${DATABASE_URL}
      - key: LOG_LEVEL
        value: INFO
    instance_count: 2
    instance_size_slug: basic-xs
    routes:
      - path: /
```

**2. Deploy**:

```bash
doctl apps create --spec app.yaml
```

---

## Production Configuration

### Environment Variables

**Production `.env`**:

```bash
# Database
DATABASE_URL=postgresql://user:password@db-host:5432/trading

# Security
SECRET_KEY=<generate-strong-secret>
CORS_ORIGINS=["https://your-domain.com"]

# Logging
LOG_LEVEL=WARNING
LOG_FILE=/var/log/neuroquant/app.log

# Performance
REDIS_ENABLED=true
REDIS_URL=redis://redis-host:6379/0

# Monitoring
SENTRY_DSN=<your-sentry-dsn>
```

### Nginx Configuration

**Production `nginx.conf`**:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        least_conn;
        server backend:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL certificates
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000" always;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;

        # Frontend
        location / {
            root /usr/share/nginx/html;
            try_files $uri $uri/ /index.html;
        }

        # API
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # WebSocket support (if needed)
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### SSL/TLS Certificates

**Using Let's Encrypt (Certbot)**:

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Monitoring & Maintenance

### Application Monitoring

**Using Prometheus + Grafana**:

```yaml
# docker-compose.yml additions
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**`prometheus.yml`**:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'neuroquant'
    static_configs:
      - targets: ['backend:8000']
```

### Logging

**Centralized logging with ELK stack**:

```yaml
# docker-compose.yml additions
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
```

### Health Checks

**Endpoint**:

```python
# backend/main.py
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "redis": "connected" if redis_client else "disabled"
    }
```

**Monitoring script**:

```bash
#!/bin/bash
# monitor.sh

while true; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    
    if [ $response -eq 200 ]; then
        echo "$(date): Service healthy"
    else
        echo "$(date): Service unhealthy (HTTP $response)"
        # Restart service
        docker-compose restart backend
    fi
    
    sleep 60
done
```

### Backup Strategy

**Database backup**:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup PostgreSQL
docker-compose exec -T postgres pg_dump -U neuroquant trading > $BACKUP_DIR/db_$DATE.sql

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $DATE"
```

**Cron job**:

```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1
```

---

## Security Best Practices

### 1. Environment Variables

- Never commit `.env` files
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Rotate credentials regularly

### 2. Network Security

- Use VPC/private subnets
- Configure security groups properly
- Enable WAF (Web Application Firewall)

### 3. Container Security

- Use non-root user in containers
- Scan images for vulnerabilities
- Keep base images updated

### 4. Authentication

- Implement JWT authentication
- Use HTTPS only
- Enable rate limiting

### 5. Database Security

- Use strong passwords
- Enable SSL connections
- Regular backups
- Restrict network access

---

## Troubleshooting

### Common Issues

**Container won't start**:
```bash
# Check logs
docker-compose logs backend

# Inspect container
docker inspect neuroquant-backend

# Verify ports
netstat -tulpn | grep 8000
```

**Database connection errors**:
```bash
# Test connection
docker-compose exec backend python -c "from database.database import engine; print(engine.connect())"
```

**High memory usage**:
```bash
# Check resource usage
docker stats

# Limit resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

**Slow performance**:
- Enable Redis caching
- Optimize database queries
- Scale horizontally

---

## See Also

- [Configuration Guide](CONFIGURATION.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Getting Started](GETTING_STARTED.md)
- [Testing Guide](TESTING.md)
