# Production Deployment Guide

## 🚀 Enterprise Deployment

This guide covers deploying NeuroQuant to production environments including Docker, Kubernetes, AWS, GCP, and Azure.

---

## Quick Deployment Options

### Option 1: Docker Single Container (Small Scale)

```bash
# Build image
docker build -t neuroquant:latest .

# Run container
docker run -d \
  --name neuroquant \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@db:5432/neuroquant" \
  -e REDIS_URL="redis://cache:6379" \
  -v /data/models:/app/models \
  -v /data/logs:/app/logs \
  neuroquant:latest

# Check logs
docker logs -f neuroquant
```

### Option 2: Docker Compose (Production Stack)

```bash
# Start full stack with database, cache, monitoring
docker-compose -f docker-compose.prod.yml up -d

# Monitor services
docker-compose logs -f

# Scale services
docker-compose up -d --scale backend=3
```

### Option 3: Kubernetes (Enterprise Scale)

```bash
# Install NeuroQuant Helm chart
helm repo add neuroquant https://charts.neuroquant.com
helm repo update

# Deploy to Kubernetes
helm install neuroquant neuroquant/neuroquant \
  --namespace trading \
  --create-namespace \
  --values values.yaml

# Monitor deployment
kubectl get pods -n trading
kubectl logs -f deployment/neuroquant-backend -n trading
```

---

## Environment Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose (for containerized deployment)
- Kubernetes cluster (for K8s deployment)

### Environment Variables

```bash
# .env.production
ENVIRONMENT=production
DEBUG=False

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:password@db-host:5432/neuroquant
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Cache
REDIS_URL=redis://cache-host:6379/0
REDIS_PASSWORD=your-secure-password

# Security
SECRET_KEY=your-very-secure-random-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Trading APIs
ALPHAVANTAGE_KEY=your-api-key
NEWS_API_KEY=your-api-key
BROKER_API_KEY=your-api-key

# Monitoring
SENTRY_DSN=https://your-sentry-url
LOG_LEVEL=INFO

# Email (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# AWS (if using AWS)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET=neuroquant-data
```

---

## Database Setup

### PostgreSQL Configuration

```bash
# Connect to PostgreSQL
psql -U postgres -h localhost

# Create database
CREATE DATABASE neuroquant;

# Create user
CREATE USER neuroquant_user WITH PASSWORD 'secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE neuroquant TO neuroquant_user;

# Create extensions
\c neuroquant
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Initialize Database

```bash
# Run migrations
python -m alembic upgrade head

# Seed initial data
python -m scripts.seed_data

# Create indices for performance
python -m scripts.create_indices
```

---

## Performance Optimization

### Database Tuning

```sql
-- Increase shared buffers
ALTER SYSTEM SET shared_buffers = '4GB';

-- Optimize for production
ALTER SYSTEM SET work_mem = '20MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET effective_cache_size = '12GB';

-- Enable parallel queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Checkpoint optimization
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Apply changes
SELECT pg_reload_conf();
```

### Caching Strategy

```python
# Redis configuration for caching
from redis import Redis

cache = Redis(
    host='cache-host',
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
    health_check_interval=30
)

# Cache trading data
cache.setex(f"backtest:{agent_id}:{symbol}", 3600, json.dumps(results))
```

### Load Balancing

```yaml
# Nginx configuration
upstream neuroquant_backend {
    server backend1:8000 weight=1;
    server backend2:8000 weight=1;
    server backend3:8000 weight=1;
    keepalive 32;
}

server {
    listen 80;
    server_name neuroquant.com;

    location / {
        proxy_pass http://neuroquant_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files
    location /static/ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

---

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
trades_total = Counter('trades_total', 'Total trades executed')
trade_duration = Histogram('trade_duration_seconds', 'Trade execution time')
portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value')
backtest_p_and_l = Gauge('backtest_pnl', 'Backtest P&L')

# Use metrics
trades_total.inc()
trade_duration.observe(execution_time)
portfolio_value.set(current_value)
```

### ELK Stack Logging

```python
# Structured logging configuration
import logging
from pythonjsonlogger import jsonlogger

handler = logging.FileHandler('logs/app.json')
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)

# Log trades
logger.info("Trade executed", extra={
    "symbol": "AAPL",
    "action": "buy",
    "price": 150.25,
    "quantity": 10,
    "timestamp": datetime.now().isoformat()
})
```

### Alerting

```yaml
# Prometheus alert rules
groups:
  - name: neuroquant
    rules:
      - alert: HighDrawdown
        expr: drawdown_percent > 20
        for: 5m
        annotations:
          summary: "Portfolio drawdown exceeds 20%"

      - alert: APILatencyHigh
        expr: http_request_duration_seconds{quantile="0.95"} > 1
        for: 5m
        annotations:
          summary: "API p95 latency exceeds 1 second"

      - alert: DatabaseConnectionPoolExhausted
        expr: db_connections > 15
        for: 2m
        annotations:
          summary: "Database connection pool nearly exhausted"
```

---

## Security Hardening

### SSL/TLS Configuration

```yaml
# Generate SSL certificates
certbot certonly --standalone -d neuroquant.com

# Configure in Nginx
ssl_certificate /etc/letsencrypt/live/neuroquant.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/neuroquant.com/privkey.pem;
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers HIGH:!aNULL:!MD5;
ssl_prefer_server_ciphers on;
```

### API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthCredential
from jose import JWTError, jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredential = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/trades", dependencies=[Depends(limiter.limit("10/minute"))])
async def execute_trade(trade: TradeRequest):
    # Trade execution
    pass
```

---

## Backup & Disaster Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh - Daily backup script

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/neuroquant"

# Backup PostgreSQL
pg_dump \
  postgresql://user:pass@localhost/neuroquant \
  | gzip > $BACKUP_DIR/db_$TIMESTAMP.sql.gz

# Backup models and data
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz /app/models
tar -czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz /app/logs

# Upload to S3
aws s3 sync $BACKUP_DIR s3://neuroquant-backups/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -mtime +30 -delete
```

### Point-in-Time Recovery

```bash
# Restore from backup
pg_restore -d neuroquant -U neuroquant_user /backups/neuroquant/db_20240101_000000.sql.gz

# Restore model files
tar -xzf /backups/neuroquant/models_20240101_000000.tar.gz -C /app/
```

---

## Scaling Strategy

### Horizontal Scaling

```yaml
# Kubernetes StatefulSet for scalable deployment
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neuroquant-backend
spec:
  serviceName: neuroquant-backend
  replicas: 3
  selector:
    matchLabels:
      app: neuroquant-backend
  template:
    metadata:
      labels:
        app: neuroquant-backend
    spec:
      containers:
      - name: backend
        image: neuroquant:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Post-Deployment Verification

```bash
# Check health
curl http://localhost:8000/health

# Check API documentation
curl http://localhost:8000/docs

# Verify database connection
python -c "from database.database import test_connection; test_connection()"

# Test trading functionality
python -m pytest tests/integration/test_trading.py -v

# Performance benchmark
python -m pytest tests/benchmarks/ --benchmark-only
```

---

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL
psql -U neuroquant_user -h localhost -d neuroquant -c "SELECT 1;"

# Check firewall
netstat -an | grep 5432
```

**High API Latency**
```bash
# Check CPU usage
top -p $(pgrep -f "uvicorn")

# Check database slow queries
SELECT query, calls, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```

**Memory Leaks**
```bash
# Monitor memory
docker stats neuroquant

# Check for memory leaks
python -m memory_profiler run backend/main.py
```

---

## Support & Maintenance

- **Documentation**: https://docs.neuroquant.com
- **Issues**: https://github.com/MDhruv03/NeuroQuant/issues
- **Email**: support@neuroquant.com
- **Slack**: [Join Community](https://slack.neuroquant.com)

---

**Last Updated**: October 2025
