# Deployment Guide

This guide covers production deployment strategies for the Document Ingestion API.

## Overview

The Document Ingestion API can be deployed in several configurations:
- **Single Server**: All components on one machine
- **Distributed**: API server and workers on separate machines
- **Containerized**: Using Docker and Docker Compose
- **Cloud**: AWS, GCP, Azure deployment options

## Production Requirements

### Hardware Requirements

**Minimum Configuration:**
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB SSD (models + processing space)
- **Network**: 1Gbps connection

**Recommended Configuration:**
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ 
- **Storage**: 200GB+ NVMe SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional)
- **Network**: 10Gbps connection

### Software Requirements

- **OS**: Ubuntu 20.04+, CentOS 8+, or RHEL 8+
- **Python**: 3.8-3.11 (3.11 recommended)
- **Redis**: 6.0+
- **Reverse Proxy**: Nginx or Apache
- **Process Manager**: systemd, supervisord, or Docker

## Docker Deployment

### Single Container Setup

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models

# Install application
RUN pip install -e .

# Expose port
EXPOSE 8000

# Start script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

**docker-entrypoint.sh:**
```bash
#!/bin/bash
set -e

# Start Redis in background
redis-server --daemonize yes

# Download models if not present
python -c "
import os
if not os.path.exists('models/ds4sd--docling-models'):
    from pipeline import DocumentIntelligencePipeline
    DocumentIntelligencePipeline()
"

# Start application
exec python main.py
```

**Build and run:**
```bash
docker build -t doc-ingestion .
docker run -p 8000:8000 -v $(pwd)/models:/app/models doc-ingestion
```

### Docker Compose Setup

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  api:
    build: .
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/9
      - CELERY_RESULT_BACKEND=redis://redis:6379/10
      - START_CELERY_WORKER=false
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis

  worker:
    build: .
    restart: unless-stopped
    command: celery -A celery_app worker --loglevel=info --concurrency=2
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/9
      - CELERY_RESULT_BACKEND=redis://redis:6379/10
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api

volumes:
  redis_data:
```

**Start services:**
```bash
docker-compose up -d
```

## Native Deployment

### System Service Setup

**Create systemd service files:**

**`/etc/systemd/system/doc-ingestion-api.service`:**
```ini
[Unit]
Description=Document Ingestion API
After=network.target redis.service
Requires=redis.service

[Service]
Type=exec
User=doc-ingestion
Group=doc-ingestion
WorkingDirectory=/opt/doc-ingestion
Environment=PATH=/opt/doc-ingestion/venv/bin
ExecStart=/opt/doc-ingestion/venv/bin/python main.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**`/etc/systemd/system/doc-ingestion-worker.service`:**
```ini
[Unit]
Description=Document Ingestion Celery Worker
After=network.target redis.service doc-ingestion-api.service
Requires=redis.service

[Service]
Type=exec
User=doc-ingestion
Group=doc-ingestion
WorkingDirectory=/opt/doc-ingestion
Environment=PATH=/opt/doc-ingestion/venv/bin
ExecStart=/opt/doc-ingestion/venv/bin/celery -A celery_app worker --loglevel=info --concurrency=4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable doc-ingestion-api doc-ingestion-worker
sudo systemctl start doc-ingestion-api doc-ingestion-worker
```

### Installation Script

**`deploy.sh`:**
```bash
#!/bin/bash
set -e

# Configuration
APP_DIR="/opt/doc-ingestion"
APP_USER="doc-ingestion"
PYTHON_VERSION="3.11"

# Create user
sudo useradd -r -s /bin/bash -d $APP_DIR $APP_USER

# Create directories
sudo mkdir -p $APP_DIR/{models,logs}
sudo chown -R $APP_USER:$APP_USER $APP_DIR

# Install system dependencies
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    redis-server \
    tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng \
    nginx \
    git

# Switch to app user
sudo -u $APP_USER bash << 'EOF'
cd /opt/doc-ingestion

# Clone repository
git clone <repository-url> .

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install gunicorn

# Download models
python -c "from pipeline import DocumentIntelligencePipeline; DocumentIntelligencePipeline()"

# Set permissions
chmod +x main.py
EOF

# Install systemd services
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable redis doc-ingestion-api doc-ingestion-worker
sudo systemctl start redis doc-ingestion-api doc-ingestion-worker

echo "✅ Deployment completed successfully"
```

## Reverse Proxy Configuration

### Nginx Configuration

**`/etc/nginx/sites-available/doc-ingestion`:**
```nginx
upstream doc_ingestion_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Request limits
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    # Proxy settings
    location / {
        proxy_pass http://doc_ingestion_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # 5 minutes for long processing
    }

    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://doc_ingestion_api;
        access_log off;
    }

    # Static files (if any)
    location /static/ {
        alias /opt/doc-ingestion/static/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }
}
```

**Enable the site:**
```bash
sudo ln -s /etc/nginx/sites-available/doc-ingestion /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Apache Configuration

**`/etc/apache2/sites-available/doc-ingestion.conf`:**
```apache
<VirtualHost *:80>
    ServerName your-domain.com
    Redirect permanent / https://your-domain.com/
</VirtualHost>

<VirtualHost *:443>
    ServerName your-domain.com

    # SSL configuration
    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/cert.pem
    SSLCertificateKeyFile /etc/ssl/private/key.pem

    # Security headers
    Header always set X-Content-Type-Options nosniff
    Header always set X-Frame-Options DENY
    Header always set X-XSS-Protection "1; mode=block"
    Header always set Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"

    # Proxy configuration
    ProxyPreserveHost On
    ProxyRequests Off
    
    ProxyPass / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/
    
    # Timeout settings
    ProxyTimeout 300
    Timeout 300
</VirtualHost>
```

## Load Balancing

### Multiple API Servers

**Nginx upstream configuration:**
```nginx
upstream doc_ingestion_api {
    server 127.0.0.1:8000 weight=1;
    server 127.0.0.1:8001 weight=1;
    server 127.0.0.1:8002 weight=1;
    
    # Health checks
    keepalive 32;
}
```

**Run multiple API servers:**
```bash
# Server 1
SERVER_PORT=8000 gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app

# Server 2
SERVER_PORT=8001 gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app

# Server 3
SERVER_PORT=8002 gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Celery Worker Scaling

**Multiple worker nodes:**
```bash
# On worker node 1
celery -A celery_app worker --hostname=worker-1@%h --concurrency=4

# On worker node 2  
celery -A celery_app worker --hostname=worker-2@%h --concurrency=4

# On worker node 3
celery -A celery_app worker --hostname=worker-3@%h --concurrency=4
```

## Cloud Deployment

### AWS Deployment

**EC2 Instance Setup:**
```bash
# Instance type: c5.xlarge or higher
# Storage: 100GB+ EBS GP3
# Security groups: HTTP (80), HTTPS (443), SSH (22)

# Install dependencies
sudo yum update -y
sudo yum install -y python3.11 redis git nginx

# Deploy application
curl -O https://raw.githubusercontent.com/your-org/doc-ingestion/main/deploy.sh
chmod +x deploy.sh
sudo ./deploy.sh
```

**RDS for Redis (ElastiCache):**
```bash
# Update connection string
export CELERY_BROKER_URL="redis://your-redis-cluster.cache.amazonaws.com:6379/9"
export CELERY_RESULT_BACKEND="redis://your-redis-cluster.cache.amazonaws.com:6379/10"
```

**Application Load Balancer:**
```yaml
# ALB configuration
Targets:
  - Instance: i-1234567890abcdef0 (port 8000)
  - Instance: i-abcdef1234567890 (port 8000)
Health Check:
  Path: /health
  Interval: 30s
  Timeout: 10s
```

### Docker Swarm

**docker-compose.prod.yml:**
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  api:
    image: your-registry/doc-ingestion:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
    ports:
      - "8000:8000"

  worker:
    image: your-registry/doc-ingestion:latest
    command: celery -A celery_app worker --loglevel=info
    deploy:
      replicas: 6
      update_config:
        parallelism: 2
        delay: 10s
```

**Deploy to swarm:**
```bash
docker stack deploy -c docker-compose.prod.yml doc-ingestion
```

### Kubernetes

**kubernetes/deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-ingestion-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: doc-ingestion-api
  template:
    metadata:
      labels:
        app: doc-ingestion-api
    spec:
      containers:
      - name: api
        image: your-registry/doc-ingestion:latest
        ports:
        - containerPort: 8000
        env:
        - name: CELERY_BROKER_URL
          value: "redis://redis:6379/9"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: doc-ingestion-api-service
spec:
  selector:
    app: doc-ingestion-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring and Logging

### Application Monitoring

**Prometheus metrics:**
```python
# Add to requirements.txt
prometheus_client==0.16.0
prometheus_fastapi_instrumentator==6.0.0

# In app.py
from prometheus_fastapi_instrumentator import Instrumentator

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
```

**Health check endpoint monitoring:**
```bash
# Add to crontab
*/5 * * * * curl -f http://localhost:8000/health || echo "API health check failed" | mail admin@company.com
```

### Log Management

**Centralized logging with rsyslog:**
```bash
# /etc/rsyslog.d/doc-ingestion.conf
$ModLoad imfile

# API logs
$InputFileName /opt/doc-ingestion/logs/app.log
$InputFileTag doc-ingestion-api:
$InputFileStateFile stat-doc-ingestion-api
$InputFileSeverity info
$InputFileFacility local7
$InputRunFileMonitor

# Send to log server
*.* @@log-server:514
```

**Log rotation:**
```bash
# /etc/logrotate.d/doc-ingestion
/opt/doc-ingestion/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 0644 doc-ingestion doc-ingestion
    postrotate
        systemctl reload doc-ingestion-api doc-ingestion-worker
    endscript
}
```

## Security Considerations

### Network Security

```bash
# Firewall rules (ufw)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

# Restrict Redis access
sudo ufw deny 6379/tcp
```

### Application Security

```python
# Add to requirements.txt
python-multipart==0.0.6  # For file upload security

# In config.py
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit
ALLOWED_HOSTS = ["your-domain.com", "api.your-domain.com"]
```

**Security headers (already in Nginx config above):**
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security

### SSL/TLS Certificate

**Using Let's Encrypt:**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
sudo certbot renew --dry-run
```

## Backup and Recovery

### Database Backup (Redis)

```bash
# Automated Redis backup
cat > /opt/scripts/backup-redis.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
redis-cli BGSAVE
sleep 10
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis-backup-$DATE.rdb

# Keep only last 7 days
find $BACKUP_DIR -name "redis-backup-*.rdb" -mtime +7 -delete
EOF

chmod +x /opt/scripts/backup-redis.sh

# Add to crontab
0 2 * * * /opt/scripts/backup-redis.sh
```

### Application Backup

```bash
# Backup models and configuration
tar -czf doc-ingestion-backup-$(date +%Y%m%d).tar.gz \
    /opt/doc-ingestion/models \
    /opt/doc-ingestion/.env \
    /opt/doc-ingestion/logs
```

## Performance Tuning

### System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Kernel parameters
echo "net.core.somaxconn = 1024" >> /etc/sysctl.conf
echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
sysctl -p
```

### Application Optimization

```env
# Production environment settings
SERVER_WORKERS=4
CELERY_WORKER_CONCURRENCY=4
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
DEVICE=cuda  # If GPU available
```

For more information, see:
- [Setup Guide](SETUP.md) - Installation and configuration
- [Development Guide](DEVELOPMENT.md) - Development workflow
- [Architecture Guide](ARCHITECTURE.md) - System design overview