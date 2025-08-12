# Ubuntu Setup Guide for PDF Processing System

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ available space
- **GPU**: Optional (for image processing acceleration)

### Software Requirements
- Ubuntu 20.04 LTS or later
- Python 3.8+
- Docker (optional, for containerized deployment)

## Installation Steps

### 1. System Updates
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3-dev python3-pip python3-venv
sudo apt install -y libpq-dev libssl-dev libffi-dev
sudo apt install -y poppler-utils tesseract-ocr
sudo apt install -y libmagickwand-dev imagemagick
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv kb_env
source kb_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies
```bash
# Install system dependencies for PDF processing
sudo apt install -y libmupdf-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libfreetype6-dev liblcms2-dev

# Install Python packages
pip install -r requirements.txt
```

### 4. Directory Structure Setup
```bash
# Create application directories
sudo mkdir -p /opt/kb/{uploads,output,logs,images}
sudo chown -R $USER:$USER /opt/kb
sudo chmod -R 755 /opt/kb

# Create symbolic links for easier access
ln -s /opt/kb ~/kb_data
```

### 5. Environment Configuration
```bash
# Create environment file
cat > .env << EOF
# Application Settings
APP_ENV=production
LOG_LEVEL=INFO
DEBUG=false

# File Storage
UPLOAD_DIR=/opt/kb/uploads
OUTPUT_DIR=/opt/kb/output
MAX_FILE_SIZE=52428800

# Vector Database (ChromaDB)
USE_PINECONE=false
VECTOR_STORE_TYPE=chromadb
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000
OPENAI_TEMPERATURE=0.1

# Security
SECRET_KEY=your_secret_key_here
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]

# Performance
WORKER_PROCESSES=4
MAX_CONCURRENT_REQUESTS=100
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EOF
```

### 6. Database Setup (ChromaDB)
```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server (for production)
chroma run --host 0.0.0.0 --port 8000 --path /opt/kb/chromadb
```

### 7. Systemd Service Setup
```bash
# Create systemd service file
sudo tee /etc/systemd/system/kb-api.service > /dev/null << EOF
[Unit]
Description=KB PDF Processing API
After=network.target

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=/home/$USER/Desktop/KB/kb
Environment=PATH=/home/$USER/Desktop/KB/kb_env/bin
ExecStart=/home/$USER/Desktop/KB/kb_env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable kb-api
sudo systemctl start kb-api
```

### 8. Nginx Configuration (Optional)
```bash
# Install Nginx
sudo apt install -y nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/kb-api << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /images/ {
        alias /opt/kb/images/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/kb-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 9. Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
```

### 10. Monitoring Setup
```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Create log rotation
sudo tee /etc/logrotate.d/kb-api << EOF
/home/$USER/Desktop/KB/kb/app.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload kb-api
    endscript
}
EOF
```

## Performance Optimization

### 1. System Tuning
```bash
# Optimize file system
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_ratio=15' | sudo tee -a /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Optimize disk I/O
echo 'ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="deadline"' | sudo tee /etc/udev/rules.d/60-scheduler.rules
```

### 2. Memory Optimization
```bash
# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Process Limits
```bash
# Increase file descriptor limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
```

## Security Hardening

### 1. User Security
```bash
# Create dedicated user
sudo adduser kbuser
sudo usermod -aG sudo kbuser
sudo passwd kbuser

# Secure SSH
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

### 2. Application Security
```bash
# Set proper permissions
sudo chown -R kbuser:kbuser /opt/kb
sudo chmod -R 750 /opt/kb
sudo chmod 600 /opt/kb/.env

# Enable AppArmor
sudo apt install -y apparmor apparmor-utils
sudo systemctl enable apparmor
sudo systemctl start apparmor
```

## Monitoring & Maintenance

### 1. Health Checks
```bash
# Create health check script
cat > /opt/kb/health_check.sh << 'EOF'
#!/bin/bash
API_URL="http://localhost:8000/health/"
LOG_FILE="/opt/kb/logs/health.log"

response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)
if [ $response -ne 200 ]; then
    echo "$(date): Health check failed - HTTP $response" >> $LOG_FILE
    systemctl restart kb-api
else
    echo "$(date): Health check passed" >> $LOG_FILE
fi
EOF

chmod +x /opt/kb/health_check.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/kb/health_check.sh") | crontab -
```

### 2. Backup Strategy
```bash
# Create backup script
cat > /opt/kb/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/kb/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup uploads
tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz -C /opt/kb uploads/

# Backup images
tar -czf $BACKUP_DIR/images_$DATE.tar.gz -C /opt/kb images/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz -C /opt/kb logs/

# Clean old backups (keep 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF

chmod +x /opt/kb/backup.sh

# Add to crontab (daily backup at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/kb/backup.sh") | crontab -
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
```bash
sudo chown -R $USER:$USER /opt/kb
sudo chmod -R 755 /opt/kb
```

2. **Memory Issues**
```bash
# Check memory usage
free -h
# Check swap usage
swapon --show
```

3. **Disk Space Issues**
```bash
# Check disk usage
df -h
# Clean old files
sudo find /opt/kb -name "*.log" -mtime +30 -delete
```

4. **Service Issues**
```bash
# Check service status
sudo systemctl status kb-api
# View logs
sudo journalctl -u kb-api -f
```

## Performance Monitoring

### 1. System Metrics
```bash
# Install monitoring tools
sudo apt install -y prometheus node-exporter grafana

# Create monitoring dashboard
# (Detailed Grafana setup instructions would go here)
```

### 2. Application Metrics
```bash
# Add Prometheus metrics to the application
pip install prometheus-client

# Create metrics endpoint
# (Implementation details in main.py)
```

## Scaling Considerations

### 1. Horizontal Scaling
- Use load balancer (HAProxy/Nginx)
- Multiple API instances
- Shared storage (NFS/S3)

### 2. Vertical Scaling
- Increase CPU cores
- Add more RAM
- Use SSD storage
- GPU acceleration for image processing

### 3. Database Scaling
- ChromaDB clustering
- Redis caching
- Database sharding

## Maintenance Schedule

### Daily
- Monitor system resources
- Check application logs
- Verify backup completion

### Weekly
- Update system packages
- Review performance metrics
- Clean old temporary files

### Monthly
- Security updates
- Performance optimization
- Capacity planning
- Disaster recovery testing

