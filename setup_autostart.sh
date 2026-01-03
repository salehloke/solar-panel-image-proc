#!/bin/bash

# ==========================================
# Solar Panel Project - Auto-Start Setup Script
# Run this on your Raspberry Pi ONE TIME.
# ==========================================

echo "🚀 Setting up Auto-Start services..."

# 1. Create Backend Service
echo "📝 Creating Backend Service..."
sudo bash -c 'cat > /etc/systemd/system/solar-backend.service << EOF
[Unit]
Description=Solar Panel Backend API
After=network.target

[Service]
User=ikmal
WorkingDirectory=/home/ikmal/Developer/solar-panel-image-proc/backend_edge
ExecStart=/home/ikmal/Developer/solar-panel-image-proc/backend_edge/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
Environment=PATH=/home/ikmal/Developer/solar-panel-image-proc/backend_edge/venv/bin:/usr/bin:/usr/local/bin

[Install]
WantedBy=multi-user.target
EOF'

# 2. Create Frontend Service
echo "📝 Creating Frontend Service..."
sudo bash -c 'cat > /etc/systemd/system/solar-frontend.service << EOF
[Unit]
Description=Solar Panel Dashboard (Frontend)
After=network.target solar-backend.service

[Service]
User=ikmal
WorkingDirectory=/home/ikmal/Developer/solar-panel-image-proc/frontend
# We use 'npm start' for production, or 'npm run dev' for development.
# 'npm start' is better for auto-start but requires 'npm run build' first.
ExecStart=/usr/bin/npm run dev
Restart=always
RestartSec=10
Environment=NODE_ENV=development
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOF'

# 3. Reload Systemd and Enable Services
echo "🔄 Reloading Systemd..."
sudo systemctl daemon-reload

echo "✅ Enabling Services (Start on Boot)..."
sudo systemctl enable solar-backend.service
sudo systemctl enable solar-frontend.service

echo "▶️ Starting Services NOW..."
sudo systemctl start solar-backend.service
sudo systemctl start solar-frontend.service

echo "🎉 Done! Check status with: sudo systemctl status solar-backend"
