#!/bin/bash

# SolarAI Backend Startup Script
# This script starts all backend services using Docker Compose

set -e  # Exit on any error

echo "🚀 Starting SolarAI Backend Services..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check if docker-compose.backend.yml exists
if [ ! -f "docker-compose.backend.yml" ]; then
    echo "❌ docker-compose.backend.yml not found in project root"
    exit 1
fi

# Stop any existing services
echo "🛑 Stopping any existing services..."
docker-compose -f docker-compose.backend.yml down --remove-orphans

# Start services
echo "🔧 Starting backend services..."
docker-compose -f docker-compose.backend.yml up -d --build

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."

# Wait for PostgreSQL
echo "   Waiting for PostgreSQL..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if docker-compose -f docker-compose.backend.yml exec -T postgres pg_isready -U solarai -d solarai > /dev/null 2>&1; then
        echo "   ✅ PostgreSQL is ready"
        break
    fi
    sleep 1
    counter=$((counter + 1))
done

if [ $counter -eq $timeout ]; then
    echo "   ❌ PostgreSQL failed to start within $timeout seconds"
    exit 1
fi

# Wait for API Gateway
echo "   Waiting for API Gateway..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✅ API Gateway is ready"
        break
    fi
    sleep 2
    counter=$((counter + 1))
done

if [ $counter -eq $timeout ]; then
    echo "   ❌ API Gateway failed to start within $timeout seconds"
    exit 1
fi

# Final health check
echo ""
echo "🔍 Performing final health check..."

if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API Gateway health check passed"
else
    echo "⚠️  API Gateway health check failed"
fi

echo ""
echo "🎉 Backend services started successfully!"
echo ""
echo "📊 Service URLs:"
echo "   API Gateway:     http://localhost:8000"
echo "   API Docs:        http://localhost:8000/docs"
echo "   Health Check:    http://localhost:8000/health"
echo "   PostgreSQL:      localhost:5432"
echo "   pgAdmin:         http://localhost:8081"
echo "   Redis:           localhost:6379"
echo ""
echo "🔧 Useful commands:"
echo "   View logs:       docker-compose -f docker-compose.backend.yml logs -f"
echo "   Stop services:   docker-compose -f docker-compose.backend.yml down"
echo "   Restart:         docker-compose -f docker-compose.backend.yml restart"
echo ""
echo "📝 Default credentials:"
echo "   PostgreSQL:      solarai/solarai123"
echo "   pgAdmin:         admin@solarai.com/admin123"
echo ""
echo "🗄️  Database Management:"
echo "   Connect to DB:   docker exec -it solarai_postgres psql -U solarai -d solarai"
echo "   View tables:     \dt"
echo "   View data:       SELECT * FROM users;"
echo "" pip