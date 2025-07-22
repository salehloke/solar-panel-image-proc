#!/bin/bash

# SolarAI Backend Status Checker
# This script shows the status of both Docker and Local backend modes

set -e

echo "🔍 SolarAI Backend Status Check"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo "🐳 Docker Backend (Port 8000):"
echo "-------------------------------"

# Check Docker backend
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "✅ Running - http://localhost:8000"
    print_status "API Docs: http://localhost:8000/docs"
    
    # Get health details
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo "{}")
    if command -v jq > /dev/null 2>&1; then
        MODEL_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.model // "unknown"')
        DB_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.database // "unknown"')
        print_status "Model: $MODEL_STATUS"
        print_status "Database: $DB_STATUS"
    fi
else
    print_error "❌ Not running"
fi

echo ""
echo "💻 Local Backend (Port 8001):"
echo "------------------------------"

# Check Local backend
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    print_success "✅ Running - http://localhost:8001"
    print_status "API Docs: http://localhost:8001/docs"
    
    # Get health details
    HEALTH_RESPONSE=$(curl -s http://localhost:8001/health 2>/dev/null || echo "{}")
    if command -v jq > /dev/null 2>&1; then
        MODEL_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.model // "unknown"')
        DB_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.database // "unknown"')
        print_status "Model: $MODEL_STATUS"
        print_status "Database: $DB_STATUS"
    fi
else
    print_warning "⚠️  Not running"
fi

echo ""
echo "🗄️  Database Services:"
echo "----------------------"

# Check PostgreSQL
if docker ps | grep -q "postgres"; then
    print_success "✅ PostgreSQL: Running"
    if docker ps | grep -q "solarai_postgres"; then
        print_status "Container: solarai_postgres (Port 5432)"
    elif docker ps | grep -q "local-postgres"; then
        print_status "Container: local-postgres (Port 5432)"
    fi
else
    print_error "❌ PostgreSQL: Not running"
fi

# Check Redis
if docker ps | grep -q "redis"; then
    print_success "✅ Redis: Running"
    if docker ps | grep -q "solarai_redis"; then
        print_status "Container: solarai_redis (Port 6379)"
    elif docker ps | grep -q "local-redis"; then
        print_status "Container: local-redis (Port 6379)"
    fi
else
    print_error "❌ Redis: Not running"
fi

echo ""
echo "🌐 Frontend:"
echo "------------"
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "✅ Running - http://localhost:3000"
else
    print_warning "⚠️  Not running"
fi

echo ""
echo "📋 Quick Commands:"
echo "------------------"
echo "• Start Docker Backend: ./scripts/start-backend-docker.sh"
echo "• Start Local Backend:  ./scripts/start-backend-local.sh"
echo "• Stop Local Backend:   ./scripts/stop-backend-hybrid.sh"
echo "• View Docker Logs:     docker logs -f solarai_api_gateway"
echo "• Stop All Docker:      docker-compose -f docker-compose.backend.yml down" 