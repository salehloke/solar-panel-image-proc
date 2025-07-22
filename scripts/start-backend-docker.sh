#!/bin/bash

# SolarAI Backend - Docker Mode
# This script starts the complete backend stack in Docker

set -e  # Exit on any error

echo "🚀 Starting SolarAI Backend in Docker Mode..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if docker-compose.backend.yml exists
if [ ! -f "docker-compose.backend.yml" ]; then
    print_error "docker-compose.backend.yml not found. Please run this script from the project root."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p models logs

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f docker-compose.backend.yml down 2>/dev/null || true

# Build and start the services
print_status "Building and starting services..."
docker-compose -f docker-compose.backend.yml up --build -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check if services are running
print_status "Checking service status..."

# Check PostgreSQL
if docker exec solarai_postgres pg_isready -U solarai > /dev/null 2>&1; then
    print_success "PostgreSQL is ready"
else
    print_warning "PostgreSQL might still be starting up..."
fi

# Check Redis
if docker exec solarai_redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis might still be starting up..."
fi

# Check API Gateway
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "API Gateway is ready"
else
    print_warning "API Gateway might still be starting up..."
fi

# Show logs
print_status "Showing recent logs..."
docker logs solarai_api_gateway --tail 20

print_success "Backend services started!"
echo ""
echo "📋 Service URLs:"
echo "  • API Gateway:     http://localhost:8000"
echo "  • API Docs:        http://localhost:8000/docs"
echo "  • Health Check:    http://localhost:8000/health"
echo "  • pgAdmin:         http://localhost:8081"
echo "  • PostgreSQL:      localhost:5432"
echo "  • Redis:           localhost:6379"
echo ""
echo "📝 Useful commands:"
echo "  • View logs:       docker logs -f solarai_api_gateway"
echo "  • Stop services:   docker-compose -f docker-compose.backend.yml down"
echo "  • Restart API:     docker-compose -f docker-compose.backend.yml restart api-gateway"
echo ""
print_success "SolarAI Backend is ready for use! 🎉" 