#!/bin/bash

# SolarAI - Stop All Services
# This script stops all services related to the SolarAI project
# Including Docker containers, local processes, and frees up all used ports

set -e  # Exit on any error

echo "🛑 Stopping ALL SolarAI Services..."

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

# Function to check if a port is in use
is_port_in_use() {
    lsof -i ":$1" &> /dev/null
    return $?
}

# Function to kill process using a specific port
kill_process_on_port() {
    local port=$1
    print_status "Checking for processes using port $port..."
    
    # Get PID of process using the port
    local pid=$(lsof -t -i :"$port" 2>/dev/null)
    
    if [ -n "$pid" ]; then
        print_status "Found process (PID: $pid) using port $port. Attempting to terminate..."
        kill -15 "$pid" 2>/dev/null || true
        sleep 1
        
        # Check if process is still running
        if kill -0 "$pid" 2>/dev/null; then
            print_warning "Process still running. Attempting to force kill..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        
        # Final check
        if ! is_port_in_use "$port"; then
            print_success "Port $port is now free"
        else
            print_error "Failed to free port $port. You may need to investigate manually."
        fi
    else
        print_status "No process found using port $port"
    fi
}

# Stop Docker containers
print_status "Stopping Docker containers..."

# Stop any containers from docker-compose
if [ -f "docker-compose.backend.yml" ]; then
    print_status "Stopping containers from docker-compose.backend.yml..."
    docker-compose -f docker-compose.backend.yml down --remove-orphans 2>/dev/null || print_warning "Failed to stop docker-compose services (they may not be running)"
fi

if [ -f "docker-compose.yml" ]; then
    print_status "Stopping containers from docker-compose.yml..."
    docker-compose -f docker-compose.yml down --remove-orphans 2>/dev/null || print_warning "Failed to stop docker-compose services (they may not be running)"
fi

# Stop specific containers by name
containers=("solarai_postgres" "solarai_redis" "solarai_api_gateway" "solarai_pgadmin" "local-postgres" "local-redis")

for container in "${containers[@]}"; do
    if docker ps -q --filter "name=$container" | grep -q .; then
        print_status "Stopping container: $container..."
        docker stop "$container" 2>/dev/null || print_warning "Failed to stop $container"
        
        # Ask if container should be removed
        read -p "Do you want to remove the $container container? (y/N): " remove_container
        if [[ "$remove_container" =~ ^[Yy]$ ]]; then
            docker rm "$container" 2>/dev/null || print_warning "Failed to remove $container"
            print_success "Container $container removed"
        else
            print_status "Container $container stopped but not removed"
        fi
    else
        print_status "Container $container is not running"
    fi
done

# Kill processes on commonly used ports
print_status "Freeing up commonly used ports..."
ports=(8000 8001 5432 6379 8081)

for port in "${ports[@]}"; do
    if is_port_in_use "$port"; then
        kill_process_on_port "$port"
    else
        print_status "Port $port is not in use"
    fi
done

# Check for any Python processes related to uvicorn/FastAPI
print_status "Checking for Python/uvicorn processes..."
pids=$(ps aux | grep "[u]vicorn\|[a]pp.main" | awk '{print $2}')

if [ -n "$pids" ]; then
    print_status "Found uvicorn/FastAPI processes. Attempting to terminate..."
    for pid in $pids; do
        kill -15 "$pid" 2>/dev/null || true
    done
    print_success "Terminated uvicorn/FastAPI processes"
else
    print_status "No uvicorn/FastAPI processes found"
fi

# Deactivate virtual environment if active
if [ -n "$VIRTUAL_ENV" ]; then
    print_status "Deactivating virtual environment..."
    deactivate 2>/dev/null || true
fi

# Final status check
print_status "Checking final status of key ports..."
all_clear=true

for port in "${ports[@]}"; do
    if is_port_in_use "$port"; then
        print_warning "Port $port is still in use"
        all_clear=false
    else
        print_success "Port $port is free"
    fi
done

if [ "$all_clear" = true ]; then
    print_success "All services have been successfully stopped!"
    print_status "You can now start the system in any mode without conflicts."
else
    print_warning "Some services may still be running. Manual intervention might be needed."
    print_status "Check running processes with: docker ps"
    print_status "Check used ports with: lsof -i :<port_number>"
fi

exit 0
