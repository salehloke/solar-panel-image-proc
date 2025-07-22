#!/bin/bash

# SolarAI Backend - Volume Management Script
# This script helps manage Docker volumes for models and logs

set -e

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

# Function to show volume info
show_volume_info() {
    local volume_name=$1
    if docker volume ls | grep -q "$volume_name"; then
        print_success "Volume $volume_name exists"
        print_status "Location: $(docker volume inspect $volume_name | grep Mountpoint | cut -d'"' -f4)"
        print_status "Size: $(docker run --rm -v $volume_name:/data alpine du -sh /data | cut -f1)"
    else
        print_warning "Volume $volume_name does not exist"
    fi
}

# Function to backup volume
backup_volume() {
    local volume_name=$1
    local backup_dir="./backups"
    
    if docker volume ls | grep -q "$volume_name"; then
        print_status "Creating backup of $volume_name..."
        mkdir -p "$backup_dir"
        
        docker run --rm -v $volume_name:/data -v $(pwd)/$backup_dir:/backup alpine tar czf /backup/${volume_name}_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .
        print_success "Backup created: $backup_dir/${volume_name}_$(date +%Y%m%d_%H%M%S).tar.gz"
    else
        print_warning "Volume $volume_name does not exist, nothing to backup"
    fi
}

# Function to restore volume
restore_volume() {
    local volume_name=$1
    local backup_file=$2
    
    if [ -z "$backup_file" ]; then
        print_error "Please specify a backup file"
        exit 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file $backup_file does not exist"
        exit 1
    fi
    
    print_status "Restoring $volume_name from $backup_file..."
    
    # Remove existing volume if it exists
    if docker volume ls | grep -q "$volume_name"; then
        docker volume rm $volume_name
    fi
    
    # Create new volume and restore
    docker volume create $volume_name
    docker run --rm -v $volume_name:/data -v $(pwd)/$(dirname $backup_file):/backup alpine tar xzf /backup/$(basename $backup_file) -C /data
    print_success "Volume $volume_name restored from $backup_file"
}

# Main script logic
case "${1:-help}" in
    "info")
        echo "📊 SolarAI Volume Information"
        echo "============================="
        echo ""
        show_volume_info "solar-image-processing_model_storage"
        show_volume_info "solar-image-processing_log_storage"
        show_volume_info "solar-image-processing_torch_cache"
        show_volume_info "solar-image-processing_postgres_data"
        show_volume_info "solar-image-processing_redis_data"
        ;;
    "backup")
        echo "💾 Creating Volume Backups"
        echo "=========================="
        echo ""
        backup_volume "solar-image-processing_model_storage"
        backup_volume "solar-image-processing_log_storage"
        backup_volume "solar-image-processing_torch_cache"
        ;;
    "restore")
        echo "🔄 Restoring Volume from Backup"
        echo "==============================="
        echo ""
        if [ -z "$2" ]; then
            print_error "Usage: $0 restore <backup_file>"
            exit 1
        fi
        restore_volume "solar-image-processing_model_storage" "$2"
        ;;
    "clean")
        echo "🧹 Cleaning Volumes"
        echo "==================="
        echo ""
        read -p "This will delete all data in the volumes. Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume rm solar-image-processing_model_storage 2>/dev/null || print_warning "Model storage volume not found"
            docker volume rm solar-image-processing_log_storage 2>/dev/null || print_warning "Log storage volume not found"
            docker volume rm solar-image-processing_torch_cache 2>/dev/null || print_warning "Torch cache volume not found"
            print_success "Volumes cleaned"
        else
            print_status "Operation cancelled"
        fi
        ;;
    "help"|*)
        echo "🛠️  SolarAI Volume Management"
        echo "============================="
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  info                    Show volume information and sizes"
        echo "  backup                  Create backups of all volumes"
        echo "  restore <backup_file>   Restore model storage from backup"
        echo "  clean                   Remove all volumes (DESTRUCTIVE)"
        echo "  help                    Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 info"
        echo "  $0 backup"
        echo "  $0 restore ./backups/solar-image-processing_model_storage_20250101_120000.tar.gz"
        echo "  $0 clean"
        ;;
esac 