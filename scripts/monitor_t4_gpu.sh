#!/bin/bash
# Real-time T4 GPU Monitoring for Inference Workloads
# Usage: ./scripts/monitor_t4_gpu.sh

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}          T4 GPU Real-Time Monitoring Dashboard              ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

monitor_gpu() {
    while true; do
        print_header
        
        # GPU Info
        echo -e "${GREEN}GPU Information:${NC}"
        nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit \
            --format=csv,noheader | awk -F', ' '{
                printf "  Name:        %s\n", $1
                printf "  Driver:      %s\n", $2
                printf "  Temperature: %s°C\n", $3
                printf "  Power:       %s / %s\n", $4, $5
            }'
        echo ""
        
        # Memory Usage
        echo -e "${GREEN}Memory Usage:${NC}"
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
            --format=csv,noheader,nounits | awk -F', ' '{
                total=$1; used=$2; free=$3
                usage_pct=int((used/total)*100)
                
                printf "  Total: %d MB\n", total
                printf "  Used:  %d MB (", used
                
                # Color code usage
                if (usage_pct > 90) printf "\033[0;31m%d%%\033[0m", usage_pct
                else if (usage_pct > 70) printf "\033[1;33m%d%%\033[0m", usage_pct
                else printf "\033[0;32m%d%%\033[0m", usage_pct
                
                printf ")\n"
                printf "  Free:  %d MB\n", free
                
                # Progress bar
                printf "  ["
                bars=int(usage_pct/2)
                for(i=1; i<=50; i++) {
                    if(i<=bars) printf "█"
                    else printf "░"
                }
                printf "]\n"
            }'
        echo ""
        
        # GPU Utilization
        echo -e "${GREEN}GPU Utilization:${NC}"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory \
            --format=csv,noheader,nounits | awk -F', ' '{
                gpu_util=$1; mem_util=$2
                
                printf "  GPU:    "
                if (gpu_util > 90) printf "\033[0;31m%d%%\033[0m", gpu_util
                else if (gpu_util > 50) printf "\033[1;33m%d%%\033[0m", gpu_util
                else printf "\033[0;32m%d%%\033[0m", gpu_util
                printf " ["
                bars=int(gpu_util/2)
                for(i=1; i<=50; i++) {
                    if(i<=bars) printf "▓"
                    else printf "░"
                }
                printf "]\n"
                
                printf "  Memory: "
                if (mem_util > 90) printf "\033[0;31m%d%%\033[0m", mem_util
                else if (mem_util > 50) printf "\033[1;33m%d%%\033[0m", mem_util
                else printf "\033[0;32m%d%%\033[0m", mem_util
                printf " ["
                bars=int(mem_util/2)
                for(i=1; i<=50; i++) {
                    if(i<=bars) printf "▓"
                    else printf "░"
                }
                printf "]\n"
            }'
        echo ""
        
        # Running Processes
        echo -e "${GREEN}GPU Processes:${NC}"
        PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null)
        
        if [ -z "$PROCESSES" ]; then
            echo "  No active GPU processes"
        else
            echo "$PROCESSES" | while IFS=', ' read -r pid name mem; do
                printf "  PID %s: %s (%s MB)\n" "$pid" "$name" "$mem"
            done
        fi
        echo ""
        
        # Performance Stats
        echo -e "${GREEN}Performance Stats:${NC}"
        nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory \
            --format=csv,noheader,nounits | awk -F', ' '{
                printf "  GPU Clock:    %d MHz / %d MHz\n", $1, $3
                printf "  Memory Clock: %d MHz / %d MHz\n", $2, $4
            }'
        echo ""
        
        echo -e "${BLUE}Press Ctrl+C to exit | Refreshing every 2 seconds${NC}"
        
        sleep 2
    done
}

# Trap Ctrl+C
trap 'echo ""; echo "Monitoring stopped."; exit 0' INT

monitor_gpu
