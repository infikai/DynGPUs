#!/bin/bash

# 1. Activate the virtual environment
source ~/Kevin/venv/vllm/bin/activate

# 2. List the folders to process
FOLDERS=(
    "/home/pacs/Kevin/DynGPUs/OnlineSearch_test/sys_v100_online_24"
    "/home/pacs/Kevin/DynGPUs/OnlineSearch_test/sys_v100_online_22"
    "/home/pacs/Kevin/DynGPUs/OnlineSearch_test/sys_v100_online_20"
    "/home/pacs/Kevin/DynGPUs/OnlineSearch_test/sys_v100_online_18"
    "/home/pacs/Kevin/DynGPUs/OnlineSearch_test/sys_v100_online_16"
)

for DIR in "${FOLDERS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "========================================"
        echo "Starting Folder: $DIR"
        echo "========================================"
        
        cd "$DIR" || continue

        # 1. Start the Benchmark in the BACKGROUND
        echo "[$(date +%T)] Starting Benchmark (PID: $!)..."
        python benchmark_api_trace.py \
            --port 8888 \
            --model-name microsoft/Phi-3.5-mini-instruct \
            --trace-file ~/Kevin/Trace/Azure_preprocessed.jsonl &
        
        # 2. Delay 10 seconds to allow the server to bind to port 8888
        sleep 10

        # 3. Start the Autoscaler in the FOREGROUND with 91m timeout
        echo "[$(date +%T)] Starting Autoscaler (Timeout: 91m)..."
        timeout 91m python autoscaler.py
        
        # 4. Wait for the benchmark to finish naturally
        echo "[$(date +%T)] Autoscaler finished. Waiting for Benchmark to exit..."
        wait
        
        echo "[$(date +%T)] Finished $DIR. Moving to next directory."
        cd ..
    else
        echo "Warning: $DIR not found. Skipping."
    fi
done

echo "All folders processed."
