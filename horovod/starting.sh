#!/bin/bash

# --- Start background command ---
nvidia-smi -q -d UTILIZATION,MEMORY -lms 10 -f gpu_usage.txt &
NVIDIA_SMI_PID=$!
echo "nvidia-smi started in the background with PID: $NVIDIA_SMI_PID"

echo "Launching training on 1 GPU in the background..."
horovodrun -np 3 --min-np 2 --max-np 3 --network-interface eth0 --host-discovery-script ./discover_hosts.sh --verbose python train_elastic_measurement.py --model resnet50 --batch-size 128 > r50b128 2>&1 &
TRAINING_PID=$(pgrep -f "train_elastic_measurement.py" | head -n 1)
echo "PID: $TRAINING_PID"
sleep 70

echo "Killing nvidia-smi process (PID: $NVIDIA_SMI_PID)..."
kill $NVIDIA_SMI_PID
echo "nvidia-smi process killed."