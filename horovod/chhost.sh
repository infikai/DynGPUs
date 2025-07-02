#!/bin/bash

# --- Start background command ---
nvidia-smi -q -d UTILIZATION,MEMORY -lms 10 -f gpu_usage.txt &
NVIDIA_SMI_PID=$!
echo "nvidia-smi started in the background with PID: $NVIDIA_SMI_PID"

# --- Wait 3 seconds and delete line ---
sleep 20
echo "Deleting 'node2:1' from host file..."
sed -i '/node2:1/d' host
echo "Done deleting."

# --- Wait 5 seconds and add line ---
sleep 30
echo "Adding 'node2:1' to host file..."
echo "node2:1" >> host
echo "Done adding."

# --- Wait 30 seconds and delete line, then kill nvidia-smi ---
sleep 70
echo "Deleting 'node2:1' from host file again..."
sed -i '/node2:1/d' host
echo "Done deleting."
sleep 30

echo "Killing nvidia-smi process (PID: $NVIDIA_SMI_PID)..."
kill $NVIDIA_SMI_PID
echo "nvidia-smi process killed."

echo "node2:1" >> host

echo "Script finished."