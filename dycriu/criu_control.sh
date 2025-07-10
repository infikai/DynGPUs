#!/bin/bash

# ==============================================================================
# This script launches a training process on the GPU, switches it to the CPU,
# and then uses CRIU to time the dump and restore operations.
#
# PREREQUISITES:
# 1. Horovod is installed.
# 2. CRIU is installed and configured (requires sudo/root).
# 3. The Python script 'dynamic_train.py' is in this directory.
# ==============================================================================

# --- Configuration ---
PYTHON_SCRIPT="dynamic_train.py"
CONTROL_FILE="device.txt"
LOG_FILE="training_output.log"
DUMP_DIR="criu_dump"

# --- 1. Initial Cleanup & Setup ---
echo "Cleaning up previous runs..."
rm -f $CONTROL_FILE $LOG_FILE
rm -rf $DUMP_DIR
mkdir -p $DUMP_DIR
echo "Setup complete."
echo "----------------------------------------"


# --- 2. Start Training on GPU ---
echo "Setting initial state to GPU in $CONTROL_FILE"
echo "gpu" > $CONTROL_FILE

echo "Launching training on 1 GPU in the background..."
horovodrun -np 1 python $PYTHON_SCRIPT --epochs 10 > $LOG_FILE 2>&1 &

sleep 5
# --- MODIFIED LINE: Ensure we only get the first PID ---
TRAINING_PID=$(pgrep -f "$PYTHON_SCRIPT" | head -n 1)

if [ -z "$TRAINING_PID" ]; then
    echo "ERROR: Could not find the training process PID."
    exit 1
fi
echo "Training started on GPU with PID: $TRAINING_PID"
echo "----------------------------------------"


# --- 3. Switch to CPU ---
echo "Running on GPU for 30 seconds..."
sleep 30

echo "Switching to CPU by updating $CONTROL_FILE"
echo "cpu" > $CONTROL_FILE
echo "Waiting 10 seconds for the process to stabilize on CPU..."
sleep 20
echo "----------------------------------------"


# --- 4. Dump the Process with CRIU ---
echo ">>> DUMPING a CPU process (PID: $TRAINING_PID) using CRIU..."
echo "This will terminate the process."

time sudo criu dump -t $TRAINING_PID --images-dir $DUMP_DIR -j --shmem-unmap -v4 -o dump.log

if [ $? -ne 0 ]; then
    echo "ERROR: CRIU dump failed. Check dump.log for details."
    exit 1
fi
echo "Process dumped successfully. It is no longer running."
echo "----------------------------------------"


# --- 5. Restore the Process with CRIU ---
echo "Waiting 5 seconds before restoring..."
sleep 5

echo ">>> RESTORING the process from the dump..."
time sudo criu restore --images-dir $DUMP_DIR -d -j -v4 -o restore.log

if [ $? -ne 0 ]; then
    echo "ERROR: CRIU restore failed. Check restore.log for details."
    exit 1
fi
echo "----------------------------------------"


# --- 6. Finalization ---
RESTORED_PID=$(pgrep -f "$PYTHON_SCRIPT" | head -n 1)
echo "✅ Process restored successfully!"
echo "New PID is $RESTORED_PID."
echo "Training is now continuing on the CPU."
echo ""
echo "You can monitor its progress with:"
echo "tail -f $LOG_FILE"
echo ""
echo "To stop the restored process, run:"
echo "kill $RESTORED_PID"