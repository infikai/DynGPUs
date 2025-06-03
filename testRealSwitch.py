import time
import os
import signal
import multiprocessing
import threading
import subprocess # For running Docker command
import shutil # For removing checkpoint directory if needed (optional)

# Attempt to import PyTorch, provide guidance if not found
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch is not installed. The 'train' function will be a placeholder.")
    print("Please install PyTorch and torchvision with: pip install torch torchvision")


# --- User-defined functions ---

# Placeholder for PyTorch Dataset if PyTorch is available
if PYTORCH_AVAILABLE:
    class DummyImageDataset(Dataset):
        """A dummy dataset that generates random images and labels."""
        def __init__(self, num_samples=1000, transform=None):
            self.num_samples = num_samples
            self.transform = transform
            # Simulate 224x224 RGB images
            self.data_shape = (3, 224, 224) 
            self.num_classes = 1000 # Like ImageNet

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Generate a random image tensor
            image = torch.randn(self.data_shape, dtype=torch.float32)
            # Generate a random label
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label

CHECKPOINT_DIR = "./checkpoints"

def save_checkpoint(epoch, model, optimizer, loss, filename_prefix="checkpoint"):
    """Saves model checkpoint and measures time taken."""
    if not PYTORCH_AVAILABLE:
        return
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"[TRAIN PID {os.getpid()}] Created checkpoint directory: {CHECKPOINT_DIR}")

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Save epoch-specific checkpoint
    epoch_filename = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch_{epoch}.pth")
    start_time_epoch_save = time.time()
    torch.save(state, epoch_filename)
    end_time_epoch_save = time.time()
    print(f"[TRAIN PID {os.getpid()}] Saved epoch checkpoint to {epoch_filename} (took {end_time_epoch_save - start_time_epoch_save:.2f}s)")

    # Save latest checkpoint (overwrites previous latest)
    latest_filename = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_time_latest_save = time.time()
    torch.save(state, latest_filename)
    end_time_latest_save = time.time()
    print(f"[TRAIN PID {os.getpid()}] Updated latest checkpoint to {latest_filename} (took {end_time_latest_save - start_time_latest_save:.2f}s)")


def load_checkpoint(model, optimizer, filename="latest_checkpoint.pth"):
    """Loads model checkpoint."""
    if not PYTORCH_AVAILABLE:
        return 0, None # start_epoch, last_loss
    
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"[TRAIN PID {os.getpid()}] Loading checkpoint '{filepath}'")
        try:
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage) # Handles CPU/GPU load
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
            last_loss = checkpoint.get('loss', None) # .get for backward compatibility
            print(f"[TRAIN PID {os.getpid()}] Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']}, loss: {last_loss})")
            print(f"[TRAIN PID {os.getpid()}] Resuming training from epoch {start_epoch}")
            return start_epoch, last_loss
        except Exception as e:
            print(f"[TRAIN PID {os.getpid()}] Error loading checkpoint {filepath}: {e}. Starting from scratch.")
            return 0, None
    else:
        print(f"[TRAIN PID {os.getpid()}] No checkpoint found at '{filepath}'. Starting from scratch.")
        return 0, None


def train():
    """
    PyTorch training function for ResNet50 with dummy ImageNet data.
    Includes checkpointing.
    """
    if not PYTORCH_AVAILABLE:
        print(f"[TRAIN PID {os.getpid()}] PyTorch not available. Running placeholder training.")
        try:
            while True:
                print(f"[TRAIN PID {os.getpid()}] Placeholder training batch processing...")
                time.sleep(5)
        except KeyboardInterrupt:
            print(f"[TRAIN PID {os.getpid()}] Placeholder training received KeyboardInterrupt. Exiting.")
        finally:
            print(f"[TRAIN PID {os.getpid()}] Placeholder training finished.")
        return

    print(f"[TRAIN PID {os.getpid()}] PyTorch Training started with ResNet50 on dummy data.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN PID {os.getpid()}] Using device: {device}")

    model = models.resnet50(weights=None) 
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    start_epoch, _ = load_checkpoint(model, optimizer) 
    
    model.train() 

    dummy_dataset = DummyImageDataset(num_samples=1280) 
    train_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=True, num_workers=2 if device.type == 'cuda' else 0) 

    max_epochs = 100 
    running = True 
    current_epoch_loss = 0.0
    current_epoch_batches = 0
    epoch_for_interrupt_save = start_epoch # Initialize with start_epoch

    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_for_interrupt_save = epoch # Keep track of current epoch for interrupt save
            if not running: break
            print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1}/{max_epochs}")
            epoch_loss_aggregator = 0.0
            num_batches_in_epoch = 0

            for i, (inputs, labels) in enumerate(train_loader):
                if not running: break 
                
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss_aggregator += loss.item()
                num_batches_in_epoch += 1
                current_epoch_loss = epoch_loss_aggregator 
                current_epoch_batches = num_batches_in_epoch

                if (i + 1) % 10 == 0: 
                    print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            if not running: break 

            avg_epoch_loss = epoch_loss_aggregator / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
            print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
            
            save_checkpoint(epoch, model, optimizer, avg_epoch_loss)
            time.sleep(1) 
        
        if running: 
             print(f"[TRAIN PID {os.getpid()}] Training completed {max_epochs} epochs (or ran out of specified range).")

    except KeyboardInterrupt:
        print(f"[TRAIN PID {os.getpid()}] Training function received KeyboardInterrupt. Saving final checkpoint...")
        running = False 
        final_loss_to_save = current_epoch_loss / current_epoch_batches if current_epoch_batches > 0 else None
        
        print(f"[TRAIN PID {os.getpid()}] Attempting to save interrupt checkpoint for epoch {epoch_for_interrupt_save}...")
        save_checkpoint(epoch_for_interrupt_save, model, optimizer, final_loss_to_save, filename_prefix="interrupt_checkpoint")
        print(f"[TRAIN PID {os.getpid()}] Training function cleanup complete after interrupt. Exiting.")
    except Exception as e:
        print(f"[TRAIN PID {os.getpid()}] Error in PyTorch training: {e}")
        try:
            print(f"[TRAIN PID {os.getpid()}] Attempting to save crash checkpoint for epoch {epoch_for_interrupt_save}...")
            save_checkpoint(epoch_for_interrupt_save, model, optimizer, None, filename_prefix="crash_checkpoint")
        except Exception as ce:
            print(f"[TRAIN PID {os.getpid()}] Could not save crash checkpoint: {ce}")
    finally:
        print(f"[TRAIN PID {os.getpid()}] PyTorch Training function finished.")


def serving():
    """
    Starts the NVIDIA Triton Inference Server using Docker.
    Manages the Docker subprocess.
    """
    print(f"[SERVING PID {os.getpid()}] Serving function started. Attempting to launch Triton Docker container...")
    
    docker_command_str = (
        "docker run --gpus=1 --rm --net=host "
        "-v /mydata/Data/server/docs/examples/model_repository:/models " 
        "nvcr.io/nvidia/tritonserver:25.02-py3 " 
        "tritonserver --model-repository=/models --model-control-mode explicit --load-model densenet_onnx"
    )
    docker_command_list = docker_command_str.split()
    docker_process = None
    
    try:
        print(f"[SERVING PID {os.getpid()}] Executing Docker command: {' '.join(docker_command_list)}")
        docker_process = subprocess.Popen(docker_command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[SERVING PID {os.getpid()}] Triton Docker container started (Process PID: {docker_process.pid}). Monitoring...")

        while docker_process.poll() is None: 
            time.sleep(1) 
        print(f"[SERVING PID {os.getpid()}] Docker process (PID: {docker_process.pid}) has exited with code {docker_process.returncode}.")

    except FileNotFoundError:
        print(f"[SERVING PID {os.getpid()}] Error: 'docker' command not found. Is Docker installed and in PATH?")
    except Exception as e:
        print(f"[SERVING PID {os.getpid()}] Error in serving function (launching/managing Docker): {e}")
    finally:
        if docker_process:
            if docker_process.poll() is None: 
                print(f"[SERVING PID {os.getpid()}] Attempting to stop Docker container (Process PID: {docker_process.pid})...")
                docker_process.terminate()  
                try:
                    docker_process.wait(timeout=15)  
                    print(f"[SERVING PID {os.getpid()}] Docker container terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print(f"[SERVING PID {os.getpid()}] Docker container (Process PID: {docker_process.pid}) did not terminate in time. Killing...")
                    docker_process.kill()  
                    try:
                        docker_process.wait(timeout=5) 
                        print(f"[SERVING PID {os.getpid()}] Docker container killed.")
                    except subprocess.TimeoutExpired:
                        print(f"[SERVING PID {os.getpid()}] Docker container (Process PID: {docker_process.pid}) failed to be killed.")
                except Exception as e_term:
                    print(f"[SERVING PID {os.getpid()}] Exception during Docker process termination: {e_term}")
            else:
                print(f"[SERVING PID {os.getpid()}] Docker process (Process PID: {docker_process.pid}) already exited with code {docker_process.returncode}.")
        else:
            print(f"[SERVING PID {os.getpid()}] Docker process was not started.")
        print(f"[SERVING PID {os.getpid()}] Serving function finished.")


# --- Global state and configuration ---
Load = 0
Max_Load = 50  

train_process = None
serving_process = None
process_management_lock = threading.Lock()
stop_monitor_event = threading.Event()

# --- Worker entry points for multiprocessing ---
def train_worker_entry():
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) starting train().")
    train()
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) exiting.")

def serving_worker_entry():
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) starting serving().")
    serving()
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) exiting.")

# --- Core logic: Process Management ---
def manage_processes():
    global train_process, serving_process, Load, Max_Load

    with process_management_lock:
        train_is_effectively_running = train_process is not None and train_process.is_alive()
        serving_is_effectively_running = serving_process is not None and serving_process.is_alive()

        if Load > Max_Load: 
            if train_is_effectively_running:
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Attempting to shut down training (PID: {train_process.pid}).")
                try:
                    os.kill(train_process.pid, signal.SIGINT) 
                    # Increased timeout for checkpoint saving
                    train_process.join(timeout=300) 
                    if train_process.is_alive():
                        print(f"MONITOR: Training (PID: {train_process.pid}) did not stop after SIGINT (300s). Forcing kill.")
                        train_process.kill()  
                        train_process.join(timeout=5)
                except ProcessLookupError: 
                    print(f"MONITOR: Training process (PID: {train_process.pid}) not found.")
                except Exception as e:
                    print(f"MONITOR: Error stopping training process (PID: {train_process.pid if train_process else 'N/A'}): {e}")
                finally:
                    if train_process and not train_process.is_alive(): 
                        train_process.close() 
                        train_process = None
                        print("MONITOR: Training process confirmed shut down.")
                    elif train_process: 
                         print(f"MONITOR: Training process (PID: {train_process.pid}) still alive after shutdown attempts.")

            serving_is_effectively_running = serving_process is not None and serving_process.is_alive() 
            if not serving_is_effectively_running:
                if serving_process and not serving_process.is_alive(): 
                    serving_process.close()
                    serving_process = None 
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Starting serving...")
                serving_process = multiprocessing.Process(target=serving_worker_entry, daemon=True)
                serving_process.start()
                print(f"MONITOR: Serving process started (PID: {serving_process.pid}).")

        elif Load <= Max_Load: 
            if serving_is_effectively_running:
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Attempting to shut down serving (PID: {serving_process.pid}).")
                try:
                    serving_process.terminate()  
                    serving_process.join(timeout=20) 
                    if serving_process.is_alive():
                        print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) did not stop. Forcing kill.")
                        serving_process.kill()
                        serving_process.join(timeout=10)
                except ProcessLookupError:
                     print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) not found.")
                except Exception as e:
                    print(f"MONITOR: Error stopping serving Python process (PID: {serving_process.pid if serving_process else 'N/A'}): {e}")
                finally:
                    if serving_process and not serving_process.is_alive():
                        serving_process.close()
                        serving_process = None
                        print("MONITOR: Serving Python process confirmed shut down.")
                    elif serving_process:
                        print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) still alive.")
            
            train_is_effectively_running = train_process is not None and train_process.is_alive() 
            if not train_is_effectively_running:
                if train_process and not train_process.is_alive(): 
                    train_process.close()
                    train_process = None
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Starting training...")
                train_process = multiprocessing.Process(target=train_worker_entry, daemon=True)
                train_process.start()
                print(f"MONITOR: Training process started (PID: {train_process.pid}).")

def monitor_thread_worker():
    print("MONITOR: Monitor thread started. Will check load every 2 seconds.")
    while not stop_monitor_event.is_set():
        manage_processes()
        time.sleep(2) 
    print("MONITOR: Monitor thread stopped.")

def cleanup_all_processes():
    print("MAIN: Initiating cleanup of all child processes...")
    global train_process, serving_process
    
    with process_management_lock: 
        if train_process and train_process.is_alive():
            print(f"MAIN: Cleaning up training process (PID: {train_process.pid})...")
            try:
                os.kill(train_process.pid, signal.SIGINT) 
                # Increased timeout for checkpoint saving during final cleanup
                train_process.join(timeout=30) 
                if train_process.is_alive(): 
                    print(f"MAIN: Training (PID: {train_process.pid}) still alive after SIGINT (30s). Killing.")
                    train_process.kill()
                    train_process.join(timeout=5) 
            except Exception as e:
                print(f"MAIN: Error cleaning up training process: {e}")
            finally:
                if train_process: train_process.close() 
                train_process = None

        if serving_process and serving_process.is_alive():
            print(f"MAIN: Cleaning up serving Python process (PID: {serving_process.pid})...")
            try:
                serving_process.terminate() 
                serving_process.join(timeout=20) 
                if serving_process.is_alive(): 
                    print(f"MAIN: Serving Python process (PID: {serving_process.pid}) still alive. Killing.")
                    serving_process.kill()
                    serving_process.join(timeout=10) 
            except Exception as e:
                print(f"MAIN: Error cleaning up serving Python process: {e}")
            finally:
                if serving_process: serving_process.close() 
                serving_process = None
    print("MAIN: Cleanup finished.")

# --- Main execution ---
if __name__ == "__main__":
    multiprocessing.freeze_support() 

    print(f"MAIN: Program started. Default Max_Load is {Max_Load}. Initial Load is {Load}.")
    if not PYTORCH_AVAILABLE:
        print("MAIN: Note - PyTorch is not installed, training will use a placeholder.")
    print("MAIN: Ensure Docker is running and you have permissions if serving is activated.")
    print("MAIN: Ensure model repository path for Triton is correct (e.g., /mydata/Data/server/docs/examples/model_repository)")
    print(f"MAIN: Checkpoints will be saved in: {os.path.abspath(CHECKPOINT_DIR)}")
    print("MAIN: Enter an integer value for 'Load' to change system behavior.")
    print("MAIN: Type 'q' or 'quit' to exit.")

    monitor = threading.Thread(target=monitor_thread_worker, daemon=True)
    monitor.start()

    try:
        while True:
            try:
                user_input = input(f"Enter new Load (current: {Load}, max: {Max_Load}) (or 'q' to quit): ")
                if user_input.lower() in ['q', 'quit']:
                    print("MAIN: Quit command received. Shutting down...")
                    break
                
                new_load = int(user_input)
                Load = new_load 
                print(f"MAIN: Load updated to {Load}. Monitor will adjust processes shortly.")

            except ValueError:
                print("MAIN: Invalid input. Please enter an integer for Load or 'q' to quit.")
            except EOFError: 
                print("MAIN: EOF detected, exiting...")
                break
    
    except KeyboardInterrupt:
        print("MAIN: KeyboardInterrupt received in main loop. Shutting down...")
    finally:
        print("MAIN: Starting final shutdown sequence...")
        stop_monitor_event.set() 

        if monitor.is_alive():
            print("MAIN: Waiting for monitor thread to finish...")
            monitor.join(timeout=5) 
            if monitor.is_alive():
                print("MAIN: Monitor thread did not stop in time.")
        
        cleanup_all_processes() 
        
        print("MAIN: Program terminated.")