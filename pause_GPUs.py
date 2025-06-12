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
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder # For loading ImageNet
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch is not installed. The 'train' function will be a placeholder or will fail if real data loading is expected.")
    print("Please install PyTorch and torchvision with: pip install torch torchvision")
    # Define dummy classes if PyTorch is not available to prevent NameErrors later
    class Dataset: pass
    class ImageFolder(Dataset): pass


# --- Multi-GPU Configuration ---
# Define the GPU device IDs to be used for training and serving.
# Example: Train on GPUs 0, 1, 2 and serve on GPU 3.
# Ensure these IDs are valid on your machine (e.g., check with `nvidia-smi`).
TRAIN_GPU_IDS = [0, 1, 2, 3]
SERVE_GPU_ID = 0
DOCKER_CONTAINER_NAME = "triton_server_instance_pm" # Unique name for the container
IMAGENET_DATA_PATH = "/mydata/Data/imagenet" # User-specified path
CHECKPOINT_DIR = "./checkpoints"

# --- DDP Setup and Cleanup ---
def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


# --- User-defined functions ---
def save_checkpoint(rank, epoch, model, optimizer, loss, filename_prefix="checkpoint"):
    """Saves model checkpoint. Only rank 0 should save."""
    if not PYTORCH_AVAILABLE or rank != 0:
        return

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"[TRAIN PID {os.getpid()} RANK {rank}] Created checkpoint directory: {CHECKPOINT_DIR}")

    # To save a DDP model, we need to access model.module
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    epoch_filename = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch_{epoch}.pth")
    start_time_epoch_save = time.time()
    torch.save(state, epoch_filename)
    end_time_epoch_save = time.time()
    print(f"[TRAIN PID {os.getpid()} RANK {rank}] Saved epoch checkpoint to {epoch_filename} (took {end_time_epoch_save - start_time_epoch_save:.2f}s)")

    latest_filename = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_time_latest_save = time.time()
    torch.save(state, latest_filename)
    end_time_latest_save = time.time()
    print(f"[TRAIN PID {os.getpid()} RANK {rank}] Updated latest checkpoint to {latest_filename} (took {end_time_latest_save - start_time_latest_save:.2f}s)")


def load_checkpoint(rank, model, optimizer, filename="latest_checkpoint.pth"):
    """Loads model checkpoint onto the correct device for the DDP process."""
    if not PYTORCH_AVAILABLE:
        return 0, None

    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        # Map location to the process's assigned GPU
        map_location = f'cuda:{rank}'
        if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Loading checkpoint '{filepath}'")
        try:
            checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            last_loss = checkpoint.get('loss', None)
            if rank == 0:
                print(f"[TRAIN PID {os.getpid()} RANK {rank}] Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']}, loss: {last_loss})")
                print(f"[TRAIN PID {os.getpid()} RANK {rank}] Resuming training from epoch {start_epoch}")
            return start_epoch, last_loss
        except Exception as e:
            if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Error loading checkpoint {filepath}: {e}. Starting from scratch.")
            return 0, None
    else:
        if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] No checkpoint found at '{filepath}'. Starting from scratch.")
        return 0, None


def ddp_train_worker(rank, world_size, st):
    """
    The actual training function that will be run on each GPU process using DDP.
    """
    print(f"[TRAIN PID {os.getpid()} RANK {rank}] DDP worker started for GPU {TRAIN_GPU_IDS[rank]}.")
    setup_ddp(rank, world_size)
    
    start_time_training_init = time.time()

    # Model setup
    model = models.resnet50(weights=None).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(rank)

    # Load checkpoint
    start_epoch = 0
    if rank == 0 and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # All processes load the checkpoint to sync model and optimizer states
    start_time_loadCP = time.time()
    start_epoch, _ = load_checkpoint(rank, ddp_model, optimizer)
    end_time_loadCP = time.time()
    if rank == 0: print(f"Loading checkpoint took {end_time_loadCP-start_time_loadCP}s")

    # Data loading setup
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dir = os.path.join(IMAGENET_DATA_PATH, 'train')
    if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Attempting to load ImageNet training data from: {train_dir}")
    
    try:
        train_dataset = ImageFolder(train_dir, train_transform)
        # Use DistributedSampler to partition the data
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        if len(train_dataset) == 0: raise ValueError("Dataset is empty.")
    except (FileNotFoundError, ValueError) as e:
        if rank == 0:
            print(f"[TRAIN PID {os.getpid()} RANK {rank}] ERROR: ImageNet data issue at {train_dir}: {e}. Please check the path and contents.")
        cleanup_ddp()
        return

    if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Successfully loaded {len(train_dataset)} total images for training.")

    num_dataloader_workers = 12 # Adjust per GPU
    batch_size = 256 # Per-GPU batch size
    if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] DataLoader using num_workers={num_dataloader_workers}, batch_size={batch_size} per GPU")
    
    start_time_data_loader = time.time()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, # Sampler handles shuffling
        num_workers=num_dataloader_workers, 
        pin_memory=True,
        sampler=train_sampler
    )
    end_time_data_loader = time.time()
    if rank == 0: print(f"Dataloader job took {end_time_data_loader - start_time_data_loader:.2f}s")
    
    max_epochs = 100
    epoch_for_interrupt_save = start_epoch

    end_time_training_init = time.time()
    if rank == 0: print(f"Fully starting training job took {end_time_training_init - start_time_training_init:.2f}s")
    
    try:
        ddp_model.train()
        for epoch in range(start_epoch, max_epochs):
            epoch_for_interrupt_save = epoch
            train_sampler.set_epoch(epoch) # Important for shuffling
            
            if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Epoch {epoch+1}/{max_epochs}")
            
            epoch_loss_aggregator = 0.0
            num_batches_in_epoch = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(rank), labels.to(rank)

                if start_epoch == epoch and i == 0 and rank == 0:
                    END_TIME_I2T = time.time()
                    print(f"Fully I2T took {END_TIME_I2T - st:.2f}s")

                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss_aggregator += loss.item()
                num_batches_in_epoch += 1

                if (i + 1) % 100 == 0 and rank == 0:
                    print(f"[TRAIN PID {os.getpid()} RANK {rank}] Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss_aggregator / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
            if rank == 0:
                print(f"[TRAIN PID {os.getpid()} RANK {rank}] Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
                save_checkpoint(rank, epoch, ddp_model, optimizer, avg_epoch_loss)
        
        if rank == 0:
             print(f"[TRAIN PID {os.getpid()} RANK {rank}] Training completed {max_epochs} epochs.")

    except KeyboardInterrupt:
        if rank == 0: print(f"[TRAIN PID {os.getpid()} RANK {rank}] Training function received KeyboardInterrupt. Rank 0 will save final checkpoint...")
    except Exception as e:
        print(f"[TRAIN PID {os.getpid()} RANK {rank}] Error in PyTorch training: {e}")
    finally:
        # Final checkpoint save on graceful exit or error (only by rank 0)
        if rank == 0:
            print(f"[TRAIN PID {os.getpid()} RANK {rank}] Attempting to save final checkpoint for epoch {epoch_for_interrupt_save}...")
            save_checkpoint(rank, epoch_for_interrupt_save, ddp_model, optimizer, None, filename_prefix="final_checkpoint")
        
        print(f"[TRAIN PID {os.getpid()} RANK {rank}] DDP worker finished.")
        cleanup_ddp()

def train(st):
    """
    Launcher for the multi-GPU DDP training process.
    """
    if not PYTORCH_AVAILABLE:
        print(f"[TRAIN LAUNCHER PID {os.getpid()}] PyTorch not available. Cannot start DDP training.")
        return

    world_size = len(TRAIN_GPU_IDS)
    if world_size == 0:
        print(f"[TRAIN LAUNCHER PID {os.getpid()}] No training GPUs specified in TRAIN_GPU_IDS. Exiting.")
        return
        
    if torch.cuda.device_count() < world_size:
        print(f"[TRAIN LAUNCHER PID {os.getpid()}] ERROR: Requested {world_size} GPUs for training, but only {torch.cuda.device_count()} are available.")
        return

    print(f"[TRAIN LAUNCHER PID {os.getpid()}] Spawning {world_size} DDP training processes for GPUs: {TRAIN_GPU_IDS}")
    mp.spawn(ddp_train_worker,
             args=(world_size, st),
             nprocs=world_size,
             join=True)
    print(f"[TRAIN LAUNCHER PID {os.getpid()}] All DDP training processes have finished.")


def serving(st):
    """
    Starts the NVIDIA Triton Inference Server on a dedicated GPU using Docker.
    """
    print(f"[SERVING PID {os.getpid()}] Serving function started. Attempting to launch Triton on GPU {SERVE_GPU_ID}...")

    # --gpus='"device=2"' is the correct syntax to specify a single GPU device for Docker
    docker_command_str = (
        f"docker run --name {DOCKER_CONTAINER_NAME} --gpus='\"device={SERVE_GPU_ID}\"' --rm --net=host "
        f"-v /mydata/Data/server/docs/examples/model_repository:/models "
        f"nvcr.io/nvidia/tritonserver:25.05-py3 "
        f"tritonserver --model-repository=/models --model-control-mode explicit --load-model resnet50"
    )
    # Note: subprocess.Popen works best with a list of arguments
    docker_command_list = docker_command_str.split()
    docker_process = None
    process_group_id = None

    try:
        print(f"[SERVING PID {os.getpid()}] Executing Docker command: {' '.join(docker_command_list)}")
        docker_process = subprocess.Popen(
            docker_command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        if docker_process.pid is not None:
            process_group_id = os.getpgid(docker_process.pid)
            print(f"[SERVING PID {os.getpid()}] Triton Docker container process started (PID: {docker_process.pid}, Process Group ID: {process_group_id}). Monitoring...")
            END_TIME_T2I = time.time()
            print(f"Fully T2I took {END_TIME_T2I - st:.2f}s")
        else:
            raise Exception("Failed to start Docker process and get PID.")

        while docker_process.poll() is None:
            time.sleep(1)
        print(f"[SERVING PID {os.getpid()}] Docker Popen process (PID: {docker_process.pid}) has exited with code {docker_process.returncode}.")

    except FileNotFoundError:
        print(f"[SERVING PID {os.getpid()}] Error: 'docker' command not found. Is Docker installed and in PATH?")
    except Exception as e:
        print(f"[SERVING PID {os.getpid()}] Error in serving function (launching/managing Docker): {e}")
    finally:
        # The rest of the finally block is robust and remains unchanged.
        print(f"[SERVING PID {os.getpid()}] ##### SERVING FUNCTION FINALLY BLOCK ENTERED ({DOCKER_CONTAINER_NAME}) #####")

        print(f"[SERVING PID {os.getpid()}] Attempting 'docker stop {DOCKER_CONTAINER_NAME}' (if not already stopped)...")
        try:
            stop_result = subprocess.run(
                ["docker", "stop", DOCKER_CONTAINER_NAME],
                timeout=20,
                capture_output=True, text=True, check=False
            )
            if stop_result.returncode == 0:
                print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' succeeded via finally block.")
            else:
                print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' (in finally) finished with RC {stop_result.returncode}. Stderr: {stop_result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' (in finally) timed out.")
        except FileNotFoundError:
            print(f"[SERVING PID {os.getpid()}] 'docker' command not found (in finally) while trying to stop container.")
        except Exception as e_docker_stop:
            print(f"[SERVING PID {os.getpid()}] Error during 'docker stop {DOCKER_CONTAINER_NAME}' (in finally): {e_docker_stop}")

        if docker_process and docker_process.pid is not None:
            if docker_process.poll() is None:
                print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) still active (in finally). Attempting to terminate its process group.")
                if process_group_id:
                    try:
                        os.killpg(process_group_id, signal.SIGTERM)
                        docker_process.wait(timeout=5)
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) for 'docker run' terminated after SIGTERM (in finally).")
                    except ProcessLookupError:
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) not found during SIGTERM (in finally).")
                    except subprocess.TimeoutExpired:
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) did not terminate after SIGTERM (in finally). Killing group...")
                        try:
                            os.killpg(process_group_id, signal.SIGKILL)
                            docker_process.wait(timeout=2)
                            print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) killed (in finally).")
                        except Exception as e_pg_kill:
                            print(f"[SERVING PID {os.getpid()}] Error killing process group (PGID: {process_group_id}) (in finally): {e_pg_kill}")
                    except Exception as e_pg_term:
                         print(f"[SERVING PID {os.getpid()}] Error terminating process group (PGID: {process_group_id}) (in finally): {e_pg_term}")
                else:
                    print(f"[SERVING PID {os.getpid()}] No PGID, attempting to terminate Popen process (PID: {docker_process.pid}) directly (in finally).")
                    docker_process.terminate()
                    try:
                        docker_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        docker_process.kill()
                        docker_process.wait(timeout=2) # Added wait after kill

            if docker_process.poll() is not None:
                 print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) has exited (in finally).")
            else:
                 print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) still alive after all attempts (in finally). Forcibly killing.")
                 docker_process.kill()
                 try:
                    docker_process.wait(timeout=2)
                 except subprocess.TimeoutExpired:
                    print(f"[SERVING PID {os.getpid()}] Popen object kill timed out (in finally).")
        print(f"[SERVING PID {os.getpid()}] Serving function finished.")


# --- Global state and configuration ---
Load = 0
Max_Load = 50
train_is_paused = False 

train_process = None
serving_process = None
process_management_lock = threading.Lock()
stop_monitor_event = threading.Event()

# --- Worker entry points for multiprocessing ---
def train_worker_entry(st):
    # This entry point now calls the DDP launcher function.
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) starting train() launcher.")
    s = st.value
    train(s)
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) exiting.")

def serving_worker_entry(st):
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) starting serving().")
    s = st.value
    serving(s)
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) exiting.")

# --- Core logic: Process Management ---
# This logic remains the same. SIGSTOP/SIGCONT sent to the parent `train_process`
# will correctly pause and resume the spawned DDP child processes.
def manage_processes():
    """
    Pauses/resumes the training process based on load.
    Does not start or stop processes, only sends signals.
    """
    global train_process, serving_process, Load, Max_Load, train_is_paused

    with process_management_lock:
        train_is_effectively_running = train_process is not None and train_process.is_alive()
        
        if not train_is_effectively_running:
            return

        # --- High Load Condition: PAUSE Training ---
        if Load > Max_Load and not train_is_paused:
            try:
                # Sending SIGSTOP to the parent process will pause its children too
                os.kill(train_process.pid, signal.SIGSTOP)
                train_is_paused = True
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Paused training process group (Parent PID: {train_process.pid}).")
            except ProcessLookupError:
                print(f"MONITOR: Could not find training process (PID: {train_process.pid}) to pause.")
            except Exception as e:
                print(f"MONITOR: Error pausing training process: {e}")

        # --- Low Load Condition: RESUME Training ---
        elif Load <= Max_Load and train_is_paused:
            try:
                # Sending SIGCONT to the parent process will resume its children too
                os.kill(train_process.pid, signal.SIGCONT)
                train_is_paused = False
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Resumed training process group (Parent PID: {train_process.pid}).")
            except ProcessLookupError:
                print(f"MONITOR: Could not find training process (PID: {train_process.pid}) to resume.")
            except Exception as e:
                print(f"MONITOR: Error resuming training process: {e}")


def monitor_thread_worker():
    print("MONITOR: Monitor thread started. Will check load every 2 seconds.")
    while not stop_monitor_event.is_set():
        manage_processes()
        time.sleep(2) # Check frequency
    print("MONITOR: Monitor thread stopped.")

# --- Cleanup Logic ---
# This logic is also largely unchanged and should correctly terminate the parent processes.
def cleanup_all_processes():
    print("MAIN: Initiating cleanup of all child processes...")
    global train_process, serving_process, train_is_paused

    with process_management_lock:
        if train_process and train_process.is_alive():
            print(f"MAIN: Cleaning up training process (PID: {train_process.pid})...")
            try:
                if train_is_paused:
                    print(f"MAIN: Resuming paused training process (PID: {train_process.pid}) before termination.")
                    os.kill(train_process.pid, signal.SIGCONT)
                    time.sleep(0.1)

                print(f"MAIN: Terminating training process (PID: {train_process.pid}). This will stop the DDP workers.")
                train_process.terminate() # Terminate is more forceful for multi-process spawners
                train_process.join(timeout=30)
                if train_process.is_alive():
                    print(f"MAIN: Training (PID: {train_process.pid}) still alive after terminate (30s). Killing.")
                    train_process.kill()
                    train_process.join(timeout=5)
            except Exception as e:
                print(f"MAIN: Error cleaning up training process: {e}")
            finally:
                if train_process: train_process.close()
                train_process = None

        if serving_process and serving_process.is_alive():
            # Serving cleanup logic is unchanged
            print(f"MAIN: Cleaning up serving Python process (PID: {serving_process.pid}). Docker container: {DOCKER_CONTAINER_NAME}")
            print(f"MAIN: Attempting 'docker stop {DOCKER_CONTAINER_NAME}' during final cleanup...")
            try:
                subprocess.run(["docker", "stop", DOCKER_CONTAINER_NAME], timeout=20, check=False)
            except Exception as e_docker_stop_final:
                print(f"MAIN: Error during final 'docker stop {DOCKER_CONTAINER_NAME}': {e_docker_stop_final}")

            try:
                serving_process.terminate()
                serving_process.join(timeout=15)
                if serving_process.is_alive():
                    print(f"MAIN: Serving Python process (PID: {serving_process.pid}) still alive after terminate (15s). Killing.")
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
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    print(f"MAIN: Program started. Default Max_Load is {Max_Load}. Initial Load is {Load}.")
    print("-" * 50)
    print(f"MAIN: CONFIGURATION")
    print(f"MAIN: Training will use GPUs: {TRAIN_GPU_IDS}")
    print(f"MAIN: Serving will use GPU: {SERVE_GPU_ID}")
    print(f"MAIN: ImageNet data path: {IMAGENET_DATA_PATH}")
    print(f"MAIN: Checkpoints will be saved in: {os.path.abspath(CHECKPOINT_DIR)}")
    print(f"MAIN: Docker container name: {DOCKER_CONTAINER_NAME}")
    print("-" * 50)
    
    if not PYTORCH_AVAILABLE:
        print("MAIN: Note - PyTorch is not installed, training will not function.")
    
    print("MAIN: Ensure Docker is running and you have permissions.")
    
    # Start BOTH processes at the beginning
    print("MAIN: Starting both training and serving processes concurrently...")
    
    start_time_dummy = multiprocessing.Value('d', time.time())
    
    train_process = multiprocessing.Process(target=train_worker_entry, args=(start_time_dummy,), daemon=False)
    train_process.start()
    print(f"MAIN: Training launcher process started (PID: {train_process.pid}).")

    serving_process = multiprocessing.Process(target=serving_worker_entry, args=(start_time_dummy,), daemon=False)
    serving_process.start()
    print(f"MAIN: Serving process started (PID: {serving_process.pid}).")
    
    print("-" * 50)
    print("MAIN: Enter an integer value for 'Load' to pause/resume the training process.")
    print("MAIN: Type 'q' or 'quit' to exit.")

    monitor = threading.Thread(target=monitor_thread_worker, daemon=True)
    monitor.start()

    try:
        while True:
            try:
                user_input = input(f"Enter new Load (current: {Load}, max: {Max_Load}, training paused: {train_is_paused}) (or 'q' to quit): ")
                if user_input.lower() in ['q', 'quit']:
                    print("MAIN: Quit command received. Shutting down...")
                    break
                Load = int(user_input)
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
        
        cleanup_all_processes()
        print("MAIN: Program terminated.")