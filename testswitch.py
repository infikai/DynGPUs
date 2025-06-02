import time
import os
import signal
import multiprocessing
import threading

# --- User-defined functions (placeholders) ---
def train():
    """Placeholder for the training function."""
    print(f"[TRAIN PID {os.getpid()}] Training started. Simulating work...")
    try:
        while True:
            print(f"[TRAIN PID {os.getpid()}] Training batch processing...")
            time.sleep(5)  # Simulate work
    except KeyboardInterrupt:
        print(f"[TRAIN PID {os.getpid()}] Training function received KeyboardInterrupt. Cleaning up...")
        time.sleep(1)  # Simulate cleanup
        print(f"[TRAIN PID {os.getpid()}] Training function cleanup complete. Exiting.")
    except Exception as e:
        print(f"[TRAIN PID {os.getpid()}] Error in training: {e}")
    finally:
        print(f"[TRAIN PID {os.getpid()}] Training function finished.")

def serving():
    """Placeholder for the serving function."""
    print(f"[SERVING PID {os.getpid()}] Serving started. Simulating request handling...")
    try:
        while True:
            print(f"[SERVING PID {os.getpid()}] Serving request...")
            time.sleep(3)  # Simulate work
    except Exception as e:
        print(f"[SERVING PID {os.getpid()}] Error in serving: {e}")
    finally:
        print(f"[SERVING PID {os.getpid()}] Serving function finished.")

# --- Global state and configuration ---
Load = 0
Max_Load = 50  # Threshold, alter as needed

train_process = None
serving_process = None

# Lock for managing process handles and related flags
process_management_lock = threading.Lock()

# Event to signal the monitor thread to stop
stop_monitor_event = threading.Event()

# --- Worker entry points for multiprocessing ---
def train_worker_entry():
    """Entry point for the training process."""
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) starting train().")
    train()
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) exiting.")

def serving_worker_entry():
    """Entry point for the serving process."""
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) starting serving().")
    serving()
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) exiting.")

# --- Core logic: Process Management ---
def manage_processes():
    """
    Checks the Load and starts/stops training or serving processes accordingly.
    This function is called periodically by the monitor thread.
    """
    global train_process, serving_process, Load, Max_Load

    with process_management_lock:
        # Check current state of processes
        # A process is effectively running if the object exists and is_alive() is true
        train_is_effectively_running = train_process is not None and train_process.is_alive()
        serving_is_effectively_running = serving_process is not None and serving_process.is_alive()

        # Condition: Load is high, prioritize serving
        if Load > Max_Load:
            # Step 1: Shut down training if it's running
            if train_is_effectively_running:
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Attempting to shut down training (PID: {train_process.pid}).")
                try:
                    os.kill(train_process.pid, signal.SIGINT) # Send KeyboardInterrupt
                    train_process.join(timeout=10)  # Wait for graceful shutdown
                    if train_process.is_alive():
                        print(f"MONITOR: Training (PID: {train_process.pid}) did not stop after SIGINT. Forcing kill.")
                        train_process.kill()  # Force kill if it doesn't respond
                        train_process.join(timeout=5) # Wait for kill to complete
                except ProcessLookupError: # Process already exited
                    print(f"MONITOR: Training process (PID: {train_process.pid}) not found, likely already exited.")
                except Exception as e:
                    print(f"MONITOR: Error stopping training process (PID: {train_process.pid if train_process else 'N/A'}): {e}")
                finally:
                    if train_process and not train_process.is_alive(): # Ensure it's truly stopped
                        train_process.close() # Release resources
                        train_process = None
                        print("MONITOR: Training process confirmed shut down.")
                    elif train_process: # If still alive after all attempts
                         print(f"MONITOR: Training process (PID: {train_process.pid}) still alive after shutdown attempts.")


            # Step 2: Start serving if it's not running
            # Re-check serving status as train shutdown might take time or serving might have started/stopped independently
            serving_is_effectively_running = serving_process is not None and serving_process.is_alive()
            if not serving_is_effectively_running:
                # Clean up any defunct serving process object before starting a new one
                if serving_process: # If object exists but not alive
                    if not serving_process.is_alive(): serving_process.close()
                    serving_process = None # Clear it regardless to be safe

                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Starting serving...")
                serving_process = multiprocessing.Process(target=serving_worker_entry, daemon=True)
                serving_process.start()
                print(f"MONITOR: Serving process started (PID: {serving_process.pid}).")
            # else:
                # print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Serving is already running (PID: {serving_process.pid}).")

        # Condition: Load is low or normal, prioritize training
        elif Load <= Max_Load:
            # Step 1: Shut down serving if it's running
            if serving_is_effectively_running:
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Attempting to shut down serving (PID: {serving_process.pid}).")
                try:
                    serving_process.terminate()  # SIGTERM for serving
                    serving_process.join(timeout=10)
                    if serving_process.is_alive():
                        print(f"MONITOR: Serving (PID: {serving_process.pid}) did not stop after terminate. Forcing kill.")
                        serving_process.kill()
                        serving_process.join(timeout=5)
                except ProcessLookupError:
                     print(f"MONITOR: Serving process (PID: {serving_process.pid}) not found, likely already exited.")
                except Exception as e:
                    print(f"MONITOR: Error stopping serving process (PID: {serving_process.pid if serving_process else 'N/A'}): {e}")
                finally:
                    if serving_process and not serving_process.is_alive():
                        serving_process.close()
                        serving_process = None
                        print("MONITOR: Serving process confirmed shut down.")
                    elif serving_process:
                        print(f"MONITOR: Serving process (PID: {serving_process.pid}) still alive after shutdown attempts.")


            # Step 2: Start training if it's not running
            # Re-check train status
            train_is_effectively_running = train_process is not None and train_process.is_alive()
            if not train_is_effectively_running:
                if train_process: # If object exists but not alive
                    if not train_process.is_alive(): train_process.close()
                    train_process = None

                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Starting training...")
                train_process = multiprocessing.Process(target=train_worker_entry, daemon=True)
                train_process.start()
                print(f"MONITOR: Training process started (PID: {train_process.pid}).")
            # else:
                # print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Training is already running (PID: {train_process.pid}).")

def monitor_thread_worker():
    """Thread worker function to periodically call manage_processes."""
    print("MONITOR: Monitor thread started. Will check load every 2 seconds.")
    while not stop_monitor_event.is_set():
        manage_processes()
        # Adjust sleep time as needed. Shorter for faster reaction, longer for less overhead.
        time.sleep(2) 
    print("MONITOR: Monitor thread stopped.")

def cleanup_all_processes():
    """Attempts to clean up any running child processes on script exit."""
    print("MAIN: Initiating cleanup of all child processes...")
    global train_process, serving_process
    
    # Acquire lock to prevent monitor thread from interfering during cleanup
    with process_management_lock:
        if train_process and train_process.is_alive():
            print(f"MAIN: Cleaning up training process (PID: {train_process.pid})...")
            try:
                os.kill(train_process.pid, signal.SIGINT) # Try graceful SIGINT first
                train_process.join(timeout=5)
                if train_process.is_alive(): # If still alive, force kill
                    print(f"MAIN: Training (PID: {train_process.pid}) still alive. Killing.")
                    train_process.kill()
                    train_process.join(timeout=2) # Wait for kill
            except Exception as e:
                print(f"MAIN: Error cleaning up training process: {e}")
            finally:
                if train_process: train_process.close() # Close handle
                train_process = None

        if serving_process and serving_process.is_alive():
            print(f"MAIN: Cleaning up serving process (PID: {serving_process.pid})...")
            try:
                serving_process.terminate() # Try SIGTERM
                serving_process.join(timeout=5)
                if serving_process.is_alive(): # If still alive, force kill
                    print(f"MAIN: Serving (PID: {serving_process.pid}) still alive. Killing.")
                    serving_process.kill()
                    serving_process.join(timeout=2) # Wait for kill
            except Exception as e:
                print(f"MAIN: Error cleaning up serving process: {e}")
            finally:
                if serving_process: serving_process.close() # Close handle
                serving_process = None
    print("MAIN: Cleanup finished.")

# --- Main execution ---
if __name__ == "__main__":
    # This is important for scripts using multiprocessing that might be frozen (e.g., with PyInstaller)
    multiprocessing.freeze_support() 

    print(f"MAIN: Program started. Default Max_Load is {Max_Load}. Initial Load is {Load}.")
    print("MAIN: Enter an integer value for 'Load' to change system behavior.")
    print("MAIN: Type 'q' or 'quit' to exit.")

    # Start the monitor thread
    # It's a daemon thread so it won't prevent the main program from exiting
    monitor = threading.Thread(target=monitor_thread_worker, daemon=True)
    monitor.start()

    try:
        while True:
            try:
                # Display current Load and Max_Load in the prompt for better UX
                user_input = input(f"Enter new Load (current: {Load}, max: {Max_Load}) (or 'q' to quit): ")
                if user_input.lower() in ['q', 'quit']:
                    print("MAIN: Quit command received. Shutting down...")
                    break
                
                new_load = int(user_input)
                Load = new_load # Update global Load variable
                print(f"MAIN: Load updated to {Load}. Monitor will adjust processes shortly.")
                # The monitor thread will pick up the change in its next cycle.

            except ValueError:
                print("MAIN: Invalid input. Please enter an integer for Load or 'q' to quit.")
            except EOFError: # Handle Ctrl+D or if input is piped and ends
                print("MAIN: EOF detected, exiting...")
                break
    
    except KeyboardInterrupt:
        print("MAIN: KeyboardInterrupt received in main loop. Shutting down...")
    finally:
        print("MAIN: Starting final shutdown sequence...")
        stop_monitor_event.set() # Signal monitor thread to stop its loop

        if monitor.is_alive():
            print("MAIN: Waiting for monitor thread to finish...")
            monitor.join(timeout=5) # Wait for monitor thread to finish
            if monitor.is_alive():
                print("MAIN: Monitor thread did not stop in time.")
        
        cleanup_all_processes() # Clean up any running child processes
        
        print("MAIN: Program terminated.")
