#!/bin/bash

# --- 1. Validate and Assign Input Argument ---
# Check if an argument for total workers was provided.
if [ -z "$1" ]; then
    echo "Error: Missing argument for total number of workers."
    echo "Usage: $0 <total_workers>"
    exit 1
fi

# Check if the provided argument is a positive integer.
if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
    echo "Error: Invalid argument. <total_workers> must be a positive integer."
    echo "Usage: $0 <total_workers>"
    exit 1
fi

# Assign the first command-line argument ($1) to the TOTAL_WORKERS variable.
TOTAL_WORKERS=$1

## --- Script Logic (largely unchanged) ---

# The file containing the list of active workers
FILENAME="active_workers.txt"

# First, check if the active workers file exists. If not, create a default one.
if [ ! -f "$FILENAME" ]; then
    echo "File '$FILENAME' not found. Creating it with default workers '0,1,2'."
    echo "0,1,2" > "$FILENAME"
fi

# Read the current ranks and count how many active workers there are.
current_ranks=$(<"$FILENAME")
num_active=$(echo "$current_ranks" | tr ',' '\n' | wc -l)

# Safety check: Ensure the number of active workers doesn't exceed the total available.
if [ "$num_active" -gt "$TOTAL_WORKERS" ]; then
    echo "Error: Number of active workers ($num_active) in '$FILENAME' is greater than the total you specified ($TOTAL_WORKERS)."
    exit 1
fi

# Create a list of all possible worker ranks *except* for rank 0.
other_workers=$(seq 1 $((TOTAL_WORKERS - 1)))

# Determine how many random workers we need to select (besides rank 0).
num_to_select=$((num_active - 1))

# Handle the two cases: only rank 0, or rank 0 plus others.
if [ "$num_to_select" -le 0 ]; then
    new_ranks="0"
else
    # Shuffle the list of other workers and pick the number we need.
    random_workers=$(echo "$other_workers" | shuf | head -n "$num_to_select")

    # Combine rank 0 with the new random workers, then sort them numerically.
    all_new_ranks=$(echo -e "0\n$random_workers" | sort -n)

    # Format the final list into a single, comma-separated string.
    new_ranks=$(echo "$all_new_ranks" | tr '\n' ',' | sed 's/,$//')
fi

# Overwrite the file with the new list of active workers.
echo "$new_ranks" > "$FILENAME"

echo "✅ Success! Updated '$FILENAME' using a pool of $TOTAL_WORKERS workers. New ranks: $new_ranks"
