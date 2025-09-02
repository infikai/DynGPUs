#!/bin/bash

## --- 1. Validate and Assign Input Argument ---
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

# Assign the input argument to the TOTAL_WORKERS variable.
TOTAL_WORKERS=$1
FILENAME="active_workers.txt"

## --- 2. Decide on a Random Number of Workers to Activate ---
# The number of active workers will be a random value between 1 and TOTAL_WORKERS (inclusive).
# We use the built-in '$RANDOM' variable.
num_active=$(( (RANDOM % TOTAL_WORKERS) + 1 ))


## --- 3. Generate the Random List of Ranks ---
# We no longer read from the file; we use the 'num_active' we just generated.

# Create a list of all possible worker ranks *except* for rank 0.
other_workers=$(seq 1 $((TOTAL_WORKERS - 1)))

# Determine how many random workers we need to select (besides rank 0).
num_to_select=$((num_active - 1))

# Handle the case where only rank 0 is chosen.
if [ "$num_to_select" -le 0 ]; then
    new_ranks="0"
else
    # Shuffle the list of other workers and pick the random number we need.
    random_workers=$(echo "$other_workers" | shuf | head -n "$num_to_select")

    # Combine rank 0 with the new random workers, then sort them numerically.
    all_new_ranks=$(echo -e "0\n$random_workers" | sort -n)

    # Format the final list into a single, comma-separated string.
    new_ranks=$(echo "$all_new_ranks" | tr '\n' ',' | sed 's/,$//')
fi

## --- 4. Write the New List to the File ---
echo "$new_ranks" > "$FILENAME"

echo "✅ Success! Generated a new list with a random count of $num_active workers."
echo "New ranks: $new_ranks"