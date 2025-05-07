#!/bin/bash

# Script to prompt for a Python file and split it into modules using aichat, placing them in a new directory

# Prompt user for input filename
echo "Enter the Python script filename (e.g., script.py):"
read -r input_file

# Validate input
if [ -z "$input_file" ]; then
    echo "Error: No filename provided."
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

if [[ ! "$input_file" =~ \.py$ ]]; then
    echo "Error: File '$input_file' is not a .py file."
    exit 1
fi

# Temporary output file for aichat
temp_file="temp_split_$$.py"

# Set up trap to remove temp_file on exit
trap "rm -f '$temp_file'" EXIT

# Determine a unique directory name based on "modules"
base_dir="modules"
dir_name="$base_dir"
counter=1
while [ -d "$dir_name" ]; do
    dir_name="${base_dir}_${counter}"
    counter=$((counter + 1))
done

# Create the directory
mkdir "$dir_name"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory '$dir_name'."
    exit 1
fi

# Run aichat with enhanced prompt
echo "Processing '$input_file' with aichat..."
cat "$input_file" | aichat "Split this long Python script into separate, self-contained modules for better organization. Each module should focus on a single responsibility (e.g., data processing, file handling, utilities). Mark each module clearly with a comment like '# File: module_name.py' at the start. Include necessary imports in each module, ensure no cross-module dependencies are broken, and provide a main script that imports and uses these modules. Output all code in a single stream with clear separation between modules." > "$temp_file"

# Check if aichat succeeded
if [ $? -ne 0 ]; then
    echo "Error: aichat processing failed. Check if aichat is installed and configured."
    exit 1
fi

if [ ! -s "$temp_file" ]; then
    echo "Error: aichat produced an empty output."
    exit 1
fi

# Split the output into separate files in the new directory using awk
echo "Splitting output into separate module files in '$dir_name'..."
awk -v dir="$dir_name" '/# File: /{filename=$3} filename{print > (dir "/" filename)}' "$temp_file"

# Check if any files were created
if ls "$dir_name"/*.py >/dev/null 2>&1; then
    echo "Modules created in '$dir_name':"
    ls "$dir_name"
else
    echo "Warning: No module files were created in '$dir_name'. Check '$temp_file' for issues."
fi

echo "Processing complete."