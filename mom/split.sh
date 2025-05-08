#!/bin/bash

# Script to prompt for a Python file and split it into modules using aichat

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

# Run aichat with enhanced prompt
echo "Processing '$input_file' with aichat..."
cat "$input_file" | aichat "Split this long Python script into separate, self-contained modules for better organization. Each module should focus on a single responsibility (e.g., data processing, file handling, utilities). Mark each module clearly with a comment like '# File: module_name.py' at the start. Include necessary imports in each module, ensure no cross-module dependencies are broken, and provide a main script that imports and uses these modules. Output all code in a single stream with clear separation between modules." > "$temp_file"

# Check if aichat succeeded
if [ $? -ne 0 ]; then
    echo "Error: aichat processing failed. Check if aichat is installed and configured."
    rm -f "$temp_file"
    exit 1
fi

if [ ! -s "$temp_file" ]; then
    echo "Error: aichat produced an empty output."
    rm -f "$temp_file"
    exit 1
fi

# Split the output into separate files using awk
echo "Splitting output into separate module files..."
awk '/# File: /{filename=$3} filename{print > filename}' "$temp_file"

# Check if any files were created
if ls *.py >/dev/null 2>&1; then
    echo "Modules created successfully:"
    ls *.py
else
    echo "Warning: No module files were created. Check '$temp_file' for issues."
fi

# Clean up temporary file
rm -f "$temp_file"
echo "Temporary file removed. Processing complete."