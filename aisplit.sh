#!/bin/bash

# Script to split a large Python script into smaller modules using aichat
# and place them in a specified or uniquely generated directory.
# Enhanced with better error handling, user options, and optional syntax checking.

show_help() {
    echo "Usage: $0 [-f <filename>] [-d <directory>] [-l <logfile>] [-s] [-h]"
    echo "Options:"
    echo "  -f <filename>   Specify the input Python file."
    echo "  -d <directory>  Specify the output directory for modules."
    echo "  -l <logfile>    Specify a log file for output."
    echo "  -s              Check syntax of generated modules."
    echo "  -h              Show this help message."
    exit 0
}

input_file=""
output_dir=""
log_file=""
check_syntax=false

while getopts "f:d:l:sh" opt; do
    case $opt in
        f) input_file="$OPTARG" ;;
        d) output_dir="$OPTARG" ;;
        l) log_file="$OPTARG" ;;
        s) check_syntax=true ;;
        h) show_help ;;
        *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

if [ -z "$input_file" ]; then
    read -p "Enter the Python script filename (e.g., script.py): " input_file
fi

if [ -z "$input_file" ]; then
    echo "Error: No filename provided." >&2
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist." >&2
    exit 1
fi

if [ ! -r "$input_file" ]; then
    echo "Error: File '$input_file' is not readable." >&2
    exit 1
fi

if [[ ! "$input_file" =~ \.py$ ]]; then
    echo "Error: File '$input_file' is not a .py file." >&2
    exit 1
fi

backup_file="${input_file}.bak"
cp "$input_file" "$backup_file" || { echo "Error: Failed to create backup." >&2; exit 1; }
echo "Backup created: $backup_file"

if [ -z "$output_dir" ]; then
    base_dir="modules"
    dir_name="$base_dir"
    counter=1
    while [ -d "$dir_name" ]; do
        dir_name="${base_dir}_${counter}"
        counter=$((counter + 1))
    done
    output_dir="$dir_name"
fi

mkdir -p "$output_dir" || { echo "Error: Failed to create directory '$output_dir'." >&2; exit 1; }
echo "Output directory: $output_dir"

temp_file=$(mktemp) || { echo "Error: Failed to create temporary file." >&2; exit 1; }
trap "rm -f '$temp_file'" EXIT

if ! command -v aichat &> /dev/null; then
    echo "Error: aichat is not installed or not in PATH." >&2
    exit 1
fi

echo "Processing '$input_file' with aichat..."
cat "$input_file" | aichat "Split this long Python script into separate, self-contained modules for better organization. Each module should focus on a single responsibility (e.g., data processing, file handling, utilities). Mark each module clearly with a comment like '# File: module_name.py' at the start. Include necessary imports in each module, ensure no cross-module dependencies are broken, and provide a main script that imports and uses these modules. Output all code in a single stream with clear separation between modules." > "$temp_file"

if [ $? -ne 0 ]; then
    echo "Error: aichat processing failed." >&2
    exit 1
fi

if [ ! -s "$temp_file" ]; then
    echo "Error: aichat produced an empty output." >&2
    exit 1
fi

if ! grep -q "^# File: " "$temp_file"; then
    echo "Error: aichat output does not contain expected '# File: ' markers." >&2
    exit 1
fi

echo "Splitting output into separate module files in '$output_dir'..."
awk -v dir="$output_dir" '/^# File: /{filename=$3} filename{print > (dir "/" filename)}' "$temp_file"

if ls "$output_dir"/*.py >/dev/null 2>&1; then
    echo "Modules created in '$output_dir':"
    ls "$output_dir"
else
    echo "Warning: No module files were created in '$output_dir'. Check '$temp_file' for issues." >&2
fi

if $check_syntax; then
    echo "Checking syntax of generated modules..."
    for file in "$output_dir"/*.py; do
        if [ -f "$file" ]; then
            python -m py_compile "$file"
            if [ $? -ne 0 ]; then
                echo "Syntax error in $file" >&2
            else
                echo "$file: Syntax OK"
            fi
        fi
    done
fi

echo "Processing complete."