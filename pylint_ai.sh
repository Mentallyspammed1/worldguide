#!/bin/bash

# Enhanced script to run pylint, fix linting errors with aichat, and update Python files

# Function to display help
show_help() {
    echo "Usage: $0 [options] <filename.py>"
    echo "Options:"
    echo "  --config <pylintrc>  Specify a custom pylint configuration file"
    echo "  --backup-count <N>   Keep only the last N backups (default: 5)"
    echo "  --log-dir <dir>      Directory to save pylint logs (default: ./pylint_logs)"
    echo "  --help               Display this help message"
    exit 0
}

# Default values
CONFIG_FILE=""
BACKUP_COUNT=5
LOG_DIR="./pylint_logs"

# Parse options
while [[ "$1" == --* ]]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --backup-count) BACKUP_COUNT="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --help) show_help ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# Input file
input_file="$1"

# Validate input
if [ -z "$input_file" ]; then
    echo "Error: No filename provided."
    show_help
fi

if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

if [[ ! "$input_file" =~ \.py$ ]]; then
    echo "Error: File '$input_file' is not a .py file."
    exit 1
fi

# Check for required tools
for tool in pylint aichat; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: $tool is not installed."
        exit 1
    fi
done

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Temporary files
timestamp=$(date +%s)
pylint_before="$LOG_DIR/pylint_before_${timestamp}.txt"
pylint_after="$LOG_DIR/pylint_after_${timestamp}.txt"
fixed_code="fixed_code_${timestamp}.py"
backup_file="${input_file%.py}_backup_${timestamp}.py"

# Run pylint with optional config
echo "Running pylint on '$input_file'..."
pylint_cmd="pylint"
[ -n "$CONFIG_FILE" ] && pylint_cmd="$pylint_cmd --rcfile=$CONFIG_FILE"
$pylint_cmd "$input_file" > "$pylint_before" 2>&1

# Check if pylint found issues
if grep -q "Your code has been rated at 10.00/10" "$pylint_before"; then
    echo "No pylint issues found."
    rm -f "$pylint_before"
    exit 0
fi

# Combine original code and pylint output for aichat
echo "Processing pylint output with aichat..."
{
    echo "Original code:"
    cat "$input_file"
    echo -e "\nPylint errors and warnings:"
    cat "$pylint_before"
} | aichat "Fix the pylint errors and warnings in this Python code based on the provided pylint output. Ensure the code adheres to PEP 8 standards, retains its original functionality, and includes clear comments where changes are made. Output only the corrected Python code." > "$fixed_code"

# Check if aichat succeeded
if [ $? -ne 0 ]; then
    echo "Error: aichat processing failed."
    rm -f "$pylint_before" "$fixed_code"
    exit 1
fi

# Check if fixed_code is empty
if [ ! -s "$fixed_code" ]; then
    echo "Error: aichat produced an empty output."
    rm -f "$pylint_before" "$fixed_code"
    exit 1
fi

# Backup original file
echo "Backing up original file to '$backup_file'..."
cp "$input_file" "$backup_file"

# Apply fixes
echo "Applying fixes to '$input_file'..."
mv "$fixed_code" "$input_file"

# Run pylint again to check for remaining issues
echo "Verifying fixes..."
$pylint_cmd "$input_file" > "$pylint_after" 2>&1

# Summarize results
if grep -q "Your code has been rated at 10.00/10" "$pylint_after"; then
    echo "All pylint issues resolved."
else
    echo "Some pylint issues may remain. Check '$pylint_after' for details."
fi

# Manage backups: keep only the last N backups
backup_pattern="${input_file%.py}_backup_*.py"
backup_files=($(ls -t $backup_pattern 2>/dev/null))
if [ ${#backup_files[@]} -gt $BACKUP_COUNT ]; then
    for ((i=$BACKUP_COUNT; i<${#backup_files[@]}; i++)); do
        rm -f "${backup_files[$i]}"
    done
    echo "Old backups removed, keeping the last $BACKUP_COUNT."
fi

# Clean up temporary files
rm -f "$pylint_before" "$pylint_after"
echo "Processing complete. Logs saved in '$LOG_DIR'."