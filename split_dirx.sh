#!/bin/bash

# split_dir.txt
# Script to prompt for a Python file or take it as a command-line argument,
# use a configurable AI tool (like 'aichat', 'ollama', etc.) to split it
# into logical modules, and place these modules into a newly created directory.
#
# The script relies critically on the AI tool producing output where each module
# or file block is clearly marked at the beginning with a line exactly like:
# # File: module_name.py
#
# This script parses that specific marker to determine filenames and boundaries.
#
# Requires:
# - bash shell
# - An AI command-line tool (default is 'aichat', but configurable via AI_COMMAND)
# - awk utility
# - standard Unix tools (cat, mkdir, rm, ls, du, command)
#
# Usage:
# 1. Make the script executable: chmod +x split_dir.txt
# 2. Run without arguments (will prompt): ./split_dir.txt
# 3. Run with input file: ./split_dir.txt input_script.py
# 4. Run with input file and specific output directory base name:
#    ./split_dir.txt input_script.py my_modules
#
# Configure AI Command:
# You can change the 'AI_COMMAND' variable below if you use a different tool.
# Example for Ollama: AI_COMMAND="ollama run codellama"
# Example for LLM: AI_COMMAND="llm --model gpt-4" # Requires LLM tool setup

# --- Configuration ---
# Command to invoke your AI tool. This tool must accept standard input
# and produce standard output containing the split code blocks.
AI_COMMAND="aichat" # !! CONFIGURE THIS TO YOUR AI TOOL !!

# Base name for the output directory. Can be overridden by a command-line argument.
DEFAULT_BASE_DIR="modules"

# Name for the temporary file storing AI's raw output before splitting
TEMP_FILE="ai_split_raw_output_$$.txt" # Use .txt extension for raw text

# The specific marker expected from the AI to delineate files.
# The awk script relies on this exact format: "# File: filename.py"
# with the filename immediately following the space after the colon.
FILE_MARKER_PATTERN="^[[:space:]]*# File:[[:space:]]+(.*)$"

# --- Trap for Cleanup ---
# Ensure the temporary file is removed even if the script exits early or is interrupted
trap "rm -f '$TEMP_FILE'; echo 'Cleanup complete.'" EXIT

# --- Check Prerequisites ---
echo "Checking for required commands..."
if ! command -v "$AI_COMMAND" &>/dev/null; then
    echo "Error: AI command '$AI_COMMAND' not found."
    echo "Please install it or change the 'AI_COMMAND' variable in the script to your AI tool."
    exit 1
fi
echo "Using AI command: '$AI_COMMAND'"

if ! command -v awk &>/dev/null; then
    echo "Error: 'awk' command not found. It is required for splitting the output."
    echo "Please install 'awk' (usually part of core system utilities)."
    exit 1
fi
echo "Required commands found."
echo "" # Add a blank line for readability

# --- Input Handling ---

input_file=""
base_dir=""

# Check for command-line arguments
if [ "$#" -ge 1 ]; then
    input_file="$1"
    if [ "$#" -ge 2 ]; then
        base_dir="$2"
    fi
else
    # No arguments, prompt the user
    echo "Enter the Python script filename to split (e.g., script.py):"
    read -r input_file
    echo "Enter the base name for the output directory (default: $DEFAULT_BASE_DIR):"
    read -r base_dir_input
    if [ -n "$base_dir_input" ]; then
        base_dir="$base_dir_input"
    fi
fi

# Use default base_dir if not provided via arg or prompt
if [ -z "$base_dir" ]; then
    base_dir="$DEFAULT_BASE_DIR"
fi

echo "Input file set to: '$input_file'"
echo "Output directory base name set to: '$base_dir'"

# --- Input Validation ---
if [ -z "$input_file" ]; then
    echo "Error: No filename provided. Exiting."
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' does not exist or is not a regular file. Exiting."
    exit 1
fi

if [[ ! "$input_file" =~ \.py$ ]]; then
    echo "Warning: File '$input_file' does not have a .py extension. Proceeding, but ensure it's valid Python."
    # Note: We don't exit here, allowing splitting of files without .py if user confirms/provides.
fi

# --- Directory Creation ---
# Determine a unique directory name based on the chosen base name
dir_name="$base_dir"
counter=1
while [ -d "$dir_name" ]; do
    dir_name="${base_dir}_${counter}"
    counter=$((counter + 1))
done

echo "" # Add a blank line
echo "Attempting to create output directory: '$dir_name'"
mkdir "$dir_name"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory '$dir_name'. Check permissions or disk space. Exiting."
    exit 1
fi
echo "Successfully created directory: '$dir_name'"

# --- AI Processing ---
# The prompt instructs the AI to split the code, mark each module with '# File: filename.py',
# include necessary imports, maintain dependencies, and provide a main script
# marked as '# File: main.py' to demonstrate usage. The parsing below relies heavily on this exact format.
# IMPORTANT: The AI's ability to follow this prompt determines the script's success.
echo "" # Add a blank line
echo "Processing '$input_file' using '$AI_COMMAND'..."
# We wrap the AI_COMMAND in 'eval' to allow for commands with arguments (e.g., "ollama run model")
# Be cautious with eval if AI_COMMAND could come from untrusted sources (not the case here, it's script config)
eval "$AI_COMMAND" <<< "$(cat "$input_file")" > "$TEMP_FILE"

# --- Check AI Output ---
ai_exit_code=$?
if [ $ai_exit_code -ne 0 ]; then
    echo "Error: AI command '$AI_COMMAND' processing failed with exit code $ai_exit_code."
    echo "This might mean the AI tool encountered an error, or could not process the request."
    # If there was *some* output despite the error, it might contain clues
    if [ -s "$TEMP_FILE" ]; then
         echo "Partial or error output from AI saved to '$TEMP_FILE'."
         echo "Contents of '$TEMP_FILE':"
         cat "$TEMP_FILE"
    fi
    exit 1
fi

if [ ! -s "$TEMP_FILE" ]; then
    echo "Error: AI command '$AI_COMMAND' produced an empty output file '$TEMP_FILE'."
    echo "This could mean the AI failed to respond or had an internal issue."
    exit 1
fi

echo "AI processing complete. Raw output saved to '$TEMP_FILE'."
echo "Raw output size: $(du -h "$TEMP_FILE" | cut -f1)"

# --- Splitting Output into Files ---
# This awk script reads the AI's raw output ($TEMP_FILE), looks for the
# specific file marker pattern, extracts the filename, closes the previous file
# being written to, and writes subsequent lines to the new file within the
# output directory until the next marker or the end of input.
echo "" # Add a blank line
echo "Splitting AI output into separate module files in '$dir_name' using awk..."
# Pass the directory name and the marker pattern to awk as variables
awk -v dir="$dir_name" -v marker_pattern="$FILE_MARKER_PATTERN" '
    BEGIN {
        filename = "" # Initialize filename
    }

    # Use match() with the provided pattern to find marker lines and capture the filename
    match($0, marker_pattern, groups) {
        # Found a file marker line

        # Close the previous file if one was open
        if (filename) {
            close(dir "/" filename)
        }

        # Extract the new filename from the first capture group (.*)
        filename = groups[1]

        # Basic validation for extracted filename
        if (filename == "" || filename ~ /[ \/\\;:]/) {
             print "Warning: Invalid filename extracted: '" filename "' from line: " $0 | "cat >&2" # Print warning to stderr
             # Assign a safe, unique fallback filename
             filename = "parse_error_" NR ".py"
             print "Using fallback filename: " filename | "cat >&2" # Print fallback notice to stderr
        }

        # Ensure filename ends with .py if it doesnt (optional, AI should do this)
        if (filename !~ /\.py$/) {
             print "Warning: Filename '" filename "' does not end with .py. Appending .py" | "cat >&2"
             filename = filename ".py"
        }


        # Skip the marker line itself from being written to the output file
        next
    }

    # If we have successfully extracted a filename from a previous marker line,
    # print the current line to the corresponding file within the output directory.
    # This block is only executed if the current line is *not* a file marker.
    filename {
        print $0 > (dir "/" filename)
    }
' "$TEMP_FILE"

awk_exit_code=$?
if [ $awk_exit_code -ne 0 ]; then
    echo "Error: Awk splitting failed with exit code $awk_exit_code."
    echo "This could indicate an issue with the awk script or unexpected content in '$TEMP_FILE'."
    # Don't exit here, proceed to check created files, but indicate a problem occurred.
fi

# --- Post-Splitting Check and User Guidance ---
echo "" # Add a blank line
echo "Splitting process complete."

# Check if any .py files were actually created in the target directory
shopt -s nullglob # Prevent error if no files match
py_files=("$dir_name"/*.py)
shopt -u nullglob # Turn nullglob off

if [ "${#py_files[@]}" -gt 0 ] && [ -f "${py_files[0]}" ]; then # Check if array is not empty and first element exists
    echo "Successfully created the following Python module files in '$dir_name':"
    printf "%s\n" "${py_files[@]}" | sed "s|^$dir_name/||" # List files relative to the new directory
    echo ""
    echo "Review Required:"
    echo "=================="
    echo "Please carefully review the generated files in '$dir_name'."
    echo "Verify that:"
    echo "1. The code is logically split and correct."
    echo "2. Imports are present and correct in each module."
    echo "3. Cross-module dependencies are handled correctly."
    echo "4. The code meets your quality, security, and performance standards."
    echo ""
    # Suggest the main entry point if it likely exists
    if [ -f "$dir_name/main.py" ]; then
        echo "The intended main entry point should be '$dir_name/main.py' (check its content)."
        echo "To run the split modules, you might execute '$dir_name/main.py'."
    else
         echo "A 'main.py' file was not found in the output directory. You may need to create one manually"
         echo "to import and use the generated modules."
    fi
    echo ""
    echo "Reminder: AI-generated code is a starting point, not a final solution."

    # Optional: Provide a summary count and total size
    echo "Summary: Created ${#py_files[@]} .py files."
    # Calculate total size, ignoring directories if present (unlikely with *.py)
    total_size=$(find "$dir_name" -maxdepth 1 -type f -name "*.py" -print0 | du -ch --files0-from=- | tail -n1 | cut -f1)
    echo "Total size of created module files: $total_size"

else
    echo "Warning: No Python module files (.py) were found in the directory '$dir_name'."
    echo "This likely means the AI output in '$TEMP_FILE' did not contain the expected marker '$FILE_MARKER_PATTERN',"
    echo "or the awk script failed to parse them correctly."
    echo "Please inspect the content of '$TEMP_FILE' to understand why splitting failed."
    echo "Contents of '$TEMP_FILE':"
    cat "$TEMP_FILE"
    exit 1 # Exit with an error code as the core task (splitting) failed
fi

# Trap will handle cleaning up $TEMP_FILE automatically upon script exit.

--- END OF FILE split_dir.txt ---