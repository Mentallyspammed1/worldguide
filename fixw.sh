#!/usr/bin/env bash
#------------------------------------------------------------------------------
# AI Text File Enhancement Script v3.0
#
# Purpose: Takes an input text file (or stdin), uses the 'aichat' command-line
#          tool to enhance its content based on a specified prompt, and saves
#          the result to an output file (or stdout).
#
# Usage:
#   ./enhance_script.sh -i <input_file> -o <output_file> [-p <prompt>] [-f]
#   ./enhance_script.sh --input <input_file> --output <output_file> [--prompt "<prompt>"] [--force]
#   cat input.txt | ./enhance_script.sh -i - -o output.txt
#   ./enhance_script.sh -i input.txt -o -
#   ./enhance_script.sh -h | --help
#
# Options:
#   -i, --input FILE    : Input text file path, or '-' for stdin. (Required)
#   -o, --output FILE   : Output file path, or '-' for stdout. (Required)
#   -p, --prompt PROMPT : The prompt to use for the AI enhancement.
#                         (Default: "Enhance the following text for clarity, conciseness, and flow:")
#   -f, --force         : Overwrite the output file if it exists without prompting.
#   -h, --help          : Display this help message and exit.
#
# Dependencies:
#   - bash (v4.0+ recommended)
#   - aichat (command-line tool, must be in PATH)
#   - coreutils (mktemp, dirname, rm, cat, tr, echo, command, sed, cmp, stat)
#   - getopt (for command-line option parsing)
#
# Features:
#   - Flexible command-line options using getopt.
#   - Supports reading from stdin and writing to stdout.
#   - Configurable AI prompt.
#   - Strict mode for robust error handling (set -euo pipefail).
#   - Comprehensive input validation (existence, readability, type).
#   - Output validation (non-empty, different from input unless identical).
#   - Checks for directory writability for output file creation.
#   - Optional force overwrite flag.
#   - Colored status messages for better UX (errors to stderr).
#   - Secure temporary file handling using 'mktemp'.
#   - Reliable cleanup mechanism using 'trap'.
#   - Checks for required command dependencies.
#   - Detailed error reporting, including 'aichat' stderr output.
#   - Modular design using functions.
#------------------------------------------------------------------------------

# --- Strict Mode & Options ---
set -e
set -u
set -o pipefail
# Consider 'shopt -s nullglob' if needed, but not required here.

# --- Constants ---
readonly SCRIPT_NAME="$(basename "$0")"
readonly TEMP_FILE_TEMPLATE="${SCRIPT_NAME}.tmp.XXXXXX"
readonly DEFAULT_AI_PROMPT="Enhance the following text for clarity, conciseness, and flow:"

# Exit Codes
readonly EXIT_SUCCESS=0
readonly EXIT_USAGE_ERROR=1
readonly EXIT_DEPENDENCY_ERROR=2
readonly EXIT_INPUT_ERROR=3
readonly EXIT_OUTPUT_ERROR=4
readonly EXIT_AICHAT_ERROR=5
readonly EXIT_UNKNOWN_ERROR=10

# Color Codes for Output Messages
# Usage: log_info "Your message" or echo -e "${COLOR_CYAN}Manual message${COLOR_RESET}"
COLOR_RESET='\e[0m'
COLOR_CYAN='\e[0;36m'
COLOR_YELLOW='\e[1;33m' # Bold Yellow
COLOR_GREEN='\e[1;32m'  # Bold Green
COLOR_RED='\e[1;31m'    # Bold Red

# --- Global Variables ---
# Note: Use uppercase for constants, lowercase for script variables.
temp_file=""
input_file=""
output_file=""
ai_prompt="${DEFAULT_AI_PROMPT}"
force_overwrite=0 # 0 = false, 1 = true

# --- Logging Functions ---
# Usage: log_info "Starting process..."
log_info() {
    echo -e "${COLOR_CYAN}[INFO]${COLOR_RESET} $1" >&2 # Info to stderr to avoid mixing with potential stdout output
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1" >&2
}

log_warn() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $1" >&2
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1" >&2
}

# --- Utility Functions ---

# Function to display usage information
usage() {
    # Extract usage instructions and options from the script's header comment
    sed -n '/^# Usage:/,/^# Options:/ { s/^# *//; p }' "$0" | sed '$d' # Print Usage block
    echo "Options:"
    sed -n '/^# Options:/,/^# Dependencies:/ { /^# Options:/d; /^# Dependencies:/d; s/^# *//; p }' "$0" # Print Options block
}

# Cleanup function called on script exit
cleanup() {
    if [[ -n "${temp_file}" && -e "${temp_file}" ]]; then
        log_info "Cleaning up temporary file: ${temp_file}"
        rm -f "${temp_file}"
    fi
}

# Check for required command dependencies
check_dependencies() {
    local missing_deps=0
    for cmd in aichat getopt mktemp dirname rm cat tr echo command sed cmp stat; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command '$cmd' not found in PATH."
            missing_deps=1
        fi
    done
    if [[ "$missing_deps" -eq 1 ]]; then
        exit "$EXIT_DEPENDENCY_ERROR"
    fi
    log_info "All dependencies found."
}

# Validate input file path or stdin
validate_input() {
    if [[ "${input_file}" == "-" ]]; then
        log_info "Reading input from stdin."
        # Cannot validate stdin further here, will rely on 'cat' later
        return 0
    fi

    if [[ ! -e "${input_file}" ]]; then
        log_error "Input file not found: ${input_file}"
        exit "$EXIT_INPUT_ERROR"
    fi
    if [[ ! -f "${input_file}" ]]; then
        log_error "Input path is not a regular file: ${input_file}"
        exit "$EXIT_INPUT_ERROR"
    fi
    if [[ ! -r "${input_file}" ]]; then
        log_error "Input file is not readable: ${input_file}"
        exit "$EXIT_INPUT_ERROR"
    fi
    if [[ ! -s "${input_file}" ]]; then
        log_warn "Input file is empty: ${input_file}"
        # Allow empty input, AI might still generate something or error gracefully
    fi
    log_info "Input file validation passed: ${input_file}"
}

# Validate output file path or stdout
validate_output() {
    if [[ "${output_file}" == "-" ]]; then
        log_info "Writing output to stdout."
        # Cannot validate stdout further here
        return 0
    fi

    local output_dir
    output_dir=$(dirname "${output_file}")

    # Check if output directory exists
    if [[ ! -d "${output_dir}" ]]; then
        log_error "Output directory does not exist: ${output_dir}"
        exit "$EXIT_OUTPUT_ERROR"
    fi

    # Check if output directory is writable
    if [[ ! -w "${output_dir}" ]]; then
        log_error "Output directory is not writable: ${output_dir}"
        exit "$EXIT_OUTPUT_ERROR"
    fi

    # Check for overwrite if file exists
    if [[ -e "${output_file}" ]]; then
        if [[ "${force_overwrite}" -eq 1 ]]; then
            log_warn "Output file '${output_file}' exists and will be overwritten (--force specified)."
        else
            # Only prompt if stderr is a TTY (interactive session)
            if [[ -t 2 ]]; then
                 read -p "$(echo -e "${COLOR_YELLOW}[PROMPT]${COLOR_RESET} Output file '${output_file}' already exists. Overwrite? (y/N): ")" -n 1 -r REPLY
                 echo # Move to new line after input
                 if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
                    log_error "Operation cancelled by user."
                    exit "$EXIT_OUTPUT_ERROR"
                 fi
            else
                log_error "Output file '${output_file}' exists. Use --force to overwrite in non-interactive mode."
                exit "$EXIT_OUTPUT_ERROR"
            fi
        fi
         # Check if output path is a directory
        if [[ -d "${output_file}" ]]; then
            log_error "Output path exists and is a directory: ${output_file}"
            exit "$EXIT_OUTPUT_ERROR"
        fi
    fi

    log_info "Output path validation passed: ${output_file}"
}

# Run the AI enhancement process
run_ai_enhancement() {
    log_info "Starting AI enhancement..."
    log_info "Using prompt: \"${ai_prompt}\""

    # Create a secure temporary file for aichat output
    temp_file=$(mktemp --tmpdir "${TEMP_FILE_TEMPLATE}")
    log_info "Created temporary file: ${temp_file}"

    local aichat_cmd_input
    if [[ "${input_file}" == "-" ]]; then
        # Read from stdin and pipe to aichat
        aichat_cmd_input="cat -"
    else
        # Read from input file
        aichat_cmd_input="cat \"${input_file}\""
    fi

    # Prepare the full command, capturing aichat's stderr
    local aichat_stderr_file
    aichat_stderr_file=$(mktemp --tmpdir "${SCRIPT_NAME}.aichat_stderr.XXXXXX")
    # Ensure the stderr temp file is also cleaned up
    trap "rm -f '${aichat_stderr_file}'; cleanup" EXIT INT TERM

    # Execute the command
    # We pipe the prompt first, then the input content to aichat
    if ! (echo "${ai_prompt}"; echo ""; eval "${aichat_cmd_input}") | aichat > "${temp_file}" 2> "${aichat_stderr_file}"; then
        log_error "The 'aichat' command failed."
        if [[ -s "${aichat_stderr_file}" ]]; then
            log_error "--- aichat stderr output ---"
            cat "${aichat_stderr_file}" >&2 # Output stderr content to our stderr
            log_error "--- end aichat stderr output ---"
        else
            log_warn "No stderr output captured from aichat."
        fi
        rm -f "${aichat_stderr_file}" # Clean up stderr file immediately after use
        exit "$EXIT_AICHAT_ERROR"
    fi

    rm -f "${aichat_stderr_file}" # Clean up stderr file on success too

    # --- Post-processing Validation ---
    # Check if the temporary output file was created and is not empty
    if [[ ! -s "${temp_file}" ]]; then
        log_error "AI enhancement resulted in an empty file. Check 'aichat' functionality or prompt."
        exit "$EXIT_AICHAT_ERROR"
    fi

    # Check if output is identical to input (if not using stdin/stdout)
    if [[ "${input_file}" != "-" && "${output_file}" != "-" ]]; then
        if cmp -s "${input_file}" "${temp_file}"; then
            log_warn "AI enhancement resulted in output identical to the input file."
            # Decide whether to proceed or exit; proceeding is usually fine.
            # If this is an error condition, uncomment the next lines:
            # log_error "Output is identical to input. Aborting."
            # exit "$EXIT_AICHAT_ERROR"
        fi
    fi

    # --- Save Result ---
    if [[ "${output_file}" == "-" ]]; then
        log_info "Writing enhanced content to stdout..."
        cat "${temp_file}"
    else
        log_info "Saving enhanced content to: ${output_file}"
        # Use mv for atomicity if possible, though cat > redirection is common
        # Using cat > redirection for simplicity and broader compatibility here
        if ! cat "${temp_file}" > "${output_file}"; then
            log_error "Failed to write output to file: ${output_file}"
            exit "$EXIT_OUTPUT_ERROR"
        fi
        # Optional: Set permissions based on input file? chmod --reference="${input_file}" "${output_file}"
    fi

    log_success "AI enhancement completed successfully."
}

# --- Main Script Logic ---
main() {
    # Set trap for cleanup on exit, interrupt, or termination signals
    trap cleanup EXIT INT TERM

    # Check dependencies first
    check_dependencies

    # --- Argument Parsing ---
    # Need to use 'getopt' which supports long options and spaces in arguments
    local parsed_options
    parsed_options=$(getopt -o hi:o:p:f --long help,input:,output:,prompt:,force -n "${SCRIPT_NAME}" -- "$@")
    if [[ $? -ne 0 ]]; then
        usage >&2
        exit "$EXIT_USAGE_ERROR"
    fi

    # Use 'eval set --' to handle arguments with spaces correctly
    eval set -- "${parsed_options}"

    while true; do
        case "$1" in
            -h | --help)
                usage
                exit "$EXIT_SUCCESS"
                ;;
            -i | --input)
                input_file="$2"
                shift 2
                ;;
            -o | --output)
                output_file="$2"
                shift 2
                ;;
            -p | --prompt)
                ai_prompt="$2"
                shift 2
                ;;
            -f | --force)
                force_overwrite=1
                shift 1
                ;;
            --)
                shift # Skip the '--' separator
                break # End of options
                ;;
            *)
                # Should not happen with getopt error checking above
                log_error "Internal error parsing options."
                exit "$EXIT_UNKNOWN_ERROR"
                ;;
        esac
    done

    # Check for remaining arguments (should be none)
    if [[ $# -gt 0 ]]; then
        log_error "Unexpected arguments: $*"
        usage >&2
        exit "$EXIT_USAGE_ERROR"
    fi

    # --- Mandatory Argument Checks ---
    if [[ -z "${input_file}" ]]; then
        log_error "Input file/path is required (-i or --input)."
        usage >&2
        exit "$EXIT_USAGE_ERROR"
    fi
    if [[ -z "${output_file}" ]]; then
        log_error "Output file/path is required (-o or --output)."
        usage >&2
        exit "$EXIT_USAGE_ERROR"
    fi

    # --- Validation ---
    validate_input
    validate_output # Includes directory checks and overwrite prompt/check

    # --- Core Logic ---
    run_ai_enhancement

    # Cleanup will be called automatically via trap on exit
    exit "$EXIT_SUCCESS"
}

# --- Script Execution ---
# Call the main function, passing all script arguments
main "$@"

# Note: The explicit 'exit $EXIT_SUCCESS' at the end of main() is good practice,
#       though technically the script would exit 0 on successful completion anyway.
