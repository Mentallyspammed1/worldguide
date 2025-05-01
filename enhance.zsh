#!/usr/bin/env zsh
# set -e: Halt on any error
# set -o pipefail: Ensure pipeline errors propagate (zsh uses setopt pipefail)
# set -u: Treat unset variables as errors
setopt errexit pipefail nounset

# === Arcane Constants & Colors ===
readonly COLOR_RESET='\033[0m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_BOLD_RED='\033[1;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_BOLD_GREEN='\033[1;32m'
readonly COLOR_YELLOW='\033[0;33m'
readonly COLOR_BOLD_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_BOLD_BLUE='\033[1;34m'
readonly COLOR_MAGENTA='\033[0;35m'
readonly COLOR_BOLD_MAGENTA='\033[1;35m'
readonly COLOR_CYAN='\033[0;36m'
readonly COLOR_BOLD_CYAN='\033[1;36m'
readonly reset_color="$COLOR_RESET"

# === Default Configuration ===
readonly DEFAULT_API_MODEL="gemini-1.5-pro-latest"
readonly DEFAULT_API_TEMPERATURE="0.3"
readonly DEFAULT_CONNECT_TIMEOUT="20" # Seconds
readonly DEFAULT_MAX_TIME="180"      # Seconds
readonly TEMP_MIN="0.0"
readonly TEMP_MAX="2.0"
readonly MAX_RETRIES=2
readonly RETRY_DELAY=3
readonly API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/models"

# === Configuration File ===
readonly PYRMETHUS_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/pyrmethus"
readonly PYRMETHUS_CONFIG_FILE="${PYRMETHUS_CONFIG_DIR}/config"

# === Global Variables ===
INPUT_FILE_PATH=""
OUTPUT_FILE_NAME=""
GEMINI_API_KEY=""
API_MODEL=""
API_TEMPERATURE=""
CONNECT_TIMEOUT="${DEFAULT_CONNECT_TIMEOUT}"
MAX_TIME="${DEFAULT_MAX_TIME}"
LANG_HINT_OVERRIDE=""
VERBOSE_MODE=0
RAW_OUTPUT_MODE=0
OUTPUT_TO_STDOUT=0
INPUT_FROM_STDIN=0
FORCE_OVERWRITE=0
RESPONSE_BODY_FILE=""
TEMP_DIR=""
# Config vars loaded later
CONFIG_GEMINI_API_KEY=""
CONFIG_API_MODEL=""
CONFIG_API_TEMPERATURE=""
cleanup_ran=0

# === Utility Functions ===

verbose_log() {
    (( VERBOSE_MODE )) && echo -e "${COLOR_CYAN}# $(date '+%T') Verbose: ${1}${reset_color}" >&2
}

usage() {
    printf "%b" "${COLOR_BOLD_CYAN}Pyrmethus Code Enhancer - Termux Script Transmutation Spell${reset_color}\n\n"
    printf "%b" "${COLOR_YELLOW}Usage:${reset_color} $0 [-i <input|-|path>] [-o <output|-|path>] [OPTIONS]\n\n"
    printf "%b" "${COLOR_GREEN}Input/Output Arguments:${reset_color}\n"
    printf "  %-20s %s\n" "-i, --input" "Input script path or '-' for stdin (default: stdin)."
    printf "  %-20s %s\n" "-o, --output" "Output artifact path or '-' for stdout (default: stdout)."
    printf "\n%b" "${COLOR_GREEN}Configuration Options:${reset_color}\n"
    printf "  %-20s %s\n" "-k, --key" "Gemini API Key (prefer env var or config file)."
    printf "  %-20s %s\n" " " "${COLOR_BOLD_RED}⚠️ Avoid -k in scripts/shared environments!${reset_color}"
    printf "  %-20s %s\n" "-m, --model" "Gemini model (default: ${DEFAULT_API_MODEL})."
    printf "  %-20s %s\n" "-t, --temperature" "API temperature (${TEMP_MIN}-${TEMP_MAX}, default: ${DEFAULT_API_TEMPERATURE})."
    printf "\n%b" "${COLOR_GREEN}Other Options:${reset_color}\n"
    printf "  %-20s %s\n" "-l, --lang" "Language hint (e.g., 'python', 'bash')."
    printf "  %-20s %s\n" "-f, --force" "Force overwrite existing output file."
    printf "  %-20s %s\n" "--connect-timeout" "Curl connect timeout (seconds, default: ${DEFAULT_CONNECT_TIMEOUT})."
    printf "  %-20s %s\n" "--max-time" "Curl max time (seconds, default: ${DEFAULT_MAX_TIME})."
    printf "  %-20s %s\n" "--retries" "API retry attempts (default: ${MAX_RETRIES})."
    printf "  %-20s %s\n" "--retry-delay" "Seconds between retries (default: ${RETRY_DELAY})."
    printf "  %-20s %s\n" "--raw" "Output raw API response text."
    printf "  %-20s %s\n" "--stdout" "Force output to stdout."
    printf "  %-20s %s\n" "-v, --verbose" "Enable verbose logging."
    printf "  %-20s %s\n" "-h, --help" "Show this help."
    printf "\n%b" "${COLOR_CYAN}Configuration File:${reset_color} ${PYRMETHUS_CONFIG_FILE}\n"
    printf "  %s\n" "Format: KEY=VALUE (e.g., GEMINI_API_KEY=your_key)"
    printf "\n%b" "${COLOR_CYAN}Environment Variables:${reset_color}\n"
    printf "  %s\n" "GEMINI_API_KEY, API_MODEL_ENV, API_TEMPERATURE_ENV, LANG_HINT_OVERRIDE_ENV, PYRMETHUS_TMPDIR, TMPDIR"
    printf "\n%b" "${COLOR_CYAN}Examples:${reset_color}\n"
    printf "  %s\n" "$0 -i script.py -o enhanced.py"
    printf "  %s\n" "cat script.py | $0 -o enhanced.py"
    printf "  %s\n" "$0 -i script.sh --lang bash --stdout"
    exit 0
}

error_exit() {
    local exit_code="${2:-1}"
    echo -e "${COLOR_BOLD_RED}❌ Error: ${1}${reset_color}" >&2
    # Check if caller is signal_handler to avoid recursive cleanup
    # zsh uses funcstack; funcstack[1] is current, funcstack[2] is caller.
    [[ -z "${funcstack[2]}" || "${funcstack[2]}" != "signal_handler" ]] && cleanup
    # Validate exit code is numeric
    [[ "$exit_code" =~ '^[0-9]+$' ]] || exit_code=1
    exit "$exit_code"
}

cleanup() {
    (( cleanup_ran )) && return
    cleanup_ran=1
    verbose_log "Cleaning up..."
    [[ -n "$RESPONSE_BODY_FILE" && -f "$RESPONSE_BODY_FILE" ]] && {
        verbose_log "Removing temporary file: $RESPONSE_BODY_FILE"
        rm -f "$RESPONSE_BODY_FILE" 2>/dev/null
    }
    verbose_log "Cleanup completed."
}

# Use TRAP<SIGNAL> functions in zsh if preferred, or trap command
signal_handler() {
    local signal_name
    # $1 is the signal name (e.g., INT, TERM) when called by trap
    case "$1" in
        HUP) signal_name="SIGHUP" ;;
        INT) signal_name="SIGINT (Ctrl+C)" ;;
        QUIT) signal_name="SIGQUIT" ;;
        TERM) signal_name="SIGTERM" ;;
        *) signal_name="Signal $1" ;;
    esac
    printf "\n%b⚠️ Interrupted by %s. Cleaning up...%b\n" "${COLOR_BOLD_YELLOW}" "${signal_name}" "${reset_color}" >&2
    # Calculate exit code based on signal number (zsh provides signal number in $?)
    # This might be complex; using a fixed offset like 128 + signum is common but requires mapping name back to number
    # For simplicity, let's use a generic non-zero code on interrupt if number isn't easy
    # However, trap passes the signal *name* not number in zsh, unlike bash.
    # Let's stick to the original bash-like calculation for consistency if possible.
    local signum
    case "$1" in
      HUP) signum=1 ;; INT) signum=2 ;; QUIT) signum=3 ;; TERM) signum=15 ;; *) signum=0 ;; # Unknown signal
    esac
    error_exit "Interrupted by ${signal_name}." $((128 + signum))
}

check_tools() {
    verbose_log "Checking for required tools..."
    local missing_tools=""
    for tool in curl jq dirname mkdir sleep mktemp; do
        command -v "$tool" >/dev/null 2>&1 || missing_tools+=" $tool"
    done
    [[ -n "$missing_tools" ]] && error_exit "Missing tools:$missing_tools. Install with: pkg install$missing_tools"
    echo '{}' | jq '.' >/dev/null 2>&1 || error_exit "jq found but non-functional."
    verbose_log "All required tools are present."
}

resolve_temp_dir() {
    # Use an array for potential directories
    local -a potential_dirs
    potential_dirs=( "${PYRMETHUS_TMPDIR:-}" "${TMPDIR:-}" "/data/data/com.termux/files/usr/tmp" "/tmp" "." )
    for dir in "${potential_dirs[@]}"; do
        # Check dir exists, is directory, and is writable
        if [[ -n "$dir" && -d "$dir" && -w "$dir" ]]; then
            TEMP_DIR="$dir"
            verbose_log "Using temp directory: $TEMP_DIR"
            return 0 # Success
        fi
    done
    error_exit "No writable temporary directory found. Set PYRMETHUS_TMPDIR or TMPDIR."
}

load_config() {
    verbose_log "Loading configuration from ${PYRMETHUS_CONFIG_FILE}..."
    local cfg_api_key="" cfg_api_model="" cfg_api_temp=""
    if [[ -f "$PYRMETHUS_CONFIG_FILE" && -r "$PYRMETHUS_CONFIG_FILE" ]]; then
        # Use read loop compatible with zsh
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Remove comments and trim whitespace (using parameter expansion)
            line="${line%%#*}" # Remove comment
            line="${line##[[:space:]]}" # Trim leading whitespace (zsh specific pattern)
            line="${line%%[[:space:]]}" # Trim trailing whitespace (zsh specific pattern)
            [[ -z "$line" ]] && continue
            # zsh regex matching: =~ 'pattern'. Access captures with $match array.
            if [[ "$line" =~ '^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$' ]]; then
                local key="${match[1]}"
                local value="${match[2]}"
                # Remove surrounding quotes from value if present
                value="${value#[\'\"]}"
                value="${value%[\'\"]}"
                case "$key" in
                    GEMINI_API_KEY) cfg_api_key="$value" ;;
                    API_MODEL) cfg_api_model="$value" ;;
                    API_TEMPERATURE) cfg_api_temp="$value" ;;
                    *) verbose_log "Ignoring unknown config key: $key" ;;
                esac
            fi
        done < "$PYRMETHUS_CONFIG_FILE"
        verbose_log "Configuration loaded."
    fi
    CONFIG_GEMINI_API_KEY="$cfg_api_key"
    CONFIG_API_MODEL="$cfg_api_model"
    CONFIG_API_TEMPERATURE="$cfg_api_temp"
}

parse_arguments() {
    # Keep local variables as is
    local cli_input_file="" cli_output_file="" cli_api_key="" cli_api_model="" cli_api_temp="" cli_lang_hint=""
    local cli_conn_timeout="" cli_max_time="" cli_retries="$MAX_RETRIES" cli_retry_delay="$RETRY_DELAY" cli_force=0
    # Zsh built-in getopts handles only short options
    # We stick to the getopt external command approach for long options if available

    # Need zsh specific module for getopt? No, external command.
    # Check for GNU getopt first
    if command -v getopt >/dev/null 2>&1 && getopt --test > /dev/null 2>&1; then
        local options="i:o:k:m:t:l:hvf"
        local long_options="input:,output:,key:,model:,temperature:,lang:,connect-timeout:,max-time:,retries:,retry-delay:,help,verbose,raw,stdout,force"
        local parsed
        # Need to handle potential errors from getopt itself
        if ! parsed=$(getopt --options "$options" --longoptions "$long_options" --name "$0" -- "$@"); then
             echo -e "${COLOR_RED}Argument parsing failed (getopt error).${reset_color}" >&2
             usage # usage exits
        fi
        # Use zsh's native array splitting instead of eval set --
        local -a args
        args=( ${(z)parsed} ) # z flag splits like sh words
        set -- "${args[@]}" # Replace positional parameters with getopt output
        while true; do
            case "$1" in
                -i|--input) cli_input_file="$2"; shift 2 ;;
                -o|--output) cli_output_file="$2"; shift 2 ;;
                -k|--key) cli_api_key="$2"; shift 2 ;;
                -m|--model) cli_api_model="$2"; shift 2 ;;
                -t|--temperature) cli_api_temp="$2"; shift 2 ;;
                -l|--lang) cli_lang_hint="$2"; shift 2 ;;
                --connect-timeout) cli_conn_timeout="$2"; shift 2 ;;
                --max-time) cli_max_time="$2"; shift 2 ;;
                --retries) cli_retries="$2"; shift 2 ;;
                --retry-delay) cli_retry_delay="$2"; shift 2 ;;
                --raw) RAW_OUTPUT_MODE=1; shift ;;
                --stdout) OUTPUT_TO_STDOUT=1; shift ;;
                -f|--force) cli_force=1; shift ;;
                -v|--verbose) VERBOSE_MODE=1; shift ;;
                -h|--help) usage ;;
                --) shift; break ;;
                 *) error_exit "Invalid argument detected by getopt: $1" ;; # Should not happen with getopt
            esac
        done
    else
        # Fallback to zsh getopts (handles short options only)
        echo -e "${COLOR_YELLOW}⚠️ GNU getopt not found, using built-in getopts (long options ignored).${reset_color}" >&2
        # Reset OPTIND for getopts
        OPTIND=1
        # Make OPTARG local to the loop
        while getopts ":i:o:k:m:t:l:hvf" opt; do
            local local_optarg="$OPTARG" # Capture OPTARG immediately
            case ${opt} in
                i) cli_input_file="${local_optarg}" ;;
                o) cli_output_file="${local_optarg}" ;;
                k) cli_api_key="${local_optarg}" ;;
                m) cli_api_model="${local_optarg}" ;;
                t) cli_api_temp="${local_optarg}" ;;
                l) cli_lang_hint="${local_optarg}" ;;
                v) VERBOSE_MODE=1 ;;
                f) cli_force=1 ;;
                h) usage ;;
                \?) error_exit "Invalid option: -${local_optarg}" ;;
                :) error_exit "Option -${local_optarg} requires an argument." ;;
            esac
        done
        shift $((OPTIND - 1))
        # Manual handling for boolean long options in fallback mode
        for arg in "$@"; do
            case "$arg" in
                 # Add any boolean long options here if needed for basic functionality
                 --raw) RAW_OUTPUT_MODE=1 ;;
                 --stdout) OUTPUT_TO_STDOUT=1 ;;
                 --force) cli_force=1 ;;
                 # Ignore other long options in getopts mode
                 --*) ;;
                 # Anything else is an unexpected argument
                 *) echo -e "${COLOR_RED}Unexpected argument in getopts mode: $arg${reset_color}" >&2; usage ;;
            esac
        done
        # Clear remaining arguments if they were just flags we processed
        # This assumes non-flag arguments are errors in getopts mode
        # Adjust if positional arguments are expected after options
        local -a remaining_args=("$@")
        local -a non_flag_args
        for arg in "${remaining_args[@]}"; do
           if [[ "$arg" != "--raw" && "$arg" != "--stdout" && "$arg" != "--force" ]]; then
               non_flag_args+=("$arg")
           fi
        done
        set -- "${non_flag_args[@]}" # Update positional parameters
    fi

    # Check for unexpected positional arguments after parsing
    [[ $# -gt 0 ]] && { echo -e "${COLOR_RED}Unexpected arguments: $@${reset_color}" >&2; usage; }

    # Set global variables from CLI, ENV, Config, or Defaults
    GEMINI_API_KEY="${cli_api_key:-${GEMINI_API_KEY:-${CONFIG_GEMINI_API_KEY:-}}}"
    API_MODEL="${cli_api_model:-${API_MODEL_ENV:-${CONFIG_API_MODEL:-${DEFAULT_API_MODEL}}}}"
    API_TEMPERATURE="${cli_api_temp:-${API_TEMPERATURE_ENV:-${CONFIG_API_TEMPERATURE:-${DEFAULT_API_TEMPERATURE}}}}"
    LANG_HINT_OVERRIDE="${cli_lang_hint:-${LANG_HINT_OVERRIDE_ENV:-}}"
    # Use provided values if set, otherwise default (already set globally)
    [[ -n "$cli_conn_timeout" ]] && CONNECT_TIMEOUT="$cli_conn_timeout"
    [[ -n "$cli_max_time" ]] && MAX_TIME="$cli_max_time"
    MAX_RETRIES="${cli_retries}" # Already defaults to constant
    RETRY_DELAY="${cli_retry_delay}" # Already defaults to constant
    FORCE_OVERWRITE="${cli_force}" # Already defaults to 0

    # Handle input file logic
    if [[ -z "$cli_input_file" || "$cli_input_file" == "-" ]]; then
        INPUT_FROM_STDIN=1
        INPUT_FILE_PATH="-" # Represent stdin
        verbose_log "Input set to stdin."
    else
        INPUT_FROM_STDIN=0
        INPUT_FILE_PATH="$cli_input_file"
        verbose_log "Input file set to '$INPUT_FILE_PATH'."
    fi

    # Handle output file logic
    if (( OUTPUT_TO_STDOUT )) || [[ -z "$cli_output_file" || "$cli_output_file" == "-" ]]; then
        OUTPUT_TO_STDOUT=1
        OUTPUT_FILE_NAME="-" # Represent stdout
        verbose_log "Output set to stdout."
    else
        OUTPUT_TO_STDOUT=0
        OUTPUT_FILE_NAME="$cli_output_file"
        verbose_log "Output file set to '$OUTPUT_FILE_NAME'."
    fi

    # Validate numeric inputs (using zsh =~ and awk)
    # Regex needs to be anchored
    local numeric_regex='^[0-9]+([.][0-9]+)?$' integer_regex='^[0-9]+$'
    [[ ! "$API_TEMPERATURE" =~ $numeric_regex ]] && error_exit "Temperature '$API_TEMPERATURE' must be a number."
    # Use awk for float comparison
    awk -v temp="$API_TEMPERATURE" -v min="$TEMP_MIN" -v max="$TEMP_MAX" 'BEGIN { exit !(temp >= min && temp <= max) }' || \
        error_exit "Temperature '$API_TEMPERATURE' must be between $TEMP_MIN and $TEMP_MAX."
    [[ ! "$CONNECT_TIMEOUT" =~ $integer_regex ]] && error_exit "Connect timeout '$CONNECT_TIMEOUT' must be an integer."
    [[ ! "$MAX_TIME" =~ $integer_regex ]] && error_exit "Max time '$MAX_TIME' must be an integer."
    [[ ! "$MAX_RETRIES" =~ $integer_regex ]] && error_exit "Retries '$MAX_RETRIES' must be an integer."
    [[ ! "$RETRY_DELAY" =~ $integer_regex ]] && error_exit "Retry delay '$RETRY_DELAY' must be an integer."

    verbose_log "Settings: Model='$API_MODEL', Temp='$API_TEMPERATURE', LangHint='$LANG_HINT_OVERRIDE', Retries=$MAX_RETRIES, Delay=$RETRY_DELAY, Force=$FORCE_OVERWRITE"
}


resolve_and_check_api_key() {
    verbose_log "Checking API key..."
    [[ -z "$GEMINI_API_KEY" ]] && {
        printf "%b" "${COLOR_BOLD_RED}--------------------------------------------------------------------
🔒 Error: GEMINI_API_KEY not set.
   Use -k, set GEMINI_API_KEY env var, or add to '${PYRMETHUS_CONFIG_FILE}'.
--------------------------------------------------------------------${reset_color}\n" >&2
        return 1 # Use return code for check functions
    }
    # Use parameter expansion for length check
    (( ${#GEMINI_API_KEY} < 30 )) && \
        echo -e "${COLOR_YELLOW}⚠️ Warning: API Key seems short (< 30 chars).${reset_color}" >&2
    # Check for placeholder values (case insensitive using :l)
    case "${GEMINI_API_KEY:l}" in
        *your_*|*actual_*|*api_key*)
             echo -e "${COLOR_YELLOW}⚠️ Warning: API Key looks like a placeholder.${reset_color}" >&2 ;;
    esac
    verbose_log "API key present."
    return 0
}


validate_paths() {
    # Use local for parameters for clarity
    local input_path="$1" output_path="$2" input_is_stdin="$3" output_is_stdout="$4" force_overwrite="$5"

    if (( ! input_is_stdin )); then # Use arithmetic evaluation for boolean check
        verbose_log "Validating input: '$input_path'..."
        [[ ! -f "$input_path" ]] && error_exit "Input file '$input_path' not found."
        [[ ! -r "$input_path" ]] && error_exit "Input file '$input_path' not readable."
        # Check if file is empty using -s
        [[ ! -s "$input_path" ]] && error_exit "Input file '$input_path' is empty."
    else
        # Check if stdin is connected to a terminal
        # -t 0 checks if FD 0 is a TTY
        [[ -t 0 ]] && error_exit "Stdin is a terminal. Pipe input or use -i <file>."
    fi

    if (( ! output_is_stdout )); then
        verbose_log "Validating output: '$output_path'..."
        # Use parameter expansion for dirname :h
        local output_dir="${output_path:h}"
        # Handle cases where dirname might be empty or '.'
        [[ -z "$output_dir" ]] && output_dir="."
        [[ ! -d "$output_dir" ]] && error_exit "Output directory '$output_dir' does not exist."
        [[ ! -w "$output_dir" ]] && error_exit "Output directory '$output_dir' not writable."

        # Check if output path exists
        if [[ -e "$output_path" ]]; then
            # Ensure it's a regular file if it exists
            [[ ! -f "$output_path" ]] && error_exit "Output path '$output_path' exists but is not a regular file."
            # Use arithmetic evaluation for force_overwrite
            if (( force_overwrite )); then
                verbose_log "Forcing overwrite of '$output_path'."
            elif [[ ! -w "$output_path" ]]; then # Check writability only if not forcing and file exists
                error_exit "Output file '$output_path' exists and is not writable."
            else
                echo -e "${COLOR_BOLD_YELLOW}⚠️ Output file '$output_path' exists.${reset_color}" >&2
                # Use zsh's read -q for simple y/n confirmation
                # The prompt needs careful quoting for echo -e and read
                local prompt_str="${COLOR_BLUE}Overwrite? (y/N): ${reset_color}"
                # Print the prompt separately without newline
                print -n "$prompt_str" > /dev/tty
                # read -k 1 reads one character, -s hides input
                # read -q checks for 'y' or 'Y' and returns 0, else 1
                if ! read -q confirm_overwrite < /dev/tty; then
                    # read -q returns 1 if not y/Y
                    echo # Add a newline after the prompt
                    echo -e "${COLOR_MAGENTA}Aborted.${reset_color}" >&2
                    exit 0
                 fi
                 echo # Add a newline after the prompt
                 verbose_log "Overwrite confirmed."
            fi
        fi
    fi
}

prepare_code_and_prompt() {
    local input_path="$1" output_name="$2" input_is_stdin="$3"
    local lang_hint="" filename="stdin" original_code_content=""

    echo -e "${COLOR_BOLD_CYAN}--- Preparing Code and Prompt ---${reset_color}" >&2

    if (( input_is_stdin )); then
        # Read from stdin into variable using cat
        original_code_content=$(cat) || error_exit "Failed to read from stdin."
        [[ -z "$original_code_content" ]] && error_exit "Stdin was empty."
    else
        # Use parameter expansion for filename :t (tail/basename)
        filename="${input_path:t}"
        # Use zsh redirection for reading file content
        original_code_content=$(<"$input_path") || error_exit "Failed to read '$input_path'."
    fi
    # Use parameter expansion for length
    verbose_log "Read ${#original_code_content} bytes from ${filename}."

    if [[ -n "$LANG_HINT_OVERRIDE" ]]; then
        lang_hint="$LANG_HINT_OVERRIDE"
    elif (( ! input_is_stdin )); then
        # Use parameter expansion for extension :e
        local file_ext="${input_path:e}"
        case "$file_ext" in
            py) lang_hint="python" ;;
            js) lang_hint="javascript" ;;
            sh|bash) lang_hint="bash" ;;
            *) lang_hint="" ;; # Default to empty if no match
        esac
    fi
    verbose_log "Language hint: ${lang_hint:-none}."

    # Use print -r -- to build the prompt safely
    # Using a temporary variable avoids issues with EOF markers inside the string itself
    local prompt_template="Analyze and enhance the following code from %s. Improve readability, resilience, efficiency, security, and use modern practices (%s).

CRITICAL: Respond *only* with the complete, enhanced code block, ready for %s. Exclude preamble, explanations, or markdown outside the code. Use inline comments only for essential clarity.

Original Code:
\`\`\`%s
%s
\`\`\`"

    local lang_display="${lang_hint:-language inferred}"
    local lang_fence="${lang_hint}" # Use actual hint for fence, empty if none
    local prompt_content # Declare local var

    # Use printf to format the prompt safely into the variable
    printf -v prompt_content "$prompt_template" \
        "$filename" "$lang_display" "$output_name" "$lang_fence" "$original_code_content" \
        || error_exit "Failed to format prompt."

    verbose_log "Prompt prepared."
    # Output the final prompt content
    print -r -- "$prompt_content"
}


call_gemini_api() {
    local prompt_content="$1"
    local api_url="${API_BASE_URL}/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"
    local attempt=0

    echo -e "${COLOR_BOLD_CYAN}--- Calling Gemini API ($API_MODEL) ---${reset_color}" >&2
    verbose_log "API URL: $api_url"

    local json_payload
    # Use print -r -- piped to jq for safety
    json_payload=$(print -r -- "$prompt_content" | jq -R -s \
        --argjson temp_num "$API_TEMPERATURE" \
        '{"contents": [{"parts": [{"text": .}]}], "generationConfig": {"temperature": $temp_num}}') || \
        error_exit "Failed to create JSON payload."

    # Ensure TEMP_DIR is set and writable before mktemp
    [[ -z "$TEMP_DIR" || ! -d "$TEMP_DIR" || ! -w "$TEMP_DIR" ]] && resolve_temp_dir
    RESPONSE_BODY_FILE=$(mktemp "${TEMP_DIR}/gemini_response.XXXXXX") || error_exit "Failed to create temp file in $TEMP_DIR."
    chmod 600 "$RESPONSE_BODY_FILE"
    verbose_log "Temp file: $RESPONSE_BODY_FILE"

    # Use arithmetic loop in zsh
    while (( attempt <= MAX_RETRIES )); do
        (( attempt++ ))
        verbose_log "Attempt $attempt/$((MAX_RETRIES + 1))..."

        local http_code
        # Capture curl exit code separately
        local curl_exit_code=0
        # Run curl and capture output code; || sets curl_exit_code on non-zero exit
        http_code=$(curl --fail -L -s -w '%{http_code}' \
            --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
            -H 'Content-Type: application/json' \
            -d "$json_payload" \
            -X POST "$api_url" \
            -o "$RESPONSE_BODY_FILE") || curl_exit_code=$?

        # Check both curl exit code and HTTP status code
        if [[ $curl_exit_code -eq 0 && "$http_code" -eq 200 ]]; then
            echo -e "${COLOR_GREEN}✅ API call successful (HTTP $http_code).${reset_color}" >&2
            return 0 # Success
        fi

        echo -e "${COLOR_YELLOW}⚠️ Attempt $attempt failed (Curl Exit: $curl_exit_code, HTTP: $http_code).${reset_color}" >&2
        # Parse error message regardless of retry decision
        parse_api_error_message "$RESPONSE_BODY_FILE" "$http_code"

        # Decide whether to retry: non-zero curl exit, server error (>=500), or rate limit (429)
        if (( attempt <= MAX_RETRIES )) && { [[ $curl_exit_code -ne 0 ]] || (( http_code >= 500 )) || (( http_code == 429 )); }; then
            echo -e "${COLOR_YELLOW}Retrying in ${RETRY_DELAY}s...${reset_color}" >&2
            sleep "$RETRY_DELAY"
        else
            # No more retries or non-retryable error
            error_exit "API call failed after $attempt attempts."
        fi
    done
    # Should not be reached if error_exit works, but good practice
    error_exit "API call failed after exhausting retries."
}

parse_api_error_message() {
    local response_file="$1" http_code="$2"
    # Check if file exists and is readable and not empty
    [[ ! -s "$response_file" ]] && { verbose_log "No response body for error parsing."; return; }

    # Check if the response is valid JSON *before* trying to parse specific fields
    if ! jq -e . "$response_file" >/dev/null 2>&1; then
        echo -e "${COLOR_YELLOW}--- Non-JSON Error Response (HTTP $http_code) ---${reset_color}" >&2
        # Use cat -v to show non-printable characters potentially
        cat -v "$response_file" >&2
        echo -e "${COLOR_YELLOW}------------------------------------------${reset_color}" >&2
        return
    fi

    # Try to extract the error message
    local error_message
    # Use jq's try-catch equivalent or default value // ""
    error_message=$(jq -r '.error.message // ""' "$response_file") || error_message="(jq failed to parse error)"

    if [[ -n "$error_message" && "$error_message" != "(jq failed to parse error)" ]]; then
        echo -e "${COLOR_YELLOW}--- API Error (HTTP $http_code) ---${reset_color}" >&2
        echo -e "${COLOR_RED}${error_message}${reset_color}" >&2
        # Provide suggestions based on error message content or HTTP code
        case "${error_message:l}" in # Use lowercase for matching :l
            *api*key*not*valid*) echo -e "${COLOR_BOLD_RED}Suggestion: Verify GEMINI_API_KEY.${reset_color}" >&2 ;;
            *quota* | *rate*limit*) echo -e "${COLOR_BOLD_RED}Suggestion: Check API quota or rate limits.${reset_color}" >&2 ;;
            *permission*denied*) echo -e "${COLOR_BOLD_RED}Suggestion: Check API key permissions.${reset_color}" >&2 ;;
            *invalid*argument*) echo -e "${COLOR_BOLD_RED}Suggestion: Check request payload/parameters.${reset_color}" >&2 ;;
            *) # Check HTTP codes if message is generic
                case "$http_code" in
                   400) echo -e "${COLOR_BOLD_RED}Suggestion: Check request syntax (Bad Request).${reset_color}" >&2 ;;
                   401|403) echo -e "${COLOR_BOLD_RED}Suggestion: Check API key/authentication (Unauthorized/Forbidden).${reset_color}" >&2 ;;
                   429) echo -e "${COLOR_BOLD_RED}Suggestion: Check API quota or rate limits (Too Many Requests).${reset_color}" >&2 ;;
                esac
                ;;
        esac
        echo -e "${COLOR_YELLOW}--------------------------------${reset_color}" >&2
    else
        # No specific .error.message, but it was JSON. Log the structure maybe?
        verbose_log "API returned JSON, but no '.error.message' found. Full error JSON:"
        # Log full JSON if verbose
        (( VERBOSE_MODE )) && verbose_log "$(jq '.' "$response_file")"
    fi
}

process_response() {
    [[ -z "$RESPONSE_BODY_FILE" || ! -r "$RESPONSE_BODY_FILE" ]] && error_exit "Response file missing or unreadable."

    echo -e "${COLOR_BOLD_CYAN}--- Processing Response ---${reset_color}" >&2
    verbose_log "Processing: $RESPONSE_BODY_FILE"

    # Log raw response if verbose
    (( VERBOSE_MODE )) && {
        verbose_log "--- Raw Response ---"
        cat "$RESPONSE_BODY_FILE" >&2
        verbose_log "--- End Raw Response ---"
    }

    # Validate JSON structure rigorously
    jq -e . >/dev/null 2>&1 "$RESPONSE_BODY_FILE" || error_exit "Response is not valid JSON."
    # Check for candidates array
    jq -e 'has("candidates") and (.candidates | type == "array")' "$RESPONSE_BODY_FILE" >/dev/null 2>&1 || {
        # Try to get more specific error info if candidates are missing
        local error_info
        error_info=$(jq -r '.error // .promptFeedback // {message:"Unknown structure"}' "$RESPONSE_BODY_FILE")
        error_exit "Response missing 'candidates' array. Info: $error_info"
    }
    # Check if candidates array is non-empty
    jq -e '.candidates | length > 0' "$RESPONSE_BODY_FILE" >/dev/null 2>&1 || {
        local block_reason feedback
        # Check for prompt feedback block reason first
        block_reason=$(jq -r '.promptFeedback.blockReason // ""' "$RESPONSE_BODY_FILE") || block_reason=""
        if [[ -n "$block_reason" ]]; then
             feedback=$(jq -r '.promptFeedback.safetyRatings // ""' "$RESPONSE_BODY_FILE")
             error_exit "Prompt blocked: ${block_reason}. Feedback: ${feedback:-N/A}"
        else
             # Maybe the API just returned empty candidates for other reasons
             error_exit "No candidates found in API response."
        fi
    }

    # Extract finish reason (handle potential null or missing key)
    local finish_reason
    finish_reason=$(jq -r '.candidates[0].finishReason // "MISSING"' "$RESPONSE_BODY_FILE") || finish_reason="READ_ERROR"
    verbose_log "Finish Reason: ${finish_reason}"
    case "$finish_reason" in
        STOP) ;; # Expected success
        MAX_TOKENS) echo -e "${COLOR_YELLOW}⚠️ Output may be truncated (finishReason: MAX_TOKENS).${reset_color}" >&2 ;;
        SAFETY|RECITATION)
            local safety_info
            safety_info=$(jq -r '.candidates[0].safetyRatings // ""' "$RESPONSE_BODY_FILE")
            error_exit "Response generation blocked: ${finish_reason}. Info: ${safety_info:-N/A}" ;;
        OTHER) echo -e "${COLOR_YELLOW}⚠️ Response generation finished due to unspecified reason (OTHER).${reset_color}" >&2 ;;
        MISSING|READ_ERROR) echo -e "${COLOR_YELLOW}⚠️ Finish reason missing or unreadable in response.${reset_color}" >&2 ;;
        *) echo -e "${COLOR_YELLOW}⚠️ Unexpected finish reason: ${finish_reason}.${reset_color}" >&2 ;;
    esac

    # Extract content text safely using // "" default
    local raw_final_content
    raw_final_content=$(jq -r '.candidates[0].content.parts[0].text // ""' "$RESPONSE_BODY_FILE") || \
        error_exit "Failed to extract text content from response part."
    # Allow empty content? If not, uncomment below.
    # [[ -z "$raw_final_content" ]] && error_exit "Extracted text content is empty."

    if (( RAW_OUTPUT_MODE )); then
        echo -e "${COLOR_CYAN}ℹ️ Raw output mode: returning full extracted text.${reset_color}" >&2
        print -r -- "$raw_final_content" # Use print -r -- for safe output
        return 0 # Indicate success
    fi

    # Extract code block using awk (seems robust enough)
    local final_code_content
    # Ensure awk gets the content correctly, use print -r --
    # The awk script extracts content between the first ``` optionally followed by lang and the next ```
    final_code_content=$(print -r -- "$raw_final_content" | \
        awk '/^[[:space:]]*```([a-zA-Z0-9_.-]*[[:space:]]*)?$/{if(!in_block){in_block=1;next}else{exit}} in_block{print}')

    # Handle cases where awk extraction might fail or not find blocks
    if [[ -z "$final_code_content" ]]; then
        # Check if triple backticks were present at all using grep -q
        if print -r -- "$raw_final_content" | grep -q '```'; then
            echo -e "${COLOR_YELLOW}⚠️ Failed to extract fenced code block (```). Using full response text.${reset_color}" >&2
        else
            echo -e "${COLOR_CYAN}ℹ️ No code fences (```) detected. Using full response text as code.${reset_color}" >&2
        fi
        # Fallback to the raw content if no block extracted
        final_code_content="$raw_final_content"
    fi

    # Output the processed content
    print -r -- "$final_code_content"
}

handle_output() {
    local content="$1" output_file="$2" output_is_stdout="$3"

    if (( output_is_stdout )); then
        verbose_log "Outputting to stdout."
        print -r -- "$content" # Use print -r -- for safe output to stdout
    else
        echo -e "${COLOR_BOLD_CYAN}--- Saving Output ---${reset_color}" >&2
        # Use print -r -- > redirection for safe writing to file
        print -r -- "$content" > "$output_file" || error_exit "Failed to write to '$output_file'."
        # Try realpath, fall back to constructing path
        local final_path
        if command -v realpath >/dev/null 2>&1; then
             # Handle potential errors from realpath (e.g., file disappears)
             final_path=$(realpath "$output_file" 2>/dev/null) || final_path="$output_file (realpath failed)"
        else
             # Construct path relative to PWD if not absolute
             # zsh parameter expansion :A resolves path
             final_path="${output_file:A}"
        fi
        echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------" >&2
        echo -e "✅ Saved to: ${COLOR_BOLD_YELLOW}${final_path}${reset_color}" # Removed quotes around path for clarity
        echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------${reset_color}" >&2
    fi
}

# === Main Execution ===

# Setup traps for common signals
trap 'signal_handler HUP' HUP
trap 'signal_handler INT' INT
trap 'signal_handler QUIT' QUIT
trap 'signal_handler TERM' TERM
# zshexit function is called automatically on normal exit or via `exit` command
zshexit() {
    # Ensure cleanup runs only once
    (( ! cleanup_ran )) && cleanup
}

echo -e "${COLOR_MAGENTA}### Pyrmethus Code Enhancer (zsh) ###${reset_color}" >&2

# Initial setup
resolve_temp_dir # Ensure TEMP_DIR is set early
# Use mkdir -p which is idempotent and handles intermediate dirs
mkdir -p "$PYRMETHUS_CONFIG_DIR" 2>/dev/null || verbose_log "Could not create config dir '$PYRMETHUS_CONFIG_DIR' (may exist or permissions issue)."
load_config
parse_arguments "$@" # Pass all arguments to parser
(( VERBOSE_MODE )) && verbose_log "Verbose mode enabled." # Log after parsing
check_tools
resolve_and_check_api_key || exit 1 # Exit if key check fails (returns non-zero)
validate_paths "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME" "$INPUT_FROM_STDIN" "$OUTPUT_TO_STDOUT" "$FORCE_OVERWRITE"

# Core logic - use local variables for intermediate results
local prompt_text
prompt_text=$(prepare_code_and_prompt "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME" "$INPUT_FROM_STDIN")
# API call function handles its own errors/retries/exit
call_gemini_api "$prompt_text"
# Process response function handles its own errors/exit
local final_content
final_content=$(process_response)
# Handle output function handles its own errors/exit
handle_output "$final_content" "$OUTPUT_FILE_NAME" "$OUTPUT_TO_STDOUT"

echo -e "${COLOR_MAGENTA}### Completed ###${reset_color}" >&2
# No need to explicitly call cleanup here, zshexit handles it
# Exit successfully - zshexit will run before the shell terminates
exit 0