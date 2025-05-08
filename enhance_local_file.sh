#!/usr/bin/env bash
# set -e: Halt on any error
# set -o pipefail: Ensure pipeline errors propagate
# set -u: Treat unset variables as errors
set -euo pipefail

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
declare -g CONFIG_GEMINI_API_KEY="" CONFIG_API_MODEL="" CONFIG_API_TEMPERATURE=""
declare -g cleanup_ran=0

# === Utility Functions ===

verbose_log() {
    [[ "$VERBOSE_MODE" -eq 1 ]] && echo -e "${COLOR_CYAN}# $(date '+%T') Verbose: ${1}${reset_color}" >&2
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
    [[ "${FUNCNAME[1]}" != "signal_handler" ]] && cleanup
    [[ "$exit_code" =~ ^[0-9]+$ ]] || exit_code=1
    exit "$exit_code"
}

cleanup() {
    [[ -n "${cleanup_ran:-}" ]] && return
    cleanup_ran=1
    verbose_log "Cleaning up..."
    [[ -n "$RESPONSE_BODY_FILE" && -f "$RESPONSE_BODY_FILE" ]] && {
        verbose_log "Removing temporary file: $RESPONSE_BODY_FILE"
        rm -f "$RESPONSE_BODY_FILE" 2>/dev/null
    }
    verbose_log "Cleanup completed."
}

signal_handler() {
    local signum="$1" signal_name
    case "$signum" in
        1) signal_name="SIGHUP" ;;
        2) signal_name="SIGINT (Ctrl+C)" ;;
        3) signal_name="SIGQUIT" ;;
        15) signal_name="SIGTERM" ;;
        *) signal_name="Signal $signum" ;;
    esac
    printf "\n%b⚠️ Interrupted by %s. Cleaning up...%b\n" "${COLOR_BOLD_YELLOW}" "${signal_name}" "${reset_color}" >&2
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
    for dir in "${PYRMETHUS_TMPDIR:-}" "${TMPDIR:-}" "/data/data/com.termux/files/usr/tmp" "/tmp" "."; do
        if [[ -n "$dir" && -d "$dir" && -w "$dir" ]]; then
            TEMP_DIR="$dir"
            verbose_log "Using temp directory: $TEMP_DIR"
            return
        fi
    done
    error_exit "No writable temporary directory found. Set PYRMETHUS_TMPDIR or TMPDIR."
}

load_config() {
    verbose_log "Loading configuration from ${PYRMETHUS_CONFIG_FILE}..."
    local cfg_api_key="" cfg_api_model="" cfg_api_temp=""
    if [[ -f "$PYRMETHUS_CONFIG_FILE" && -r "$PYRMETHUS_CONFIG_FILE" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            line=$(echo "$line" | sed -e 's/^[[:space:]]*#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
            [[ -z "$line" ]] && continue
            if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*)=(.*)$ ]]; then
                case "${BASH_REMATCH[1]}" in
                    GEMINI_API_KEY) cfg_api_key="${BASH_REMATCH[2]//[\'\"]/}" ;;
                    API_MODEL) cfg_api_model="${BASH_REMATCH[2]//[\'\"]/}" ;;
                    API_TEMPERATURE) cfg_api_temp="${BASH_REMATCH[2]//[\'\"]/}" ;;
                    *) verbose_log "Ignoring unknown config key: ${BASH_REMATCH[1]}" ;;
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
    local cli_input_file="" cli_output_file="" cli_api_key="" cli_api_model="" cli_api_temp="" cli_lang_hint=""
    local cli_conn_timeout="" cli_max_time="" cli_retries="$MAX_RETRIES" cli_retry_delay="$RETRY_DELAY" cli_force=0
    local options="i:o:k:m:t:l:hvf"
    local long_options="input:,output:,key:,model:,temperature:,lang:,connect-timeout:,max-time:,retries:,retry-delay:,help,verbose,raw,stdout,force"

    if command -v getopt >/dev/null 2>&1 && getopt --test >/dev/null 2>&1; then
        local parsed
        parsed=$(getopt --options "$options" --longoptions "$long_options" --name "$0" -- "$@") || {
            echo -e "${COLOR_RED}Argument parsing failed.${reset_color}" >&2
            usage
        }
        eval set -- "$parsed"
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
                *) error_exit "Invalid argument: $1" ;;
            esac
        done
    else
        echo -e "${COLOR_YELLOW}⚠️ getopt not found, using basic getopts.${reset_color}" >&2
        while getopts ":i:o:k:m:t:l:hvf" opt; do
            case ${opt} in
                i) cli_input_file="${OPTARG}" ;;
                o) cli_output_file="${OPTARG}" ;;
                k) cli_api_key="${OPTARG}" ;;
                m) cli_api_model="${OPTARG}" ;;
                t) cli_api_temp="${OPTARG}" ;;
                l) cli_lang_hint="${OPTARG}" ;;
                v) VERBOSE_MODE=1 ;;
                f) cli_force=1 ;;
                h) usage ;;
                \?) error_exit "Invalid option: -${OPTARG}" ;;
                :) error_exit "Option -${OPTARG} requires an argument." ;;
            esac
        done
        shift $((OPTIND - 1))
        for arg in "$@"; do
            case "$arg" in
                --raw) RAW_OUTPUT_MODE=1 ;;
                --stdout) OUTPUT_TO_STDOUT=1 ;;
                --force) cli_force=1 ;;
            esac
        done
    fi

    [[ $# -gt 0 ]] && { echo -e "${COLOR_RED}Unexpected arguments: $@${reset_color}" >&2; usage; }

    GEMINI_API_KEY="${cli_api_key:-${GEMINI_API_KEY:-${CONFIG_GEMINI_API_KEY:-}}}"
    API_MODEL="${cli_api_model:-${API_MODEL_ENV:-${CONFIG_API_MODEL:-${DEFAULT_API_MODEL}}}}"
    API_TEMPERATURE="${cli_api_temp:-${API_TEMPERATURE_ENV:-${CONFIG_API_TEMPERATURE:-${DEFAULT_API_TEMPERATURE}}}}"
    LANG_HINT_OVERRIDE="${cli_lang_hint:-${LANG_HINT_OVERRIDE_ENV:-}}"
    CONNECT_TIMEOUT="${cli_conn_timeout:-$DEFAULT_CONNECT_TIMEOUT}"
    MAX_TIME="${cli_max_time:-$DEFAULT_MAX_TIME}"
    MAX_RETRIES="${cli_retries}"
    RETRY_DELAY="${cli_retry_delay}"
    FORCE_OVERWRITE="${cli_force}"

    # Handle input file logic
    if [[ -z "$cli_input_file" || "$cli_input_file" == "-" ]]; then
        INPUT_FROM_STDIN=1
        INPUT_FILE_PATH="-"
        verbose_log "Input set to stdin."
    else
        INPUT_FROM_STDIN=0
        INPUT_FILE_PATH="$cli_input_file"
        verbose_log "Input file set to '$INPUT_FILE_PATH'."
    fi

    # Handle output file logic
    if [[ "$OUTPUT_TO_STDOUT" -eq 1 || -z "$cli_output_file" || "$cli_output_file" == "-" ]]; then
        OUTPUT_TO_STDOUT=1
        OUTPUT_FILE_NAME="-"
        verbose_log "Output set to stdout."
    else
        OUTPUT_TO_STDOUT=0
        OUTPUT_FILE_NAME="$cli_output_file"
        verbose_log "Output file set to '$OUTPUT_FILE_NAME'."
    fi

    # Validate numeric inputs
    local numeric_regex='^[0-9]+([.][0-9]+)?$' integer_regex='^[0-9]+$'
    [[ ! "$API_TEMPERATURE" =~ $numeric_regex ]] && error_exit "Temperature '$API_TEMPERATURE' must be a number."
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
        return 1
    }
    [[ "$GEMINI_API_KEY" == *"YOUR_"* || "$GEMINI_API_KEY" == *"ACTUAL_"* || "$GEMINI_API_KEY" == *"API_KEY"* || ${#GEMINI_API_KEY} -lt 30 ]] && \
        echo -e "${COLOR_YELLOW}⚠️ Warning: API Key seems invalid or too short.${reset_color}" >&2
    verbose_log "API key validated."
    return 0
}

validate_paths() {
    local input_path="$1" output_path="$2" input_is_stdin="$3" output_is_stdout="$4" force_overwrite="$5"

    if [[ "$input_is_stdin" -eq 0 ]]; then
        verbose_log "Validating input: '$input_path'..."
        [[ ! -f "$input_path" ]] && error_exit "Input file '$input_path' not found."
        [[ ! -r "$input_path" ]] && error_exit "Input file '$input_path' not readable."
        [[ ! -s "$input_path" ]] && error_exit "Input file '$input_path' is empty."
    else
        [[ -t 0 ]] && error_exit "Stdin is a terminal. Pipe input or use -i <file>."
    fi

    if [[ "$output_is_stdout" -eq 0 ]]; then
        verbose_log "Validating output: '$output_path'..."
        local output_dir
        output_dir=$(dirname "$output_path") || output_dir="."
        [[ ! -d "$output_dir" ]] && error_exit "Output directory '$output_dir' does not exist."
        [[ ! -w "$output_dir" ]] && error_exit "Output directory '$output_dir' not writable."

        if [[ -e "$output_path" ]]; then
            [[ ! -f "$output_path" ]] && error_exit "Output path '$output_path' is not a regular file."
            if [[ "$force_overwrite" -eq 1 ]]; then
                verbose_log "Forcing overwrite of '$output_path'."
            elif [[ ! -w "$output_path" ]]; then
                error_exit "Output file '$output_path' not writable."
            else
                echo -e "${COLOR_BOLD_YELLOW}⚠️ Output file '$output_path' exists.${reset_color}" >&2
                read -r -p "$(echo -e "${COLOR_BLUE}Overwrite? (y/N): ${reset_color}")" confirm_overwrite < /dev/tty
                [[ "${confirm_overwrite,,}" != "y" ]] && { echo -e "${COLOR_MAGENTA}Aborted.${reset_color}" >&2; exit 0; }
                verbose_log "Overwrite confirmed."
            fi
        fi
    fi
}

prepare_code_and_prompt() {
    local input_path="$1" output_name="$2" input_is_stdin="$3"
    local lang_hint="" filename="stdin" original_code_content=""

    echo -e "${COLOR_BOLD_CYAN}--- Preparing Code and Prompt ---${reset_color}" >&2

    if [[ "$input_is_stdin" -eq 1 ]]; then
        original_code_content=$(cat) || error_exit "Failed to read from stdin."
        [[ -z "$original_code_content" ]] && error_exit "Stdin was empty."
    else
        filename="${input_path##*/}"
        original_code_content=$(cat "$input_path") || error_exit "Failed to read '$input_path'."
    fi
    verbose_log "Read ${#original_code_content} bytes from ${filename}."

    if [[ -n "$LANG_HINT_OVERRIDE" ]]; then
        lang_hint="$LANG_HINT_OVERRIDE"
    elif [[ "$input_is_stdin" -eq 0 ]]; then
        local file_ext="${input_path##*.}"
        case "$file_ext" in
            py) lang_hint="python" ;;
            js) lang_hint="javascript" ;;
            sh|bash) lang_hint="bash" ;;
            *) lang_hint="" ;;
        esac
    fi
    verbose_log "Language hint: ${lang_hint:-none}."

    local prompt
    read -r -d '' prompt <<-EOF
Analyze and enhance the following code from ${filename}. Improve readability, resilience, efficiency, security, and use modern practices (${lang_hint:-language inferred}).

CRITICAL: Respond *only* with the complete, enhanced code block, ready for ${output_name}. Exclude preamble, explanations, or markdown outside the code. Use inline comments only for essential clarity.

Original Code:
\`\`\`${lang_hint}
${original_code_content}
\`\`\`
EOF
    verbose_log "Prompt prepared."
    printf "%s" "$prompt"
}

call_gemini_api() {
    local prompt_content="$1"
    local api_url="${API_BASE_URL}/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"
    local attempt=0

    echo -e "${COLOR_BOLD_CYAN}--- Calling Gemini API ($API_MODEL) ---${reset_color}" >&2
    verbose_log "API URL: $api_url"

    local json_payload
    json_payload=$(printf "%s" "$prompt_content" | jq -R -s \
        --argjson temp_num "$API_TEMPERATURE" \
        '{"contents": [{"parts": [{"text": .}]}], "generationConfig": {"temperature": $temp_num}}') || \
        error_exit "Failed to create JSON payload."

    RESPONSE_BODY_FILE=$(mktemp "${TEMP_DIR}/gemini_response.XXXXXX") || error_exit "Failed to create temp file."
    chmod 600 "$RESPONSE_BODY_FILE"
    verbose_log "Temp file: $RESPONSE_BODY_FILE"

    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        attempt=$((attempt + 1))
        verbose_log "Attempt $attempt/$((MAX_RETRIES + 1))..."

        local http_code
        http_code=$(curl --fail -L -s -w '%{http_code}' \
            --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
            -H 'Content-Type: application/json' \
            -d "$json_payload" \
            -X POST "$api_url" \
            -o "$RESPONSE_BODY_FILE")
        local curl_exit_code=$?

        if [[ $curl_exit_code -eq 0 && "$http_code" -eq 200 ]]; then
            echo -e "${COLOR_GREEN}✅ API call successful (HTTP $http_code).${reset_color}" >&2
            return 0
        fi

        echo -e "${COLOR_YELLOW}⚠️ Attempt $attempt failed (Curl Exit: $curl_exit_code, HTTP: $http_code).${reset_color}" >&2
        parse_api_error_message "$RESPONSE_BODY_FILE" "$http_code"

        if [[ "$attempt" -le "$MAX_RETRIES" ]] && { [[ $curl_exit_code -ne 0 ]] || [[ "$http_code" -ge 500 ]] || [[ "$http_code" -eq 429 ]]; }; then
            echo -e "${COLOR_YELLOW}Retrying in ${RETRY_DELAY}s...${reset_color}" >&2
            sleep "$RETRY_DELAY"
        else
            error_exit "API call failed after $attempt attempts."
        fi
    done
}

parse_api_error_message() {
    local response_file="$1" http_code="$2"
    [[ ! -s "$response_file" ]] && { verbose_log "No response body for error parsing."; return; }

    if ! jq -e . "$response_file" >/dev/null 2>&1; then
        echo -e "${COLOR_YELLOW}--- Non-JSON Error Response ---${reset_color}" >&2
        cat "$response_file" >&2
        echo -e "${COLOR_YELLOW}-----------------------------${reset_color}" >&2
        return
    fi

    local error_message
    error_message=$(jq -r '.error.message // ""' "$response_file") || error_message=""
    if [[ -n "$error_message" ]]; then
        echo -e "${COLOR_YELLOW}--- API Error ---${reset_color}" >&2
        echo -e "${COLOR_RED}${error_message}${reset_color}" >&2
        case "$error_message" in
            *API*key*not*valid*) echo -e "${COLOR_BOLD_RED}Suggestion: Verify GEMINI_API_KEY.${reset_color}" >&2 ;;
            *quota* | *"$http_code" -eq 429) echo -e "${COLOR_BOLD_RED}Suggestion: Check API quota or rate limits.${reset_color}" >&2 ;;
            *permission*denied* | *"$http_code" -eq 403) echo -e "${COLOR_BOLD_RED}Suggestion: Check API key permissions.${reset_color}" >&2 ;;
            *invalid*argument* | *"$http_code" -eq 400) echo -e "${COLOR_BOLD_RED}Suggestion: Check request payload.${reset_color}" >&2 ;;
        esac
        echo -e "${COLOR_YELLOW}----------------${reset_color}" >&2
    else
        verbose_log "No specific error message extracted."
    fi
}

process_response() {
    [[ -z "$RESPONSE_BODY_FILE" || ! -r "$RESPONSE_BODY_FILE" ]] && error_exit "Response file missing or unreadable."

    echo -e "${COLOR_BOLD_CYAN}--- Processing Response ---${reset_color}" >&2
    verbose_log "Processing: $RESPONSE_BODY_FILE"

    [[ "$VERBOSE_MODE" -eq 1 ]] && {
        verbose_log "--- Raw Response ---"
        cat "$RESPONSE_BODY_FILE" >&2
        verbose_log "--- End Raw Response ---"
    }

    jq -e . >/dev/null 2>&1 "$RESPONSE_BODY_FILE" || error_exit "Response is not valid JSON."
    jq -e 'has("candidates") and (.candidates | type == "array")' "$RESPONSE_BODY_FILE" >/dev/null 2>&1 || \
        error_exit "Response missing 'candidates' array."
    jq -e '.candidates | length > 0' "$RESPONSE_BODY_FILE" >/dev/null 2>&1 || {
        local block_reason
        block_reason=$(jq -r '.promptFeedback.blockReason // ""' "$RESPONSE_BODY_FILE") || block_reason=""
        [[ -n "$block_reason" ]] && error_exit "Prompt blocked: ${block_reason}."
        error_exit "No candidates in response."
    }

    local finish_reason
    finish_reason=$(jq -r '.candidates[0].finishReason // "MISSING"' "$RESPONSE_BODY_FILE") || finish_reason="READ_ERROR"
    verbose_log "Finish Reason: ${finish_reason}"
    case "$finish_reason" in
        STOP) ;;
        MAX_TOKENS) echo -e "${COLOR_YELLOW}⚠️ Output may be truncated (max tokens).${reset_color}" >&2 ;;
        SAFETY|RECITATION) error_exit "Response blocked: ${finish_reason}." ;;
        MISSING|READ_ERROR) echo -e "${COLOR_YELLOW}⚠️ Finish reason missing or unreadable.${reset_color}" >&2 ;;
        *) echo -e "${COLOR_YELLOW}⚠️ Unexpected finish reason: ${finish_reason}.${reset_color}" >&2 ;;
    esac

    local raw_final_content
    raw_final_content=$(jq -r '.candidates[0].content.parts[0].text' "$RESPONSE_BODY_FILE") || \
        error_exit "Failed to extract text content."
    [[ "$raw_final_content" == "null" ]] && error_exit "Response text is null."

    if [[ "$RAW_OUTPUT_MODE" -eq 1 ]]; then
        echo -e "${COLOR_CYAN}ℹ️ Raw output mode: returning full response.${reset_color}" >&2
        printf "%s\n" "$raw_final_content"
        return
    fi

    local final_code_content
    final_code_content=$(printf "%s\n" "$raw_final_content" | \
        awk '/^[[:space:]]*```[[:alnum:]_.-]*[[:space:]]*$/{if(!in_block){in_block=1;next}else{exit}} in_block{print}')
    if [[ -z "$final_code_content" ]]; then
        if grep -q '```' <<< "$raw_final_content"; then
            echo -e "${COLOR_YELLOW}⚠️ Failed to extract code block. Saving full response.${reset_color}" >&2
        else
            echo -e "${COLOR_CYAN}ℹ️ No code fences detected. Using full response as code.${reset_color}" >&2
        fi
        final_code_content="$raw_final_content"
    fi

    printf "%s\n" "$final_code_content"
}

handle_output() {
    local content="$1" output_file="$2" output_is_stdout="$3"

    if [[ "$output_is_stdout" -eq 1 ]]; then
        verbose_log "Outputting to stdout."
        printf "%s\n" "$content"
    else
        echo -e "${COLOR_BOLD_CYAN}--- Saving Output ---${reset_color}" >&2
        printf "%s\n" "$content" > "$output_file" || error_exit "Failed to write to '$output_file'."
        local final_path
        command -v realpath >/dev/null 2>&1 && final_path=$(realpath "$output_file" 2>/dev/null) || final_path="${PWD}/${output_file}"
        echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------" >&2
        echo -e "✅ Saved to: ${COLOR_BOLD_YELLOW}'$final_path'${reset_color}" >&2
        echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------${reset_color}" >&2
    fi
}

# === Main Execution ===

trap 'signal_handler 1' SIGHUP
trap 'signal_handler 2' SIGINT
trap 'signal_handler 3' SIGQUIT
trap 'signal_handler 15' SIGTERM
trap cleanup EXIT

echo -e "${COLOR_MAGENTA}### Pyrmethus Code Enhancer ###${reset_color}" >&2

resolve_temp_dir
mkdir -p "$PYRMETHUS_CONFIG_DIR" 2>/dev/null || verbose_log "Failed to create config dir."
load_config
parse_arguments "$@"
verbose_log "Verbose mode enabled."
check_tools
resolve_and_check_api_key || exit 1
validate_paths "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME" "$INPUT_FROM_STDIN" "$OUTPUT_TO_STDOUT" "$FORCE_OVERWRITE"
prompt=$(prepare_code_and_prompt "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME" "$INPUT_FROM_STDIN")
call_gemini_api "$prompt"
final_content=$(process_response)
handle_output "$final_content" "$OUTPUT_FILE_NAME" "$OUTPUT_TO_STDOUT"

echo -e "${COLOR_MAGENTA}### Completed ###${reset_color}" >&2
trap - EXIT
cleanup
exit 0