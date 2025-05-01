#!/usr/bin/env bash
# set -e: Halt the incantation if any spell falters.
# set -o pipefail: Ensure pipeline spirits report failure accurately.
# set -u: Treat unset variables as forbidden shadows unless explicitly handled.
set -euo pipefail

# === Arcane Constants & Colors ===
# Define the hues of our digital tapestry (standard ANSI)
readonly COLOR_RESET='\033[0m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[0;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_MAGENTA='\033[0;35m'
readonly COLOR_CYAN='\033[0;36m'
readonly COLOR_BOLD_RED='\033[1;31m'
readonly COLOR_BOLD_GREEN='\033[1;32m'
readonly COLOR_BOLD_YELLOW='\033[1;33m'
readonly COLOR_BOLD_CYAN='\033[1;36m'
readonly reset_color="$COLOR_RESET" # Alias for convenience

# === Default Configuration ===
readonly DEFAULT_API_MODEL="gemini-1.5-pro-latest"
readonly DEFAULT_API_TEMPERATURE="0.3"
readonly DEFAULT_CONNECT_TIMEOUT="20" # Seconds
readonly DEFAULT_MAX_TIME="180"      # Seconds

# === Global Essences (Defaults/Placeholders) ===
INPUT_FILE_PATH=""
OUTPUT_FILE_NAME=""
GEMINI_API_KEY=""
API_MODEL="${DEFAULT_API_MODEL}"
API_TEMPERATURE="${DEFAULT_API_TEMPERATURE}"
CONNECT_TIMEOUT="${DEFAULT_CONNECT_TIMEOUT}"
MAX_TIME="${DEFAULT_MAX_TIME}"
RESPONSE_BODY_FILE=""

# === Utility Incantations ===

# --- Help Message ---
usage() {
  printf "%b" "${COLOR_BOLD_CYAN}Pyrmethus Code Enhancer - Termux Script Transmutation Spell${reset_color}\n\n"
  printf "%b" "${COLOR_YELLOW}Usage:${reset_color} $0 -i <input_script> -o <output_artifact> [-k <api_key>] [-m <model>] [-t <temp>] [-h]\n\n"
  printf "%b" "${COLOR_GREEN}Required Arguments:${reset_color}\n"
  printf "  %-20s %s\n" "-i, --input" "Path to the local script needing transmutation."
  printf "  %-20s %s\n" "-o, --output" "Desired name for the final enhanced artifact."
  printf "\n%b" "${COLOR_GREEN}Optional Arguments:${reset_color}\n"
  printf "  %-20s %s\n" "-k, --key" "Gemini API Key (overrides GEMINI_API_KEY environment variable)."
  printf "  %-20s %s\n" "-m, --model" "Gemini model to use (default: '${DEFAULT_API_MODEL}')."
  printf "  %-20s %s\n" "-t, --temperature" "API temperature (0.0-2.0, default: ${DEFAULT_API_TEMPERATURE})." # Note: Range can vary by model
  printf "  %-20s %s\n" "--connect-timeout" "Curl connect timeout in seconds (default: ${DEFAULT_CONNECT_TIMEOUT})."
  printf "  %-20s %s\n" "--max-time" "Curl max total time in seconds (default: ${DEFAULT_MAX_TIME})."
  printf "  %-20s %s\n" "-h, --help" "Display this arcane guidance."
  printf "\n%b" "${COLOR_CYAN}Example:${reset_color} $0 -i my_script.py -o my_script_v2.py\n"
  exit 0
}

# --- Error Manifestation ---
error_exit() {
  echo -e "${COLOR_BOLD_RED}❌ Abort! ${1}${reset_color}" >&2
  cleanup # Ensure ephemeral traces are banished before exiting
  exit 1
}

# --- Ephemeral Banishment ---
cleanup() {
  # Check if the variable is non-empty and points to a file before removing
  if [[ -n "$RESPONSE_BODY_FILE" && -f "$RESPONSE_BODY_FILE" ]]; then
    echo -e "${COLOR_CYAN}# Banishing ephemeral file: $RESPONSE_BODY_FILE...${reset_color}" >&2
    rm -f "$RESPONSE_BODY_FILE" 2>/dev/null || true # Ignore rm errors during cleanup
  fi
}

# --- Tool Scrying ---
check_tools() {
  echo -e "${COLOR_BOLD_CYAN}⚙️  Scrying for required artifacts (curl, jq)...${reset_color}"
  local missing_tools=""
  command -v curl >/dev/null 2>&1 || missing_tools+=" curl"
  command -v jq >/dev/null 2>&1 || missing_tools+=" jq"

  if [[ -n "$missing_tools" ]]; then
      error_exit "Required artifacts missing:$missing_tools. Summon them with: ${COLOR_YELLOW}pkg install$missing_tools${reset_color}"
  fi
  echo -e "${COLOR_GREEN}✅ Artifacts are present.${reset_color}"
}

# --- Argument Parsing ---
# Uses getopt (util-linux) if available for long options, falls back to getopts (bash built-in)
parse_arguments() {
  local options="i:o:k:m:t:h"
  local long_options="input:,output:,key:,model:,temperature:,connect-timeout:,max-time:,help"

  # Check if external getopt supports long options
  if getopt --test > /dev/null 2>&1 && command -v getopt >/dev/null 2>&1; then
      local parsed getopt_status
      parsed=$(getopt --options "$options" --longoptions "$long_options" --name "$0" -- "$@")
      getopt_status=$? # Capture status immediately

      if [[ $getopt_status -ne 0 ]]; then
          echo -e "${COLOR_RED}Argument parsing failed (getopt error). Use -h for help.${reset_color}" >&2
          exit 1
      fi
      # Replace current arguments ($@) with the ordered ones from getopt
      eval set -- "$parsed"

      while true; do
          case "$1" in
              -i|--input) INPUT_FILE_PATH="$2"; shift 2 ;;
              -o|--output) OUTPUT_FILE_NAME="$2"; shift 2 ;;
              -k|--key) GEMINI_API_KEY="$2"; shift 2 ;;
              -m|--model) API_MODEL="$2"; shift 2 ;;
              -t|--temperature) API_TEMPERATURE="$2"; shift 2 ;;
              --connect-timeout) CONNECT_TIMEOUT="$2"; shift 2 ;;
              --max-time) MAX_TIME="$2"; shift 2 ;;
              -h|--help) usage; shift ;; # usage exits
              --) shift; break ;; # End of options
              *) echo -e "${COLOR_RED}Internal error in argument processing loop! Argument: '$1'${reset_color}"; exit 1 ;;
          esac
      done
  else
      # Fallback to bash built-in getopts (only short options)
      echo -e "${COLOR_YELLOW}# Notice: Enhanced 'getopt' not found, using basic 'getopts'. Long options (--option) unavailable.${reset_color}" >&2
      # Leading ':' enables silent error reporting.
      while getopts ":i:o:k:m:t:h" opt; do
          case ${opt} in
              i) INPUT_FILE_PATH="${OPTARG}" ;;
              o) OUTPUT_FILE_NAME="${OPTARG}" ;;
              k) GEMINI_API_KEY="${OPTARG}" ;;
              m) API_MODEL="${OPTARG}" ;;
              t) API_TEMPERATURE="${OPTARG}" ;;
              h) usage ;;
              \?) echo -e "${COLOR_RED}Invalid option: -${OPTARG}${reset_color}" >&2; usage ;;
              :) echo -e "${COLOR_RED}Option -${OPTARG} requires an argument.${reset_color}" >&2; usage ;;
          esac
      done
      shift $((OPTIND -1)) # Remove processed options
  fi

   # Check for any remaining non-option arguments (should be none)
  if [[ $# -gt 0 ]]; then
      echo -e "${COLOR_RED}Unexpected arguments found: '$@'${reset_color}" >&2
      usage
  fi

  # --- Validate Required Arguments Presence ---
  if [[ -z "$INPUT_FILE_PATH" ]]; then
      error_exit "Input script path (-i or --input) is mandatory. Use -h for guidance."
  fi
  if [[ -z "$OUTPUT_FILE_NAME" ]]; then
      error_exit "Output artifact name (-o or --output) is mandatory. Use -h for guidance."
  fi

  # Note: Advanced validation of temperature/timeout values (e.g., range checks) omitted for simplicity.
  # The API/curl will likely handle invalid numeric formats.
}

# --- Oracle Key Resolution & Verification ---
resolve_and_check_api_key() {
  # Prioritize key from -k argument, then environment variables
  if [[ -z "$GEMINI_API_KEY" ]]; then
      # Check common env var names
      GEMINI_API_KEY="${ENV_GEMINI_API_KEY:-${GEMINI_API_KEY:-${GEMINI_API_KEY_STD:-}}}"
  fi

  # Final check if a key was found
  if [[ -z "$GEMINI_API_KEY" ]]; then
    printf "%b" "${COLOR_BOLD_RED}--------------------------------------------------------------------
🔒 Error: Gemini Oracle Key not found.
   Provide it using the -k argument or set the GEMINI_API_KEY environment variable.

   Example:
   ${COLOR_YELLOW}export GEMINI_API_KEY='YOUR_ACTUAL_API_KEY'${COLOR_BOLD_RED}
   Or run with: $0 ... -k 'YOUR_ACTUAL_API_KEY' ...

   🚨 ${COLOR_YELLOW}A Wizard's Warning:${COLOR_BOLD_RED} Protect your keys diligently.
--------------------------------------------------------------------${reset_color}\n" >&2
    return 1 # Signal failure
  fi
  echo -e "${COLOR_GREEN}🔑 Oracle Key source resolved.${reset_color}"
  return 0
}

# --- Input File Validation ---
validate_input_file() {
    local file_path="$1"
    echo -e "${COLOR_CYAN}# Validating input artifact: '$file_path'...${reset_color}"
    if [[ ! -f "$file_path" ]]; then
        error_exit "Input script artifact not found at '$file_path'."
    fi
    if [[ ! -r "$file_path" ]]; then
        error_exit "Cannot perceive input script artifact '$file_path'. Check read permissions."
    fi
    echo -e "${COLOR_GREEN}✅ Input artifact validated.${reset_color}"
}

# --- Code Assimilation & Prompt Forging ---
prepare_code_and_prompt() {
  local input_path="$1"
  local output_name="$2"
  local lang_hint=""
  local file_ext="${input_path##*.}"
  local filename="${input_path##*/}"

  echo -e "${COLOR_BOLD_CYAN}--- Assimilating Script Essence & Forging the Oracle Prompt ---${reset_color}"
  echo -e "${COLOR_CYAN}# Reading script energies from '$filename'...${reset_color}"

  local original_code_content
  # Read file content; set -e handles immediate exit on failure
  original_code_content=$(cat "$input_path") || error_exit "Failed to assimilate script energies from '$input_path'."

  # Divine the language from the file extension
  case "$file_ext" in
    py) lang_hint="python" ;; js) lang_hint="javascript" ;; sh|bash) lang_hint="bash" ;; *) lang_hint="" ;;
  esac
  echo -e "${COLOR_GREEN}✅ Script essence assimilated. Language Divination: ${COLOR_YELLOW}${lang_hint:-'Oracle\'s Guess'}${reset_color}."

  local prompt
  # Here Document for clarity; ensure embedded color vars are ${...}
  read -r -d '' prompt <<-EOF
Greetings, Oracle Gemini. Pyrmethus, Termux Wizard, seeks your wisdom. Please transmute the following script artifact (originally from ${filename}). Infuse it with enhancements addressing readability, resilience, efficiency, security, and modern Termux-aware practices (${lang_hint:-for its native tongue}).

${COLOR_BOLD_RED}CRITICAL DECREE:${reset_color} Your entire response must be *only* the complete, final, enhanced code block, ready to be scribed directly into the new artifact (${output_name}). Exclude *all* preamble, summaries, explanations, or markdown adornments (like \`\`\`) outside the final code itself. Embed clarifying comments *within* the code only where essential enlightenment is needed.

Original Script Artifact:
\`\`\`${lang_hint}
${original_code_content}
\`\`\`
EOF
  # Use printf "%s" for safe output without adding a newline
  printf "%s" "$prompt"
}

# --- Oracle Communion ---
call_gemini_api() {
  local prompt_content="$1"
  local api_key="$2"
  local model="$3"
  local temp="$4"
  local conn_timeout="$5"
  local max_timeout="$6"
  local api_url="${API_BASE_URL}/${model}:generateContent?key=${api_key}"

  echo -e "${COLOR_BOLD_CYAN}--- Communing with the Gemini Oracle ($model)... ---${reset_color}"
  echo -e "${COLOR_CYAN}# Channeling the request through the digital ether...${reset_color}"

  local json_payload
  # Use jq -R -s to read raw stdin as a single JSON string, avoiding ARG_MAX limits.
  # Use --argjson for temp to ensure it's treated as a number.
  json_payload=$(printf "%s" "$prompt_content" | jq -R -s \
    --argjson temp_num "$temp" \
    '{
      "contents": [{"parts": [{"text": .}]}],
      "generationConfig": {"temperature": $temp_num}
    }') || error_exit "Failed to forge JSON payload using jq's arcane arts."

  # Conjure temporary file before curl call
  RESPONSE_BODY_FILE=$(mktemp "gemini_oracle_whispers.tmp.XXXXXX") || error_exit "Failed to conjure temporary scroll."
  chmod 600 "$RESPONSE_BODY_FILE" # Secure permissions

  local http_code curl_exit_code
  # Use single quotes for -w format string for robustness.
  # Use variables for timeouts.
  http_code=$(curl --fail -L -s -w '%{http_code}' \
       --connect-timeout "$conn_timeout" --max-time "$max_timeout" \
       -H 'Content-Type: application/json' \
       -d "$json_payload" \
       -X POST "$api_url" \
       -o "$RESPONSE_BODY_FILE")
  curl_exit_code=$? # Capture curl's exit status immediately

  # Check curl exit code first (catches network errors, timeouts, --fail errors)
   if [[ $curl_exit_code -ne 0 ]]; then
      echo -e "${COLOR_BOLD_RED}❌ Error: Messenger spirit (curl) faltered (Exit Code: $curl_exit_code, HTTP Status: $http_code).${reset_color}" >&2
      if [[ -s "$RESPONSE_BODY_FILE" ]]; then
          echo -e "${COLOR_YELLOW}--- Oracle's Whispers (Error Context) ---${reset_color}" >&2
          cat "$RESPONSE_BODY_FILE" >&2
          echo -e "${COLOR_YELLOW}---------------------------------------${reset_color}" >&2
      fi
      error_exit "Oracle communion failed during transmission." # error_exit calls cleanup
  fi

  # Check HTTP status code (redundant with --fail but good practice)
  if [[ "$http_code" -ne 200 ]]; then
    echo -e "${COLOR_BOLD_RED}❌ Error: Oracle communion yielded an unexpected status (HTTP $http_code).${reset_color}" >&2
    if [[ -s "$RESPONSE_BODY_FILE" ]]; then
        echo -e "${COLOR_YELLOW}--- Oracle's Whispers (HTTP Error $http_code) ---${reset_color}" >&2
        cat "$RESPONSE_BODY_FILE" >&2
        echo -e "${COLOR_YELLOW}---------------------------------------${reset_color}" >&2
    fi
    error_exit "Oracle responded with non-OK HTTP status." # error_exit calls cleanup
  fi

  echo -e "${COLOR_GREEN}✅ Oracle communion successful (HTTP $http_code).${reset_color}"
}

# --- Response Interpretation ---
process_response() {
  # Ensure the response file exists and is readable
  if [[ -z "$RESPONSE_BODY_FILE" || ! -r "$RESPONSE_BODY_FILE" ]]; then
       error_exit "Internal Error: Oracle response scroll is missing or unreadable."
  fi
  local response_file="$RESPONSE_BODY_FILE" # Use local var for clarity

  echo -e "${COLOR_BOLD_CYAN}--- Interpreting the Oracle's Whispers ---${reset_color}"
  echo -e "${COLOR_CYAN}# Deciphering the response scroll: '$response_file'...${reset_color}"

  # Validate JSON structure first
  if ! jq -e . >/dev/null 2>&1 "$response_file"; then
    echo -e "${COLOR_YELLOW}⚠️ Warning: Oracle's response scroll does not contain valid JSON runes.${reset_color}" >&2
    echo -e "${COLOR_YELLOW}--- Corrupted Scroll Content ---${reset_color}" >&2
    cat "$response_file" >&2
    echo -e "${COLOR_YELLOW}-----------------------------${reset_color}" >&2
    error_exit "Failed to parse Oracle's response as JSON."
  fi

  # Check if candidates array exists and has content
  if ! jq -e '.candidates | length > 0' "$response_file" >/dev/null 2>&1; then
     echo -e "${COLOR_YELLOW}⚠️ Warning: Oracle's response contains no valid candidates.${reset_color}" >&2
     echo -e "${COLOR_YELLOW}--- Raw Oracle Response ---${reset_color}" >&2
     cat "$response_file" >&2
     echo -e "${COLOR_YELLOW}------------------------${reset_color}" >&2
     error_exit "No candidates found in Oracle response."
  fi

  # Check finish reason of the first candidate for potential issues
  local finish_reason
  # Use // "" default in jq to handle cases where finishReason might be null or missing
  finish_reason=$(jq -er '.candidates[0].finishReason // ""' "$response_file") || {
      echo -e "${COLOR_YELLOW}# Warning: Could not determine finishReason from Oracle response. Proceeding cautiously.${reset_color}" >&2
      finish_reason="UNKNOWN" # Assign a default if jq fails here
  }

  # Add more specific finish reason checks if needed
  case "$finish_reason" in
      "STOP")
          echo -e "${COLOR_CYAN}# Oracle indicates generation completed normally.${reset_color}"
          ;;
      "MAX_TOKENS")
          echo -e "${COLOR_YELLOW}⚠️ Warning: Oracle stopped due to maximum token limit. Output may be truncated.${reset_color}" >&2
          ;;
      "SAFETY"|"RECITATION")
          echo -e "${COLOR_BOLD_RED}⚠️ Warning: Oracle stopped due to policy reasons (${finish_reason}). Output likely blocked or incomplete.${reset_color}" >&2
          # Optionally error_exit here depending on desired behavior
          ;;
      "UNKNOWN"|"") # Handle cases where it couldn't be determined or was empty
          echo -e "${COLOR_YELLOW}# Warning: Oracle finish reason is unknown or missing. Proceeding.${reset_color}" >&2
          ;;
      *) # Other unexpected reasons
          echo -e "${COLOR_YELLOW}⚠️ Warning: Oracle reported an unexpected finish reason: '${finish_reason}'. Proceeding.${reset_color}" >&2
          ;;
  esac

  # Extract text content using jq. -e exits non-zero if path null/missing. -r raw output.
  local raw_final_content jq_exit_code
  raw_final_content=$(jq -er '.candidates[0].content.parts[0].text // ""' "$response_file")
  jq_exit_code=$? # Capture jq exit status

  if [[ $jq_exit_code -ne 0 || -z "$raw_final_content" ]]; then
      if [[ $jq_exit_code -ne 0 ]]; then
          echo -e "${COLOR_YELLOW}⚠️ Warning: Failed to extract text essence via jq path '.candidates[0].content.parts[0].text' (jq Exit: $jq_exit_code).${reset_color}" >&2
      else # Content was empty string
          echo -e "${COLOR_YELLOW}⚠️ Warning: Successfully parsed Oracle's response, but the extracted 'text' essence was empty.${reset_color}" >&2
      fi
      echo -e "${COLOR_YELLOW}--- Raw Oracle Response ---${reset_color}" >&2
      cat "$response_file" >&2
      echo -e "${COLOR_YELLOW}------------------------${reset_color}" >&2
      error_exit "Could not find or extract expected text essence in Oracle's response."
  fi

  # Attempt to isolate the pure code block using awk's pattern magic.
  # This pattern handles ``` and ```lang tags.
  local final_code_content
  final_code_content=$(printf "%s\n" "$raw_final_content" | \
    awk '
      BEGIN { in_block=0; emitted=0 }
      # Match ``` optionally followed by language identifier (letters, numbers, _, ., -)
      /^[[:space:]]*```[[:alnum:]_.-]*[[:space:]]*$/ {
          if (!in_block) {
              # Found the opening fence
              in_block=1
              emitted=1 # Mark that we found the start marker
              next      # Skip printing the opening fence line
          } else {
              # Found the closing fence, our quest ends here
              exit
          }
      }
      # If we are inside the block, print the line
      in_block { print }
    ')

  # Analyze the result of the extraction spell
  if [[ -z "$final_code_content" ]] && [[ "$raw_final_content" == *"```"* ]]; then
      # Awk failed, but fences were present - perhaps misformed?
      echo -e "${COLOR_YELLOW}--------------------------------------------------------------------" >&2
      echo -e "⚠️ Warning: Could not automatically isolate code between markdown fences (\`\`\`)." >&2
      echo -e "   The Oracle's formatting might be unusual or incomplete." >&2
      echo -e "   Presenting the ${COLOR_BOLD_RED}full text response${COLOR_YELLOW} received. Manual refinement may be needed." >&2
      echo -e "--------------------------------------------------------------------${reset_color}" >&2
      final_code_content="$raw_final_content" # Present the raw essence as fallback
  elif [[ -z "$final_code_content" ]]; then
      # No fences found anywhere. Assume the Oracle obeyed the CRITICAL DECREE.
      echo -e "${COLOR_CYAN}ℹ️ Info: No markdown code fences (\`\`\`) detected. Assuming Oracle's entire response is the pure code essence as commanded.${reset_color}"
      final_code_content="$raw_final_content"
  else
      echo -e "${COLOR_GREEN}✅ Successfully isolated the enhanced code essence.${reset_color}"
  fi

  # Consider trimming leading/trailing empty lines if needed, but often not necessary
  # final_code_content=$(printf "%s\n" "$final_code_content" | sed -e '/^[[:space:]]*$/d') # Removes ALL blank lines

  # Return the final code essence using printf for safety
  printf "%s\n" "$final_code_content"
}

# --- Artifact Scribing ---
save_output() {
  local content="$1"
  local output_file="$2"

  echo -e "${COLOR_BOLD_CYAN}--- Scribing the Enhanced Artifact ---${reset_color}"
  # Use printf for safer scribing into the output file.
  if printf "%s\n" "$content" > "$output_file"; then
    # Determine the artifact's location in the realm
    local final_path
    # Use realpath if available for the canonical path, else construct it
    if command -v realpath >/dev/null; then
       # Follow symbolic links to the true location; handle potential realpath errors
       final_path=$(realpath "$output_file") || final_path="$output_file"
    else
       # Combine current realm path with the artifact name
       # Using PWD is generally reliable in scripts
       final_path="${PWD}/${output_file}"
    fi
    echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------"
    echo -e "✅ Transmutation Complete! Final enhanced artifact scribed to:"
    echo -e "   ${COLOR_BOLD_YELLOW}'$final_path'${reset_color}" # Highlight the path
    echo -e "${COLOR_BOLD_GREEN}--------------------------------------------------------------------${reset_color}"
  else
    # error_exit handles cleanup
    error_exit "Failed to scribe final artifact to '$output_file'. Check realm permissions or disk space."
  fi
}


# === Main Incantation Flow ===

# --- Preparation Ritual ---
# Ensure ephemeral traces are banished upon completion or interruption
trap cleanup EXIT TERM INT
echo -e "${COLOR_MAGENTA}### Pyrmethus Code Enhancer Activated ###${reset_color}"

# --- Parse Arguments ---
# Populates global variables like INPUT_FILE_PATH, OUTPUT_FILE_NAME, GEMINI_API_KEY, etc.
parse_arguments "$@" # Pass all script arguments

# --- Prerequisite Checks ---
check_tools
if ! resolve_and_check_api_key; then
  exit 1 # Halt if the key resolution failed.
fi
validate_input_file "$INPUT_FILE_PATH"

# --- Crafting the Request ---
# prepare_code_and_prompt reads the file and forges the Oracle prompt
prompt=$(prepare_code_and_prompt "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME")

# --- Oracle Communication ---
# Pass resolved/parsed config values to the API call function
call_gemini_api "$prompt" "$GEMINI_API_KEY" "$API_MODEL" "$API_TEMPERATURE" "$CONNECT_TIMEOUT" "$MAX_TIME"

# --- Deciphering the Response ---
# process_response reads the global RESPONSE_BODY_FILE set by call_gemini_api
final_code=$(process_response)

# --- Final Scribing ---
# Pass the target output file name
save_output "$final_code" "$OUTPUT_FILE_NAME"

# --- Ritual Completion ---
echo -e "${COLOR_MAGENTA}### Incantation Complete. The enhanced artifact awaits your command! ###${reset_color}"
# Cleanup is handled by the trap
exit 0