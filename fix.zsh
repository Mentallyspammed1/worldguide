#!/usr/bin/env zsh
# Zsh specific enhancements and options
setopt EXTENDED_GLOB NO_NOMATCH
set -euo pipefail

# === Arcane Constants & Colors ===
readonly COLOR_RESET='\033[0m'
# ... (keep all other COLOR_ variables) ...
readonly COLOR_BOLD_CYAN='\033[1;36m'
# Define the reset sequence explicitly
readonly reset_color="$COLOR_RESET" # <<< FIX: Define reset_color

# === Default Configuration ===
# ... (rest of defaults are fine) ...
readonly DEFAULT_MAX_TIME="180"      # Seconds

# === Global Essences (Defaults/Placeholders) ===
# ... (these are fine) ...
RESPONSE_BODY_FILE=""

# === Utility Incantations ===

# --- Help Message ---
usage() {
  # Using print -P with explicit reset_color
  print -P "%{${COLOR_BOLD_CYAN}%}Pyrmethus Code Enhancer - Zsh Script Transmutation Spell%{$reset_color%}\n"
  print -P "%{${COLOR_YELLOW}%}Usage:%{$reset_color%} $0 -i <input_script> -o <output_artifact> [-k <api_key>] [-m <model>] [-t <temp>] [-h]\n"
  # ... (rest of usage message - ensure all %{$reset_color%} are used) ...
  print -P "\n%{${COLOR_CYAN}%}Example:%{$reset_color%} $0 -i my_script.py -o my_script_v2.py"
  exit 0
}

# --- Error Manifestation ---
error_exit() {
  print -P "%{${COLOR_BOLD_RED}%}❌ Abort! ${1}%{$reset_color%}" >&2
  cleanup
  exit 1
}

# --- Ephemeral Banishment ---
cleanup() {
  if [[ -n "$RESPONSE_BODY_FILE" && -f "$RESPONSE_BODY_FILE" ]]; then
    print -P "%{${COLOR_CYAN}%}# Banishing ephemeral file: $RESPONSE_BODY_FILE...%{$reset_color%}" >&2
    rm -f "$RESPONSE_BODY_FILE" 2>/dev/null || true
  fi
}

# --- Tool Scrying ---
check_tools() {
    print -P "%{${COLOR_BOLD_CYAN}%}⚙️  Scrying for required artifacts (curl, jq)...%{$reset_color%}"
    # ... (rest of check_tools is likely fine) ...
    print -P "%{${COLOR_GREEN}%}✅ Artifacts are present.%{$reset_color%}"
}

# --- Argument Parsing using zparseopts ---
parse_arguments() {
  zmodload zsh/zutil || error_exit "Zsh module 'zsh/zutil' needed for zparseopts failed to load."
  local -A args
  local options_spec=(
    'h|help' 'i|input:' 'o|output:' 'k|key:' 'm|model:' 't|temperature:'
    'connect-timeout:' 'max-time:'
  )
  # zparseopts call should be okay
  zparseopts -D -E -K -A args -- "$options_spec[@]" || {
      print -P "%{${COLOR_RED}%}Argument parsing spell misfired. Use -h for guidance.%{$reset_color%}" >&2
      exit 1
  }
  # Processing args... (ensure all checks like [[ -v args[...] ]] are correct)
  if [[ -v args[(I)-h|-help] ]]; then usage; fi
  if [[ -v args[(I)-i|--input] ]]; then INPUT_FILE_PATH=$args[(I)-i|--input]; fi
  if [[ -v args[(I)-o|--output] ]]; then OUTPUT_FILE_NAME=$args[(I)-o|--output]; fi
  if [[ -v args[(I)-k|--key] ]]; then GEMINI_API_KEY=$args[(I)-k|--key]; fi
  if [[ -v args[(I)-m|--model] ]]; then API_MODEL=$args[(I)-m|--model]; fi
  if [[ -v args[(I)-t|--temperature] ]]; then API_TEMPERATURE=$args[(I)-t|--temperature]; fi
  if [[ -v args[--connect-timeout] ]]; then CONNECT_TIMEOUT=$args[--connect-timeout]; fi
  if [[ -v args[--max-time] ]]; then MAX_TIME=$args[--max-time]; fi

  if (( $# > 0 )); then
      print -P "%{${COLOR_RED}%}Unexpected spectral arguments found: $@%{$reset_color%}" >&2
      usage
  fi
  # Required arg validation
  if [[ -z "$INPUT_FILE_PATH" ]]; then error_exit "Input script path (-i or --input) is mandatory."; fi
  if [[ -z "$OUTPUT_FILE_NAME" ]]; then error_exit "Output artifact name (-o or --output) is mandatory."; fi
}

# --- Oracle Key Resolution & Verification ---
resolve_and_check_api_key() {
    # ... (logic is likely fine, ensure reset_color is used in print -P) ...
    if [[ -z "$GEMINI_API_KEY" ]]; then
        GEMINI_API_KEY="${ENV_GEMINI_API_KEY:-${GEMINI_API_KEY_STD:-}}"
    fi
    if [[ -z "$GEMINI_API_KEY" ]]; then
        print -P "%{${COLOR_BOLD_RED}%}...(your multi-line error)...%{$reset_color%}" >&2
        return 1
    fi
    print -P "%{${COLOR_GREEN}%}🔑 Oracle Key source resolved.%{$reset_color%}"
    return 0
}

# --- Input File Validation ---
validate_input_file() {
    # ... (logic is likely fine, ensure reset_color is used in print -P) ...
    print -P "%{${COLOR_GREEN}%}✅ Input artifact validated.%{$reset_color%}"
}

# --- Code Assimilation & Prompt Forging ---
prepare_code_and_prompt() {
    # ... (logic is likely fine, ensure reset_color is used in print -P and here doc) ...
    # Ensure colors in the HERE DOC use explicit variables like ${COLOR_BOLD_RED}
    # and ${reset_color} or ${COLOR_RESET}
    read -r -d '' prompt <<-EOF
Greetings, Oracle Gemini...

%{${COLOR_BOLD_RED}%}CRITICAL DECREE:%{$reset_color%} Your entire response...

Original Script Artifact:
\`\`\`${lang_hint}
${original_code_content}
\`\`\`
EOF
    print -n "$prompt"
}

# --- Oracle Communion ---
call_gemini_api() {
  # ... (parameter definitions) ...
  local api_url="${API_BASE_URL}/${model}:generateContent?key=${api_key}"

  print -P "%{${COLOR_BOLD_CYAN}%}--- Communing with the Gemini Oracle ($model)... ---%{$reset_color%}"
  # ... (jq payload creation is likely fine) ...
  json_payload=$(print -n "$prompt_content" | jq ...) || error_exit "Failed to forge JSON payload..."

  RESPONSE_BODY_FILE=$(mktemp "gemini_oracle_whispers.tmp.XXXXXX") || error_exit "Failed to conjure..."
  chmod 600 "$RESPONSE_BODY_FILE"

  local http_code
  # Use single quotes around the -w argument
  http_code=$(curl --fail -L -s -w '%{http_code}' \
       --connect-timeout "$conn_timeout" --max-time "$max_timeout" \
       -H 'Content-Type: application/json' \
       -d "$json_payload" \
       -X POST "$api_url" \
       -o "$RESPONSE_BODY_FILE") # <<< FIX: Quoted -w format string

  local curl_exit_code=$?
  # ... (rest of error checking, ensure reset_color used in print -P) ...
  print -P "%{${COLOR_GREEN}%}✅ Oracle communion successful (HTTP $http_code).%{$reset_color%}"
}

# --- Response Interpretation ---
process_response() {
    # ... (logic is likely fine, ensure reset_color used in print -P) ...
    # awk invocation should be okay
    # Ensure final print uses reset_color if needed
    print -r -- "$final_code_content"
}

# --- Artifact Scribing ---
save_output() {
    # ... (logic is likely fine, ensure reset_color used in print -P) ...
    print -P "%{${COLOR_BOLD_YELLOW}%}'$final_path'%{$reset_color%}" # Highlight path
}

# === Main Incantation Flow ===
trap cleanup EXIT TERM INT
print -P "%{${COLOR_MAGENTA}%}### Pyrmethus Code Enhancer (Zsh) Activated ###%{$reset_color%}"
parse_arguments "$@"
check_tools
if ! resolve_and_check_api_key; then exit 1; fi
validate_input_file "$INPUT_FILE_PATH"
prompt=$(prepare_code_and_prompt "$INPUT_FILE_PATH" "$OUTPUT_FILE_NAME")
call_gemini_api "$prompt" "$GEMINI_API_KEY" "$API_MODEL" "$API_TEMPERATURE" "$CONNECT_TIMEOUT" "$MAX_TIME"
final_code=$(process_response)
save_output "$final_code" "$OUTPUT_FILE_NAME"
print -P "%{${COLOR_MAGENTA}%}### Zsh Incantation Complete. The enhanced artifact awaits! ###%{$reset_color%}"
exit 0