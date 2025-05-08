#!/usr/bin/env zsh

# Zsh script to:
# 1. Fetch code from a repository.
# 2. Ask Gemini to analyze, upgrade, enhance, and incorporate all ideas.
# 3. Request Gemini returns ONLY the complete, final code block.
# 4. Save this final code to a specified file.

# --- Configuration ---
REPO_URL="https://github.com/Mentallyspammed1/worldguide.git"
REPO_NAME="worldguide" # Used for messages
TEMP_DIR_BASE="temp_repo_enhance_replace_zsh"
# Create a unique temp directory name using mktemp if available
TEMP_DIR=$(mktemp -d "${TEMP_DIR_BASE}_XXXXXX" 2>/dev/null || echo "${TEMP_DIR_BASE}_$$") # Fallback using PID


# --- Function for cleanup ---
cleanup() {
  echo "🧹 Cleaning up temporary directory ($TEMP_DIR)..."
  rm -rf "$TEMP_DIR"
}

# --- Ensure cleanup runs on exit, interrupt, or termination ---
trap 'cleanup' EXIT TERM INT

# --- Check for required tools ---
echo "⚙️ Checking for required tools (curl, jq, git)..."
missing_tools=()
command -v curl > /dev/null 2>&1 || missing_tools+=("curl")
command -v jq > /dev/null 2>&1 || missing_tools+=("jq")
command -v git > /dev/null 2>&1 || missing_tools+=("git")

if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "❌ Error: Required tools are missing: ${missing_tools[*]}." >&2
    echo "   Please install them (e.g., 'pkg install ${missing_tools[*]}' in Termux)." >&2
    exit 1
fi
echo "✅ Tools found."

# --- Check for API Key (Critical Security Step) ---
if [[ -z "$GEMINI_API_KEY" ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "🔒 Error: GEMINI_API_KEY environment variable is not set." >&2
  echo "   This script requires your Google AI Gemini API Key." >&2
  echo >&2
  echo "   To set it temporarily for this session, run:" >&2
  echo "   export GEMINI_API_KEY='YOUR_ACTUAL_API_KEY'" >&2
  echo "   (Replace YOUR_ACTUAL_API_KEY with your real, SECURE key)" >&2
  echo >&2
  echo "   🚨 IMPORTANT: NEVER paste your API key directly into scripts" >&2
  echo "   🚨          NEVER commit API keys to Git repositories." >&2
  echo "   🚨 You previously exposed a key - please ensure you are using" >&2
  echo "   🚨 a NEW, SECURE key and the old one is REVOKED." >&2
  echo "--------------------------------------------------------------------" >&2
  exit 1
fi
echo "🔑 Gemini API Key found in environment."

# --- Get User Input ---
echo "---"
read "?Enter the path to the file inside '$REPO_NAME' to enhance (e.g., main.py): " INPUT_FILE_PATH
read "?Enter the desired name for the final enhanced output file (e.g., main.final.py): " OUTPUT_FILE_NAME

# --- Validate Input ---
if [[ -z "$INPUT_FILE_PATH" ]] || [[ "$INPUT_FILE_PATH" == /* ]] || [[ "$INPUT_FILE_PATH" == *..* ]]; then
    echo "❌ Error: Invalid input file path. Please use a relative path within the repo." >&2
    exit 1
fi
if [[ -z "$OUTPUT_FILE_NAME" ]]; then
  echo "❌ Error: No output file name provided." >&2
  exit 1
fi

# --- Clone the repository ---
echo "---"
echo "🌐 Cloning repository: $REPO_URL ..."
[[ "$TEMP_DIR" == "${TEMP_DIR_BASE}_$$" ]] && rm -rf "$TEMP_DIR" # Clean before clone if mktemp failed
git clone --depth 1 "$REPO_URL" "$TEMP_DIR"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to clone repository '$REPO_URL'." >&2
  exit 1 # Cleanup handled by trap
fi
echo "✅ Repository cloned."

# --- Check if the input file exists ---
FULL_INPUT_PATH="$TEMP_DIR/$INPUT_FILE_PATH"
echo "---"
echo "🔍 Checking for file: $FULL_INPUT_PATH"
if [[ ! -f "$FULL_INPUT_PATH" ]]; then
  echo "❌ Error: File '$INPUT_FILE_PATH' not found in repo." >&2
  exit 1 # Cleanup handled by trap
fi
echo "✅ Found file: $INPUT_FILE_PATH"

# --- Read and Format Original Code Content ---
echo "---"
echo "📑 Reading and formatting original code content..."
# Escape backslashes, then quotes, then newlines
ORIGINAL_CODE_CONTENT=$(cat "$FULL_INPUT_PATH" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to read/format original code from '$FULL_INPUT_PATH'." >&2
  exit 1
fi

# --- Detect Language ---
FILE_EXT=${INPUT_FILE_PATH:e}
LANG_HINT=""
case $FILE_EXT in
  py) LANG_HINT="python" ;;
  js) LANG_HINT="javascript" ;;
  sh|zsh) LANG_HINT="shell" ;;
  *) LANG_HINT="" ;;
esac
echo "✅ Original code formatted. Language hint: ${LANG_HINT:-'none'}."


# ================================================================
# API Call: Analyze, Enhance, Implement, and Return Final Code
# ================================================================
echo "---"
echo "🚀 Sending request to Gemini API to enhance and return final code..."
# Use a capable model like 1.5 Pro
API_MODEL="gemini-2.5-pro-exp-03-25"
API_URL="https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"

# Explicit prompt asking for analysis, enhancement, and ONLY the final code
PROMPT_ENHANCE="Please thoroughly analyze, upgrade, and enhance the following code snippet from '${REPO_NAME}' (file: ${INPUT_FILE_PATH}). Incorporate *all* your suggested improvements regarding readability, maintainability, efficiency, error handling, security, and modern best practices (${LANG_HINT:-for the language}).

CRITICAL INSTRUCTION: Your entire response must consist *only* of the complete, final, enhanced code block, ready to be saved directly as the improved file. Do not include *any* introductory text, explanations, summaries, or markdown formatting outside the final code block itself. Use comments *within* the code only if necessary for clarity on specific changes.

Original Code:
\`\`\`${LANG_HINT}\n${ORIGINAL_CODE_CONTENT}\n\`\`\`"

# Lower temperature for more focused, less "creative" code generation
JSON_PAYLOAD=$(jq -n --arg prompt "$PROMPT_ENHANCE" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}')

RESPONSE_FINAL_FILE=$(mktemp "${TEMP_DIR}/response_final.XXXXXX")
HTTP_CODE=$(curl -s -w "%{http_code}" -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD" -X POST "$API_URL" -o "$RESPONSE_FINAL_FILE")

# --- Check API Response Status ---
if [[ "$HTTP_CODE" -ne 200 ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "❌ Error: API request failed (HTTP $HTTP_CODE)." >&2
  if [[ -f "$RESPONSE_FINAL_FILE" ]]; then
      echo "   API Response:" >&2
      cat "$RESPONSE_FINAL_FILE" >&2
  fi
  echo "--------------------------------------------------------------------" >&2
  exit 1
fi
echo "✅ API call successful (HTTP $HTTP_CODE)."

# --- Extract Final Code Block ---
echo "---"
echo "💾 Processing API response and extracting final code..."
RAW_FINAL_CONTENT=$(jq -r '.candidates[0].content.parts[0].text // empty' "$RESPONSE_FINAL_FILE")
if [[ $? -ne 0 ]] || [[ -z "$RAW_FINAL_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Successfully called API, but couldn't extract text or response was empty." >&2
    echo "   Raw response was in: $RESPONSE_FINAL_FILE (will be deleted)" >&2
    echo "--------------------------------------------------------------------" >&2
    # Save raw response as fallback
    echo "-- ERROR: Failed to extract valid text content from API response. --" > "$OUTPUT_FILE_NAME"
    cat "$RESPONSE_FINAL_FILE" >> "$OUTPUT_FILE_NAME"
    exit 1
fi

# Attempt to extract only the code block using awk
FINAL_CODE_CONTENT=$(echo "$RAW_FINAL_CONTENT" | \
  awk '/^\s*\`\`\`/{ in_block=1; next } /^\s*\`\`\`/{ exit } in_block{ print }')

# Check if awk actually extracted anything
if [[ -z "$FINAL_CODE_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Could not automatically extract final code block using awk (Gemini might not have used markdown fences \`\`\`)." >&2
    echo "   Saving the full text response instead. Review needed." >&2
    echo "--------------------------------------------------------------------" >&2
    FINAL_CODE_CONTENT="$RAW_FINAL_CONTENT" # Fallback
fi

# --- Save Final Code ---
echo "$FINAL_CODE_CONTENT" > "$OUTPUT_FILE_NAME"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to write final enhanced code to '$OUTPUT_FILE_NAME'." >&2
  exit 1
fi

echo "--------------------------------------------------------------------"
echo "✅ Success! Final enhanced code saved to:"
if command -v realpath >/dev/null 2>&1; then echo "   $(realpath "$OUTPUT_FILE_NAME")"; else echo "   $(pwd)/$OUTPUT_FILE_NAME"; fi
echo "--------------------------------------------------------------------"

# Trap handles cleanup
exit 0