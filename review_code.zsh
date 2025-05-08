#!/usr/bin/env zsh

# Zsh script to fetch code, ask Gemini to enhance it, and save the improved code.

# --- Configuration ---
REPO_URL="https://github.com/Mentallyspammed1/worldguide.git"
REPO_NAME="worldguide" # Used for messages
TEMP_DIR_BASE="temp_repo_enhance_zsh" # Base for temporary directory
# Create a unique temp directory name
TEMP_DIR=$(mktemp -d "${TEMP_DIR_BASE}_XXXXXX" 2>/dev/null || echo "${TEMP_DIR_BASE}_$$")


# --- Function for cleanup ---
cleanup() {
  # echo -e "\n🧹 Cleaning up temporary directory ($TEMP_DIR)..." # Optional: Add newline for clarity
  echo "🧹 Cleaning up temporary directory ($TEMP_DIR)..."
  rm -rf "$TEMP_DIR"
}

# --- Ensure cleanup runs on exit, interrupt, or termination ---
# Use zsh's trap syntax if needed, but EXIT often covers enough
# zstyle ':completion:*' completer _complete _ignored _approximate
# setopt PROMPT_SUBST
# trap cleanup EXIT INT TERM
# Simpler trap for broad compatibility:
trap 'cleanup' EXIT TERM INT

# --- Check for required tools ---
echo "⚙️ Checking for required tools (curl, jq, git)..."
missing_tools=() # Use zsh array
command -v curl > /dev/null 2>&1 || missing_tools+=("curl")
command -v jq > /dev/null 2>&1 || missing_tools+=("jq")
command -v git > /dev/null 2>&1 || missing_tools+=("git")

if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "❌ Error: Required tools are missing: ${missing_tools[*]}." >&2 # Print to stderr
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
# Zsh's read -p equivalent using ?
read "?Enter the path to the file inside '$REPO_NAME' to enhance (e.g., main.py): " INPUT_FILE_PATH
read "?Enter the desired name for the output file (e.g., enhanced_main.py): " OUTPUT_FILE_NAME

# --- Validate Input ---
if [[ -z "$INPUT_FILE_PATH" ]]; then
  echo "❌ Error: No input file path provided." >&2
  exit 1
fi
# Basic check to prevent using absolute paths or traversing up
if [[ "$INPUT_FILE_PATH" == /* ]] || [[ "$INPUT_FILE_PATH" == *..* ]]; then
    echo "❌ Error: Please provide a relative path within the repository (e.g., main.py or src/file.py)." >&2
    exit 1
fi
if [[ -z "$OUTPUT_FILE_NAME" ]]; then
  echo "❌ Error: No output file name provided." >&2
  exit 1
fi

# --- Clone the repository ---
echo "---"
echo "🌐 Cloning repository: $REPO_URL ..."
# Clean up just before cloning if mktemp failed and we reused a name
[[ "$TEMP_DIR" == "${TEMP_DIR_BASE}_$$" ]] && rm -rf "$TEMP_DIR"

# Use --depth 1 for faster clone
git clone --depth 1 "$REPO_URL" "$TEMP_DIR"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to clone repository '$REPO_URL'." >&2
  exit 1 # Cleanup will be handled by trap
fi
echo "✅ Repository cloned successfully into '$TEMP_DIR'."

# --- Check if the input file exists ---
FULL_INPUT_PATH="$TEMP_DIR/$INPUT_FILE_PATH"
echo "---"
echo "🔍 Checking for file: $FULL_INPUT_PATH"
if [[ ! -f "$FULL_INPUT_PATH" ]]; then
  echo "❌ Error: File '$INPUT_FILE_PATH' not found within the cloned repository." >&2
  echo "   Please check the path and try again." >&2
  exit 1 # Cleanup will be handled by trap
fi
echo "✅ Found file: $INPUT_FILE_PATH"

# --- Read and Format Code Content ---
echo "---"
echo "📑 Reading and formatting code content..."
# Read file, escape backslashes FIRST, then double quotes, then handle newlines for JSON
# Using standard tools for portability
CODE_CONTENT=$(cat "$FULL_INPUT_PATH" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to read or format file content from '$FULL_INPUT_PATH'." >&2
  exit 1 # Cleanup will be handled by trap
fi

# Detect language based on extension for better prompting (simple version)
# Zsh parameter expansion :e gives the extension
FILE_EXT=${INPUT_FILE_PATH:e}
LANG_HINT=""
case $FILE_EXT in
  py) LANG_HINT="python" ;;
  js) LANG_HINT="javascript" ;;
  sh|zsh) LANG_HINT="shell" ;; # Use 'shell' or 'bash' usually works
  *) LANG_HINT="" ;; # Let Gemini infer if unknown
esac
echo "✅ Code formatted for API. Detected language hint: ${LANG_HINT:-'none'}."

# --- Make the API Call ---
echo "---"
echo "🚀 Sending request to Gemini API for enhancement..."
API_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent?key=${GEMINI_API_KEY}"

# Construct the specific prompt asking for the complete enhanced code
PROMPT_TEXT="Please analyze, upgrade, and enhance the following code snippet from the repository '${REPO_NAME}' (file: ${INPUT_FILE_PATH}). \
Focus on improving: readability, maintainability, efficiency, performance, robustness, error handling, security best practices, and usage of modern language features/idioms for ${LANG_HINT:-the detected language}.

IMPORTANT: Your response should consist *only* of the complete, final, enhanced code block. Do not include explanations outside the code block itself (use comments within the code if needed). Ensure the output is ready to be saved directly as the improved file.

\`\`\`${LANG_HINT}\n${CODE_CONTENT}\n\`\`\`"

# Create JSON payload using jq for robust quoting
JSON_PAYLOAD=$(jq -n --arg prompt "$PROMPT_TEXT" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}') # Lower temperature for more deterministic code generation

# Temporary file for the response body
RESPONSE_BODY_FILE=$(mktemp "${TEMP_DIR}/response_body.XXXXXX")

# Perform the curl request, capture HTTP status code separately
HTTP_CODE=$(curl -s -w "%{http_code}" \
     -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD" \
     -X POST "$API_URL" \
     -o "$RESPONSE_BODY_FILE") # Save body to temp file

# --- Check API Response Status ---
if [[ "$HTTP_CODE" -ne 200 ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "❌ Error: API request failed with HTTP status code $HTTP_CODE." >&2
  if [[ -f "$RESPONSE_BODY_FILE" ]]; then
    echo "   API Response Body:" >&2
    cat "$RESPONSE_BODY_FILE" >&2
  else
    echo "   No response body received or saved." >&2
  fi
   echo "--------------------------------------------------------------------" >&2
  # No need to rm RESPONSE_BODY_FILE here, cleanup trap handles it
  exit 1 # Cleanup will be handled by trap
fi

echo "✅ API call successful (HTTP $HTTP_CODE)."

# --- Extract Code Block and Save to Output File ---
echo "---"
echo "💾 Processing API response and extracting enhanced code..."

# Use jq to get the main text content
RAW_TEXT_CONTENT=$(jq -r '.candidates[0].content.parts[0].text // empty' "$RESPONSE_BODY_FILE")
JQ_EXIT_CODE=$?

if [[ $JQ_EXIT_CODE -ne 0 ]] || [[ -z "$RAW_TEXT_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Successfully called API, but couldn't extract text content using jq or the response text was empty." >&2
    echo "   Raw Response Body was saved to: $RESPONSE_BODY_FILE (will be deleted on exit)" >&2
    echo "   Please check the API key, quota, and the structure of the API response." >&2
    echo "--------------------------------------------------------------------" >&2
    # Save the raw text or an error message
    echo "-- ERROR: Failed to extract valid text content from API response. --" > "$OUTPUT_FILE_NAME"
    cat "$RESPONSE_BODY_FILE" >> "$OUTPUT_FILE_NAME" # Append raw response for debugging
    exit 1 # Indicate failure, main cleanup via trap
fi

# Attempt to extract only the code block using awk
# This looks for the first ``` block and prints lines between the first and second ``` marker.
# It handles optional language hints on the first marker (e.g., ```python)
# It assumes the primary output IS the code block as requested in the prompt.
EXTRACTED_CODE=$(echo "$RAW_TEXT_CONTENT" | \
  awk 'BEGIN{in_block=0} /^\s*\`\`\`/{ if(!in_block){in_block=1; next} else {exit} } in_block{print}')

# Check if awk actually extracted anything (it might fail if format is unexpected)
if [[ -z "$EXTRACTED_CODE" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Could not automatically extract code block using awk." >&2
    echo "   Saving the full text response from Gemini instead." >&2
    echo "   You may need to manually copy the code from '$OUTPUT_FILE_NAME'." >&2
    echo "--------------------------------------------------------------------" >&2
    # Fallback to saving the raw text content received
    EXTRACTED_CODE="$RAW_TEXT_CONTENT"
fi

# Save the extracted code (or the full response as fallback)
echo "$EXTRACTED_CODE" > "$OUTPUT_FILE_NAME"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to write enhanced code to output file '$OUTPUT_FILE_NAME'." >&2
  exit 1 # Cleanup will be handled by trap
fi

echo "--------------------------------------------------------------------"
echo "✅ Success! Enhanced code saved to:"
echo "   $(pwd)/$OUTPUT_FILE_NAME" # Show full path
echo "--------------------------------------------------------------------"

# Trap will handle the final cleanup
exit 0