#!/data/data/com.termux/files/usr/bin/bash

# Bash script for Termux to enhance a local code file using Gemini API,
# requesting the complete enhanced code block incorporating all improvements.

# --- Configuration ---
# Use a capable model like 1.5 Pro for better results on complex tasks
API_MODEL="gemini-2.5-pro-exp-03-25"
# API_MODEL="gemini-1.0-pro" # Alternative stable model

# Temporary file for the API response body
RESPONSE_BODY_FILE="gemini_response_body.tmp.$$" # Add PID for basic uniqueness

# --- Function for cleanup ---
# Cleans up the temporary file created during execution
cleanup() {
  # echo "🧹 Cleaning up temporary file ($RESPONSE_BODY_FILE)..." # Optional debug message
  rm -f "$RESPONSE_BODY_FILE"
}

# --- Ensure cleanup runs on script exit or interruption ---
trap cleanup EXIT TERM INT

# --- Check for required tools ---
echo "⚙️ Checking for required tools (curl, jq)..."
missing_tools=""
command -v curl >/dev/null 2>&1 || missing_tools+=" curl"
command -v jq >/dev/null 2>&1 || missing_tools+=" jq"

if [[ -n "$missing_tools" ]]; then
    # Print errors to standard error
    echo "❌ Error: Required tools are missing:$missing_tools." >&2
    echo "   Please install them using: pkg install$missing_tools" >&2
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
  echo "   🚨 IMPORTANT: NEVER paste your API key directly into scripts." >&2
  echo "   🚨          Ensure your previously exposed key is REVOKED." >&2
  echo "--------------------------------------------------------------------" >&2
  exit 1
fi
echo "🔑 Gemini API Key found in environment."

# --- Get User Input ---
echo "---"
read -p "Enter the full path to the local code file to enhance: " INPUT_FILE_PATH
read -p "Enter the desired name for the final enhanced output file: " OUTPUT_FILE_NAME

# --- Validate Input ---
if [[ -z "$INPUT_FILE_PATH" ]]; then
    echo "❌ Error: No input file path provided." >&2
    exit 1
fi
if [[ ! -f "$INPUT_FILE_PATH" ]]; then
    echo "❌ Error: Input file not found at '$INPUT_FILE_PATH'." >&2
    exit 1
fi
if [[ ! -r "$INPUT_FILE_PATH" ]]; then
    echo "❌ Error: Cannot read input file '$INPUT_FILE_PATH'. Check permissions." >&2
    exit 1
fi
if [[ -z "$OUTPUT_FILE_NAME" ]]; then
  echo "❌ Error: No output file name provided." >&2
  exit 1
fi

# --- Read and Format Original Code Content ---
echo "---"
echo "📑 Reading and formatting code content from '$INPUT_FILE_PATH'..."
# Escape backslashes, then quotes, then newlines. Standard sed loop is generally reliable.
ORIGINAL_CODE_CONTENT=$(cat "$INPUT_FILE_PATH" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
# Check status of the pipeline
if [[ ${PIPESTATUS[0]} -ne 0 ]] || [[ ${PIPESTATUS[1]} -ne 0 ]] || [[ ${PIPESTATUS[2]} -ne 0 ]]; then
  echo "❌ Error: Failed to read or format file content from '$INPUT_FILE_PATH'." >&2
  exit 1
fi

# --- Detect Language Hint (Optional but helpful) ---
FILE_EXT="${INPUT_FILE_PATH##*.}"
LANG_HINT=""
case "$FILE_EXT" in
  py) LANG_HINT="python" ;;
  js) LANG_HINT="javascript" ;;
  sh|bash) LANG_HINT="bash" ;;
  *) LANG_HINT="" ;;
esac
echo "✅ Code formatted. Language hint: ${LANG_HINT:-'none'}."


# ================================================================
# API Call: Analyze, Enhance, Implement, and Return Final Code
# ================================================================
echo "---"
echo "🚀 Sending request to Gemini API ($API_MODEL) to analyze, enhance, and return final code..."
API_URL="https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"

# Explicit prompt asking for analysis, enhancement, and ONLY the final code block
PROMPT_ENHANCE="Please perform a thorough analysis of the following code snippet (from local file: ${INPUT_FILE_PATH##*/}). Identify areas for upgrades and enhancements related to readability, maintainability, efficiency, performance, robustness, error handling, security best practices, and the use of modern language features or idioms (${LANG_HINT:-for the language}). After your internal analysis, incorporate *all* identified improvements into the code.

CRITICAL INSTRUCTION: Your entire response must consist *only* of the complete, final, enhanced code block, ready to be saved directly as the improved file. Do not include *any* introductory text, explanations, summaries, or markdown formatting outside the final code block itself. Use comments *within* the code only if necessary for clarity on specific changes.

Original Code:
\`\`\`${LANG_HINT}\n${ORIGINAL_CODE_CONTENT}\n\`\`\`"

# Create JSON payload using jq for robust quoting
# Use lower temperature for more focused code generation based on analysis
JSON_PAYLOAD=$(jq -n --arg prompt "$PROMPT_ENHANCE" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}')
if [[ $? -ne 0 ]]; then
    echo "❌ Error: Failed to create JSON payload using jq." >&2
    exit 1
fi

# Perform the curl request, capture HTTP status code separately
# Use -L to follow redirects
HTTP_CODE=$(curl -L -s -w "%{http_code}" \
     -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD" \
     -X POST "$API_URL" \
     -o "$RESPONSE_BODY_FILE") # Save body to temp file

# --- Check API Response Status ---
if [[ "$HTTP_CODE" -ne 200 ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "❌ Error: API request failed (HTTP $HTTP_CODE)." >&2
  if [[ -f "$RESPONSE_BODY_FILE" ]] && [[ -r "$RESPONSE_BODY_FILE" ]]; then
      echo "   API Response:" >&2
      cat "$RESPONSE_BODY_FILE" >&2 # Print error response body
  else
      echo "   Could not read API response body file." >&2
  fi
  echo "--------------------------------------------------------------------" >&2
  exit 1 # Trap will cleanup RESPONSE_BODY_FILE
fi
echo "✅ API call successful (HTTP $HTTP_CODE)."

# --- Extract Final Code Block ---
echo "---"
echo "💾 Processing API response and extracting final code..."
# Check if response file exists and is readable before parsing
if [[ ! -r "$RESPONSE_BODY_FILE" ]]; then
    echo "❌ Error: Cannot read API response file '$RESPONSE_BODY_FILE'." >&2
    exit 1
fi

RAW_FINAL_CONTENT=$(jq -r '.candidates[0].content.parts[0].text // ""' "$RESPONSE_BODY_FILE")
JQ_EXIT_CODE=$?

# Check jq exit code and if content is empty
if [[ $JQ_EXIT_CODE -ne 0 ]] || [[ -z "$RAW_FINAL_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Successfully called API, but failed to extract text content using jq or the response text was empty." >&2
    echo "   jq exit code: $JQ_EXIT_CODE" >&2
    echo "   Raw response was in: $RESPONSE_BODY_FILE (will be deleted by cleanup)" >&2
    echo "--------------------------------------------------------------------" >&2
    # Save raw response as fallback
    echo "-- ERROR: Failed to extract valid text content from API response. Raw response follows: --" > "$OUTPUT_FILE_NAME"
    cat "$RESPONSE_BODY_FILE" >> "$OUTPUT_FILE_NAME" # Append raw response for debugging
    exit 1 # Indicate failure
fi

# Attempt to extract only the code block using awk
# Looks for the first ``` block and prints lines between the first and second ``` marker.
FINAL_CODE_CONTENT=$(echo "$RAW_FINAL_CONTENT" | \
  awk 'BEGIN{in_block=0} /^\s*```/{ if(!in_block){in_block=1; next} else {exit} } in_block{print}')

# Check if awk actually extracted anything (it might fail if format is unexpected)
if [[ -z "$FINAL_CODE_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Could not automatically extract final code block using awk (Gemini might not have used markdown fences \`\`\`)." >&2
    echo "   Saving the full text response received from Gemini instead. Manual review needed." >&2
    echo "--------------------------------------------------------------------" >&2
    FINAL_CODE_CONTENT="$RAW_FINAL_CONTENT" # Fallback to using the full text
fi

# --- Save Final Code ---
# Use printf for potentially safer output than echo
printf "%s\n" "$FINAL_CODE_CONTENT" > "$OUTPUT_FILE_NAME"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to write final enhanced code to '$OUTPUT_FILE_NAME'. Check permissions or disk space." >&2
  exit 1
fi

echo "--------------------------------------------------------------------"
echo "✅ Success! Final enhanced code saved to:"
# Use pwd for current directory context
echo "   $(pwd)/$OUTPUT_FILE_NAME"
echo "--------------------------------------------------------------------"

# Trap handles cleanup
exit 0
