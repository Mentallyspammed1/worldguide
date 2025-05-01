#!/data/data/com.termux/files/usr/bin/bash

# Bash script for Termux to enhance a local code file using Gemini API,
# return the complete enhanced code, and optionally attempt to run it.

# --- Configuration ---
API_MODEL="gemini-2.5-pro-exp-03-25" # Use a capable model
# API_MODEL="gemini-2.5-pro-exp-03-25"

# Temporary file for the API response body
RESPONSE_BODY_FILE="gemini_response_body.tmp.$$" # Basic unique temp filename

# --- Function for cleanup ---
cleanup() {
  # echo "🧹 Cleaning up temporary file ($RESPONSE_BODY_FILE)..." # Optional debug
  rm -f "$RESPONSE_BODY_FILE"
}

# --- Ensure cleanup runs on script exit or interruption ---
trap cleanup EXIT TERM INT

# --- Check for required tools ---
echo "⚙️ Checking for required tools (curl, jq)..."
missing_tools=""
command -v curl >/dev/null 2>&1 || missing_tools+=" curl"
command -v jq >/dev/null 2>&1 || missing_tools+=" jq"
# Check for potential interpreters needed later (optional, but good practice)
# command -v python >/dev/null 2>&1 || echo "⚠️ Warning: 'python' interpreter not found." >&2
# command -v node >/dev/null 2>&1 || echo "⚠️ Warning: 'node' (for JavaScript) interpreter not found." >&2
# command -v bash >/dev/null 2>&1 || echo "⚠️ Warning: 'bash' interpreter not found (should be present!)." >&2


if [[ -n "$missing_tools" ]]; then
    echo "❌ Error: Required tools are missing:$missing_tools." >&2
    echo "   Please install them using: pkg install$missing_tools" >&2
    exit 1
fi
echo "✅ Tools found."

# --- Check for API Key (Critical Security Step) ---
if [[ -z "$GEMINI_API_KEY" ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "🔒 Error: GEMINI_API_KEY environment variable is not set." >&2
  # ...(rest of security message)...
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
if [[ -z "$INPUT_FILE_PATH" ]] || [[ ! -f "$INPUT_FILE_PATH" ]] || [[ ! -r "$INPUT_FILE_PATH" ]]; then
    echo "❌ Error: Invalid or unreadable input file path '$INPUT_FILE_PATH'." >&2
    exit 1
fi
if [[ -z "$OUTPUT_FILE_NAME" ]]; then
  echo "❌ Error: No output file name provided." >&2
  exit 1
fi

# --- Read and Format Original Code Content ---
echo "---"
echo "📑 Reading and formatting code content from '$INPUT_FILE_PATH'..."
ORIGINAL_CODE_CONTENT=$(cat "$INPUT_FILE_PATH" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed -z 's/\n/\\n/g')
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to read/format file content from '$INPUT_FILE_PATH'." >&2
  exit 1
fi

# --- Detect Language Hint ---
FILE_EXT="${INPUT_FILE_PATH##*.}"
LANG_HINT=""
INTERPRETER="" # Variable to hold the command to run the code
case "$FILE_EXT" in
  py) LANG_HINT="python"; INTERPRETER="python" ;;
  js) LANG_HINT="javascript"; INTERPRETER="node" ;;
  sh|bash) LANG_HINT="bash"; INTERPRETER="bash" ;;
  *) LANG_HINT=""; INTERPRETER="" ;; # Unknown language
esac
echo "✅ Code formatted. Language hint: ${LANG_HINT:-'none'}."
if [[ -n "$INTERPRETER" ]]; then
    echo "   Detected potential interpreter: $INTERPRETER"
else
    echo "   ⚠️ Could not determine interpreter based on file extension for run attempt."
fi


# ================================================================
# API Call: Analyze, Enhance, Implement, and Return Final Code
# ================================================================
echo "---"
echo "🚀 Sending request to Gemini API ($API_MODEL) to enhance and return final code..."
API_URL="https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"

# Enhanced prompt asking model to think internally about suggestions before implementing
PROMPT_ENHANCE="Please thoroughly analyze the following code snippet (from local file: ${INPUT_FILE_PATH##*/}). First, internally consider improvements and enhancements regarding readability, maintainability, efficiency, error handling, security, and modern best practices (${LANG_HINT:-for the language}). Then, incorporate *all* those improvements into the code.

CRITICAL INSTRUCTION: Your entire response must consist *only* of the complete, final, enhanced code block, ready to be saved directly as the improved file. Do not include *any* introductory text, explanations, summaries, or markdown formatting outside the final code block itself. Use comments *within* the code only if necessary for clarity on specific changes.

Original Code:
\`\`\`${LANG_HINT}\n${ORIGINAL_CODE_CONTENT}\n\`\`\`"

JSON_PAYLOAD=$(jq -n --arg prompt "$PROMPT_ENHANCE" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}')
if [[ $? -ne 0 ]]; then echo "❌ Error: Failed to create JSON payload." >&2; exit 1; fi

HTTP_CODE=$(curl -L -s -w "%{http_code}" \
     -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD" -X POST "$API_URL" -o "$RESPONSE_BODY_FILE")

# --- Check API Response Status ---
if [[ "$HTTP_CODE" -ne 200 ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "❌ Error: API request failed (HTTP $HTTP_CODE)." >&2
  [[ -f "$RESPONSE_BODY_FILE" ]] && cat "$RESPONSE_BODY_FILE" >&2
  echo "--------------------------------------------------------------------" >&2
  exit 1
fi
echo "✅ API call successful (HTTP $HTTP_CODE)."

# --- Extract Final Code Block ---
echo "---"
echo "💾 Processing API response and extracting final code..."
if [[ ! -r "$RESPONSE_BODY_FILE" ]]; then echo "❌ Error: Cannot read API response file." >&2; exit 1; fi

RAW_FINAL_CONTENT=$(jq -r '.candidates[0].content.parts[0].text // ""' "$RESPONSE_BODY_FILE")
JQ_EXIT_CODE=$?

if [[ $JQ_EXIT_CODE -ne 0 ]] || [[ -z "$RAW_FINAL_CONTENT" ]]; then
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Warning: Failed to extract text content via jq or response was empty (jq exit: $JQ_EXIT_CODE)." >&2
    echo "   Raw response was in: $RESPONSE_BODY_FILE (will be deleted)" >&2
    echo "--------------------------------------------------------------------" >&2
    echo "-- ERROR: Failed to extract valid text content from API response. Raw response follows: --" > "$OUTPUT_FILE_NAME"
    cat "$RESPONSE_BODY_FILE" >> "$OUTPUT_FILE_NAME" # Save raw for debugging
    exit 1
fi

# Attempt to extract only the code block using awk
FINAL_CODE_CONTENT=$(echo "$RAW_FINAL_CONTENT" | \
  awk 'BEGIN{in_block=0} /^\s*```/{ if(!in_block){in_block=1; next} else {exit} } in_block{print}')

if [[ -z "$FINAL_CODE_CONTENT" ]]; then
    echo "⚠️ Warning: Could not automatically extract final code block (Gemini might not have used \`\`\`). Saving full response." >&2
    FINAL_CODE_CONTENT="$RAW_FINAL_CONTENT" # Fallback
fi

# --- Save Final Code ---
printf "%s\n" "$FINAL_CODE_CONTENT" > "$OUTPUT_FILE_NAME"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed to write final enhanced code to '$OUTPUT_FILE_NAME'." >&2
  exit 1
fi

echo "--------------------------------------------------------------------"
echo "✅ Success! Final enhanced code saved to:"
echo "   $(pwd)/$OUTPUT_FILE_NAME"
echo "--------------------------------------------------------------------"


# ================================================================
# Optional: Attempt to Run the Generated Code
# ================================================================
echo # Add a newline for spacing
echo "--- ⚠️ Optional: Attempt to Run Enhanced Code ---"
echo "🚨🚨🚨 WARNING: Running AI-generated code can be VERY RISKY! 🚨🚨🚨"
echo "   It could contain errors, malicious commands, or cause unexpected behavior."
echo "   Review the code in '$OUTPUT_FILE_NAME' first if unsure."
echo "────────────────────────────────────────────────────────────────────"

read -p "Do you want to attempt to run the enhanced code ('$OUTPUT_FILE_NAME')? (y/N): " confirm_run

if [[ "${confirm_run,,}" == "y" ]]; then
    if [[ -n "$INTERPRETER" ]]; then
        # Check if interpreter exists
        if command -v "$INTERPRETER" >/dev/null 2>&1; then
            echo "🚀 Attempting to run with '$INTERPRETER $OUTPUT_FILE_NAME'..."
            echo "--- Code Output Start ---"
            # Execute the command. Output (stdout & stderr) will be displayed directly.
            "$INTERPRETER" "$OUTPUT_FILE_NAME"
            run_exit_code=$? # Capture the exit code of the executed command
            echo "--- Code Output End ---"
            if [[ $run_exit_code -eq 0 ]]; then
                echo "✅ Code execution finished successfully (Exit Code: 0)."
            else
                echo "⚠️ Code execution finished with errors (Exit Code: $run_exit_code)."
            fi
        else
             echo "❌ Error: Interpreter '$INTERPRETER' not found. Cannot run the code." >&2
        fi
    else
        echo "❌ Error: Could not determine the correct interpreter for '$OUTPUT_FILE_NAME' based on extension." >&2
        echo "   Cannot attempt to run the code automatically." >&2
    fi
else
    echo "⏩ Skipping code execution step."
fi
echo "--------------------------------------------------------------------"


# Trap handles cleanup
exit 0