#!/usr/bin/env zsh

# Zsh script for Termux/Linux to:
# 1. Enhance local code via Gemini API (Initial Pass).
# 2. [USER CONFIRMS RISK] Optionally RUN the enhanced code and capture output/errors.
# 3. Send code AND execution results to Gemini for final enhancement based on runtime.
# 4. Save the final, refined code.
#
# 🚨🚨🚨 EXTREMELY RISKY SCRIPT - READ ALL WARNINGS CAREFULLY BEFORE USE 🚨🚨🚨
#      AUTOMATIC EXECUTION OF AI-GENERATED CODE CAN CAUSE SEVERE DAMAGE.

# --- Configuration ---
API_MODEL="gemini-1.5-pro-latest" # Needs a capable model for multi-step reasoning

# --- Temporary File/Directory Setup ---
# Create a temporary directory using mktemp for better security and cleanup
TEMP_DIR=$(mktemp -d "gemini_enhance_run.XXXXXX" 2>/dev/null)
# Fallback if mktemp -d fails (less secure, avoid if possible)
if [[ -z "$TEMP_DIR" ]] || [[ ! -d "$TEMP_DIR" ]]; then
    TIMESTAMP=$(date +%s)
    TEMP_DIR="gemini_enhance_run_${TIMESTAMP}_$$"
    echo "⚠️ Warning: mktemp -d failed, using potentially less secure fallback directory: $TEMP_DIR" >&2
    mkdir -p "$TEMP_DIR" || { echo "❌ Error: Cannot create temporary directory '$TEMP_DIR'." >&2; exit 1; }
fi

# Define temporary file paths within the temp directory
RESPONSE_BODY_FILE_1="${TEMP_DIR}/response1.json"
RESPONSE_BODY_FILE_2="${TEMP_DIR}/response2.json"
INTERMEDIATE_CODE_FILE="${TEMP_DIR}/intermediate_code.tmp"
EXECUTION_OUTPUT_FILE="${TEMP_DIR}/execution_output.log"

# --- Function for cleanup ---
cleanup() {
  # Ensure TEMP_DIR exists before attempting removal
  if [[ -n "$TEMP_DIR" ]] && [[ -d "$TEMP_DIR" ]]; then
      echo "🧹 Cleaning up temporary directory ($TEMP_DIR)..."
      command rm -rf "$TEMP_DIR" # Use command prefix to bypass aliases if any
  else
      # Cleanup individual files if TEMP_DIR creation failed but files might exist
      rm -f "gemini_response_body.tmp.$$" \
            "intermediate_code_${TIMESTAMP:-0}_$$.tmp" \
            "execution_output_${TIMESTAMP:-0}_$$.tmp" \
            "response1.json" "response2.json" # Attempt cleanup anyway
  fi
}

# --- Ensure cleanup runs on script exit or interruption ---
trap cleanup EXIT TERM INT

# --- Check for required tools ---
echo "⚙️ Checking for required tools (curl, jq)..."
typeset -a missing_tools
command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
# Check for timeout utility (highly recommended for the run step)
TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then # macOS via coreutils
    TIMEOUT_CMD="gtimeout"
fi


if [[ ${#missing_tools[@]} -gt 0 ]]; then
    echo "❌ Error: Required tools missing: ${missing_tools[*]}." >&2
    echo "   Install with: pkg install ${missing_tools[*]}  (or equivalent)" >&2
    exit 1
fi
echo "✅ Tools found."
if [[ -z "$TIMEOUT_CMD" ]]; then
    echo "⚠️ Warning: 'timeout' command not found. Consider installing 'coreutils' (`pkg install coreutils`)." >&2
    echo "   The execution step will run without a time limit, increasing risk of hangs." >&2
fi

# --- Check for API Key ---
if [[ -z "$GEMINI_API_KEY" ]]; then
  echo "--------------------------------------------------------------------" >&2
  echo "🔒 Error: GEMINI_API_KEY environment variable is not set." >&2
  echo "   This script requires your Google AI Gemini API Key." >&2
  echo "   Set it using: export GEMINI_API_KEY='YOUR_SECURE_KEY'" >&2
  echo "   🚨 Ensure any previously exposed keys are REVOKED." >&2
  echo "--------------------------------------------------------------------" >&2
  exit 1
fi
echo "🔑 Gemini API Key found."

# --- Get User Input ---
echo "---"
# Use Zsh's prompt syntax
read "?Enter the full path to the local code file to enhance: " INPUT_FILE_PATH
read "?Enter the desired name for the FINAL enhanced output file: " FINAL_OUTPUT_FILE_NAME

# --- Validate Input ---
# Check file existence and readability
if [[ -z "$INPUT_FILE_PATH" ]] || [[ ! -f "$INPUT_FILE_PATH" ]] || [[ ! -r "$INPUT_FILE_PATH" ]]; then
    echo "❌ Error: Invalid or unreadable input file path '$INPUT_FILE_PATH'." >&2
    exit 1
fi
if [[ -z "$FINAL_OUTPUT_FILE_NAME" ]]; then
  echo "❌ Error: No final output file name provided." >&2
  exit 1
fi
# Check if output directory is writable (basic check)
OUTPUT_DIR=$(dirname "$FINAL_OUTPUT_FILE_NAME")
[[ "$OUTPUT_DIR" == "." ]] && OUTPUT_DIR=$PWD # Handle relative path in current dir
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "❌ Error: Output directory '$OUTPUT_DIR' is not writable." >&2
    exit 1
fi


# --- Read and Format Original Code ---
echo "---"
echo "📑 Reading and formatting original code from '$INPUT_FILE_PATH'..."
# Escape backslashes, quotes, newlines. Check pipeline status.
ORIGINAL_CODE_CONTENT=$(cat "$INPUT_FILE_PATH" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
if [[ $pipestatus[1] -ne 0 ]] || [[ $pipestatus[2] -ne 0 ]] || [[ $pipestatus[3] -ne 0 ]]; then
  echo "❌ Error: Failed to read or format original file content." >&2
  echo "   Pipeline exit codes: ${pipestatus[*]}" >&2
  exit 1
fi

# --- Detect Language Hint & Interpreter ---
FILE_EXT=${INPUT_FILE_PATH:e} # Zsh extension extraction
LANG_HINT=""
INTERPRETER=""
case "$FILE_EXT" in
  py) LANG_HINT="python"; INTERPRETER="python" ;;
  js) LANG_HINT="javascript"; INTERPRETER="node" ;;
  sh|zsh) LANG_HINT="shell"; INTERPRETER="zsh" ;; # Use zsh if it's a zsh script
  bash) LANG_HINT="bash"; INTERPRETER="bash" ;;
  *) LANG_HINT=""; INTERPRETER="" ;;
esac
echo "✅ Original code formatted. Language hint: ${LANG_HINT:-'none'}. Potential interpreter: ${INTERPRETER:-'unknown'}"

# ================================================================
# API Call 1: Initial Enhancement
# ================================================================
echo "---"
echo "🚀 Step 1: Sending request to Gemini for initial code enhancement..."
API_URL="https://generativelanguage.googleapis.com/v1beta/models/${API_MODEL}:generateContent?key=${GEMINI_API_KEY}"

PROMPT_ENHANCE_1="Please perform a thorough analysis of the following code snippet (from local file: ${INPUT_FILE_PATH##*/}). Identify areas for upgrades and enhancements (readability, efficiency, error handling, security, modern practices for ${LANG_HINT:-the language}). Incorporate *all* identified improvements into the code.

CRITICAL INSTRUCTION: Your entire response must consist *only* of the complete, final, enhanced code block, ready to be saved. Do not include *any* text outside the final code block.

Original Code:
\`\`\`${LANG_HINT}\n${ORIGINAL_CODE_CONTENT}\n\`\`\`"

JSON_PAYLOAD_1=$(jq -n --arg prompt "$PROMPT_ENHANCE_1" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}')
if [[ $? -ne 0 ]]; then echo "❌ Error: Failed creating JSON payload 1." >&2; exit 1; fi

# Perform curl, save output, check HTTP code
HTTP_CODE_1=$(curl -L -s -w "%{http_code}" -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD_1" -X POST "$API_URL" -o "$RESPONSE_BODY_FILE_1")
CURL_EXIT_CODE_1=$? # Capture curl's own exit code

# --- Check API Response 1 ---
if [[ $CURL_EXIT_CODE_1 -ne 0 ]]; then
    echo "❌ Error: curl command failed during API request 1 (Exit code: $CURL_EXIT_CODE_1)." >&2
    # Optionally print curl error details if available
    exit 1
elif [[ "$HTTP_CODE_1" -ne 200 ]]; then
  echo "❌ Error: API request 1 failed (HTTP $HTTP_CODE_1)." >&2
  # Check readability before catting
  [[ -f "$RESPONSE_BODY_FILE_1" ]] && [[ -r "$RESPONSE_BODY_FILE_1" ]] && cat "$RESPONSE_BODY_FILE_1" >&2
  exit 1
fi
echo "✅ API call 1 successful."

# --- Extract Intermediate Code ---
echo "💾 Processing response 1 and extracting intermediate code..."
if [[ ! -r "$RESPONSE_BODY_FILE_1" ]]; then echo "❌ Error: Cannot read API response 1 file '$RESPONSE_BODY_FILE_1'." >&2; exit 1; fi
RAW_CONTENT_1=$(jq -r '.candidates[0].content.parts[0].text // ""' "$RESPONSE_BODY_FILE_1")
JQ_EXIT_CODE_1=$?
if [[ $JQ_EXIT_CODE_1 -ne 0 ]] || [[ -z "$RAW_CONTENT_1" ]]; then
    echo "⚠️ Warning: Failed extracting text from response 1 (jq exit: $JQ_EXIT_CODE_1) or content empty." >&2
    echo "-- ERROR: Failed processing API response 1. Raw response follows: --" > "$FINAL_OUTPUT_FILE_NAME"
    # Check readability before catting
    [[ -r "$RESPONSE_BODY_FILE_1" ]] && cat "$RESPONSE_BODY_FILE_1" >> "$FINAL_OUTPUT_FILE_NAME"
    exit 1
fi

# Extract code block using awk
INTERMEDIATE_CODE=$(echo "$RAW_CONTENT_1" | \
  awk 'BEGIN{in_block=0} /^\s*```/{ if(!in_block){in_block=1; next} else {exit} } in_block{print}')
if [[ -z "$INTERMEDIATE_CODE" ]]; then
    echo "⚠️ Warning: Could not extract intermediate code block via awk. Using full response 1." >&2
    INTERMEDIATE_CODE="$RAW_CONTENT_1"
fi

# Save intermediate code to temporary file
# Use print -r -- for safety
print -r -- "$INTERMEDIATE_CODE" > "$INTERMEDIATE_CODE_FILE"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed writing intermediate code to '$INTERMEDIATE_CODE_FILE'." >&2
  exit 1
fi
echo "✅ Intermediate enhanced code saved for review to:"
echo "   '$INTERMEDIATE_CODE_FILE'"

# ================================================================
# Step 2: Optionally Run the Intermediate Code
# ================================================================
# Use local scope for variables primarily used in this block
local EXECUTION_PERFORMED=false
local EXECUTION_EXIT_CODE=-1 # Default: -1 = skipped by user
local EXECUTION_OUTPUT=""
local confirm_run=""

echo # Newline for spacing
print -P "%F{red}%B********************************************************************%b%f"
print -P "%F{red}%B***          🚨🚨🚨 DANGER ZONE: CODE EXECUTION 🚨🚨🚨         ***%b%f"
print -P "%F{red}%B***                                                              ***%b%f"
print -P "%F{yellow}%B***   You are about to execute AI-generated code located at:   ***%b%f"
echo     "       '$INTERMEDIATE_CODE_FILE'"
print -P "%F{yellow}%B***                                                              ***%b%f"
print -P "%F{yellow}%B***   Executing this code carries EXTREME RISKS, including:    ***%b%f"
print -P "%F{yellow}%B***   - DATA LOSS (file deletion, corruption)                  ***%b%f"
print -P "%F{yellow}%B***   - SYSTEM INSTABILITY (crashes, resource exhaustion)      ***%b%f"
print -P "%F{yellow}%B***   - SECURITY BREACHES (malware download/execution)         ***%b%f"
print -P "%F{yellow}%B***   - UNPREDICTABLE and potentially HARMFUL side effects.    ***%b%f"
print -P "%F{yellow}%B***                                                              ***%b%f"
print -P "%F{white}%B***   RECOMMENDATION:                                          ***%b%f"
print -P "%F{white}%B***   1. Press Ctrl+C NOW to abort the script.                 ***%b%f"
print -P "%F{white}%B***   2. MANUALLY REVIEW the code in the file shown above.     ***%b%f"
print -P "%F{white}%B***   3. Only proceed if you fully understand the code AND     ***%b%f"
print -P "%F{white}%B***      accept ALL associated risks.                          ***%b%f"
print -P "%F{red}%B********************************************************************%b%f"
read "?Type 'yes' (lowercase) and press Enter to execute the code, or anything else to skip: " confirm_run


if [[ "$confirm_run" == "yes" ]]; then
    # Check again if interpreter is known
    if [[ -z "$INTERPRETER" ]]; then
        echo "❌ Error: Cannot determine interpreter. Skipping execution." >&2
        EXECUTION_OUTPUT="Interpreter unknown, execution skipped."
        EXECUTION_EXIT_CODE=-3 # Specific code for unknown interpreter
    # Check if interpreter command exists
    elif ! command -v "$INTERPRETER" >/dev/null 2>&1; then
         echo "❌ Error: Interpreter '$INTERPRETER' not found. Cannot run." >&2
         EXECUTION_OUTPUT="Interpreter '$INTERPRETER' not found, execution skipped."
         EXECUTION_EXIT_CODE=-2 # Specific code for interpreter not found
    else
        # Interpreter exists, proceed with execution attempt
        echo "🚀 Attempting to run intermediate code with '$INTERPRETER'..."
        # Ensure script has execute permissions if it's a shell type
        if [[ "$LANG_HINT" == "shell" ]] || [[ "$LANG_HINT" == "bash" ]]; then
            chmod +x "$INTERMEDIATE_CODE_FILE" || echo "⚠️ Warning: Failed to chmod +x intermediate script." >&2
        fi

        # Prepare command with optional timeout
        local cmd_to_run=("$INTERPRETER" "$INTERMEDIATE_CODE_FILE")
        if [[ -n "$TIMEOUT_CMD" ]]; then
             echo "   (Using 30 second timeout via '$TIMEOUT_CMD')"
             cmd_to_run=("$TIMEOUT_CMD" 30s "${cmd_to_run[@]}")
        else
             echo "   (Running without timeout limit)"
        fi

        # Execute, redirecting stdout and stderr to the output file
        # Run in a subshell to capture exit code correctly even with redirects
        ( "${cmd_to_run[@]}" > "$EXECUTION_OUTPUT_FILE" 2>&1 )
        EXECUTION_EXIT_CODE=$?
        EXECUTION_PERFORMED=true

        echo "--- Execution Output (stdout/stderr) ---"
        # Check if output file was created and readable
         if [[ -f "$EXECUTION_OUTPUT_FILE" ]] && [[ -r "$EXECUTION_OUTPUT_FILE" ]]; then
            # Using <() process substitution avoids subshell issues with variable assignment
            EXECUTION_OUTPUT=$(<"$EXECUTION_OUTPUT_FILE")
            cat "$EXECUTION_OUTPUT_FILE" # Show the user the output
        else
            echo "   (No output captured or output file '$EXECUTION_OUTPUT_FILE' unreadable)"
            EXECUTION_OUTPUT="(No output captured or file unreadable)"
        fi
        echo "----------------------------------------"

        # Report exit code meaning
        case $EXECUTION_EXIT_CODE in
          0) echo "✅ Intermediate code execution finished successfully (Exit Code: 0)." ;;
          124) echo "⚠️ Intermediate code execution TIMED OUT (Exit Code: 124)." ;;
          *) echo "⚠️ Intermediate code execution finished with errors/non-zero status (Exit Code: $EXECUTION_EXIT_CODE)." ;;
        esac

        # Escape EXECUTION_OUTPUT for JSON payload
        EXECUTION_OUTPUT=$(print -r -- "$EXECUTION_OUTPUT" | sed 's/\\/\\\\/g; s/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
        if [[ $pipestatus[1] -ne 0 ]] || [[ $pipestatus[2] -ne 0 ]] || [[ $pipestatus[3] -ne 0 ]]; then
             echo "❌ Error: Failed escaping execution output for JSON." >&2
             EXECUTION_OUTPUT="Error escaping execution output."
        fi
    fi # End interpreter checks
else
    # User did not type 'yes'
    echo "⏩ Skipping code execution step as requested by user."
    EXECUTION_OUTPUT="Execution was skipped by the user." # Provide context for next step
    EXECUTION_EXIT_CODE=-1 # Keep -1 for user skip
fi


# ================================================================
# API Call 2: Enhance based on Execution Output
# ================================================================
echo "---"
echo "🚀 Step 3: Sending intermediate code AND execution results to Gemini for final enhancement..."

PROMPT_ENHANCE_2="Please analyze the following:\
\n1. The intermediate code generated in a previous step.\
\n2. The output produced (or status message) when attempting to run that intermediate code.\
\n
\nBased ONLY on the execution output/status (look for errors, specific failure messages, timeouts, unexpected results, or confirmation of success/skip), further enhance the intermediate code. Focus specifically on fixing any errors revealed or addressing issues implied by the execution (e.g., potential unhandled cases suggested by output, performance issues if it timed out). If execution was skipped or successful and no runtime issues are apparent from the output, perform final refinements for clarity, robustness, or style IF appropriate based on the code itself, otherwise return the intermediate code largely unchanged.
\n
\nIntermediate Code:
\`\`\`${LANG_HINT}\n${INTERMEDIATE_CODE}\n\`\`\`
\n
Execution Results (Exit Code: ${EXECUTION_EXIT_CODE}):
--- BEGIN EXECUTION OUTPUT ---
${EXECUTION_OUTPUT:-"(No output captured or available)"}
--- END EXECUTION OUTPUT ---
\n
CRITICAL INSTRUCTION: Your entire response must consist *only* of the complete, final, improved code block based on this analysis. Do not include *any* text outside the final code block. Use comments *within* the code only if necessary."

JSON_PAYLOAD_2=$(jq -n --arg prompt "$PROMPT_ENHANCE_2" \
  '{contents: [{parts: [{text: $prompt}]}], "generationConfig": {"temperature": 0.3}}') # Keep temp reasonable for refinement
if [[ $? -ne 0 ]]; then echo "❌ Error: Failed creating JSON payload 2." >&2; exit 1; fi

# Perform curl, save output, check HTTP code
HTTP_CODE_2=$(curl -L -s -w "%{http_code}" -H 'Content-Type: application/json' \
     -d "$JSON_PAYLOAD_2" -X POST "$API_URL" -o "$RESPONSE_BODY_FILE_2")
CURL_EXIT_CODE_2=$?

# --- Check API Response 2 ---
if [[ $CURL_EXIT_CODE_2 -ne 0 ]]; then
    echo "❌ Error: curl command failed during API request 2 (Exit code: $CURL_EXIT_CODE_2)." >&2
    echo "⚠️ Saving INTERMEDIATE code to '$FINAL_OUTPUT_FILE_NAME' as fallback." >&2
    print -r -- "$INTERMEDIATE_CODE" > "$FINAL_OUTPUT_FILE_NAME"
    exit 1
elif [[ "$HTTP_CODE_2" -ne 200 ]]; then
  echo "❌ Error: API request 2 failed (HTTP $HTTP_CODE_2)." >&2
   # Check readability before catting
   [[ -f "$RESPONSE_BODY_FILE_2" ]] && [[ -r "$RESPONSE_BODY_FILE_2" ]] && cat "$RESPONSE_BODY_FILE_2" >&2
   echo "--------------------------------------------------------------------" >&2
   echo "⚠️ Saving INTERMEDIATE code to '$FINAL_OUTPUT_FILE_NAME' as fallback." >&2
   print -r -- "$INTERMEDIATE_CODE" > "$FINAL_OUTPUT_FILE_NAME"
  exit 1
fi
echo "✅ API call 2 successful."

# --- Extract Final Code ---
echo "💾 Processing response 2 and extracting final code..."
if [[ ! -r "$RESPONSE_BODY_FILE_2" ]]; then echo "❌ Error: Cannot read API response 2 file '$RESPONSE_BODY_FILE_2'." >&2; exit 1; fi
RAW_CONTENT_2=$(jq -r '.candidates[0].content.parts[0].text // ""' "$RESPONSE_BODY_FILE_2")
JQ_EXIT_CODE_2=$?
if [[ $JQ_EXIT_CODE_2 -ne 0 ]] || [[ -z "$RAW_CONTENT_2" ]]; then
    echo "⚠️ Warning: Failed extracting text from response 2 (jq exit: $JQ_EXIT_CODE_2) or content empty." >&2
    echo "--------------------------------------------------------------------" >&2
    echo "⚠️ Saving INTERMEDIATE code to '$FINAL_OUTPUT_FILE_NAME' as fallback." >&2
    print -r -- "$INTERMEDIATE_CODE" > "$FINAL_OUTPUT_FILE_NAME"
    exit 1 # Exit after saving fallback
fi

# Extract final code block using awk
FINAL_CODE=$(echo "$RAW_CONTENT_2" | \
  awk 'BEGIN{in_block=0} /^\s*```/{ if(!in_block){in_block=1; next} else {exit} } in_block{print}')
if [[ -z "$FINAL_CODE" ]]; then
    echo "⚠️ Warning: Could not extract final code block via awk from response 2. Using full response 2." >&2
    FINAL_CODE="$RAW_CONTENT_2"
fi

# --- Save Final Code ---
# Use print -r -- for safety
print -r -- "$FINAL_CODE" > "$FINAL_OUTPUT_FILE_NAME"
if [[ $? -ne 0 ]]; then
  echo "❌ Error: Failed writing FINAL enhanced code to '$FINAL_OUTPUT_FILE_NAME'. Check permissions." >&2
  # Optionally try saving intermediate as fallback again?
  exit 1
fi

echo "--------------------------------------------------------------------"
echo "✅ Success! Final enhanced code saved to:"
# Use PWD for current directory context
echo "   ${PWD}/$FINAL_OUTPUT_FILE_NAME"
echo "--------------------------------------------------------------------"

# Trap handles cleanup
exit 0