#!/data/data/com.termux/files/usr/bin/bash

# === Gemini Code Review Script v3.3 ===
# Target Model: gemini-2.5-pro-exp-03-25 (or override via GEMINI_MODEL env var)
# Enhancements:
# - Handles large input files by piping content to jq (avoids ARG_MAX limit, uses --rawfile).
# - Handles large JSON payloads by piping to curl (avoids ARG_MAX limit, uses --data @-).
# - Modular functions for clarity and maintenance.
# - Optional command-line arguments for input and output files.
# - API Key handling (environment variable or prompt).
# - Configurable model name.
# - Clear status messages and error handling.

# === Configuration ===
# Set the default model. Can be overridden by GEMINI_MODEL environment variable.
# --- MODIFIED FOR gemini-2.5-pro-exp-03-25 ---
readonly DEFAULT_MODEL="gemini-2.5-pro-exp-03-25"
# ---------------------------------------------
# Allow model override via environment variable
MODEL_NAME="${GEMINI_MODEL:-$DEFAULT_MODEL}"
readonly API_URL_BASE="https://generativelanguage.googleapis.com/v1beta/models"
readonly API_URL="${API_URL_BASE}/${MODEL_NAME}:generateContent"

# API Request settings
# Note: Newer models might support higher token limits. Adjust if needed/documented.
readonly MAX_OUTPUT_TOKENS=60000 # Standard high limit, adjust if model supports more
readonly TEMPERATURE=0.3        # Lower value for more deterministic code output
readonly TOP_P=1.0
readonly TOP_K=1

# Global variables that will be set by functions
GEMINI_API_KEY_INTERNAL=""
INPUT_FILENAME=""
CODE_CONTENT=""
JSON_PAYLOAD=""
API_RESPONSE=""
ENHANCED_CODE=""

# === Functions ===

# --- Print Error and Exit ---
error_exit() {
    echo "Error: $1" >&2
    exit "${2:-1}" # Default exit code 1 if not specified
}

# --- Check for Required Dependencies ---
check_dependencies() {
    echo "Checking dependencies..."
    local missing_dep=0
    if ! command -v curl &> /dev/null; then
        echo "Error: 'curl' command not found. Please install it: pkg install curl" >&2
        missing_dep=1
    fi
    if ! command -v jq &> /dev/null; then
        echo "Error: 'jq' command not found. Please install it: pkg install jq" >&2
        missing_dep=1
    fi
    if [[ "$missing_dep" -ne 0 ]]; then
        exit 1 # Exit directly from here
    fi
    echo "Dependencies satisfied."
}

# --- Get Gemini API Key ---
get_api_key() {
    local key
    # Check if API key is set as an environment variable
    if [[ -n "$GEMINI_API_KEY" ]]; then
        key="$GEMINI_API_KEY"
        echo "Using API Key from GEMINI_API_KEY environment variable."
    else
        # If not in env, prompt the user
        read -sp "Enter your Google Gemini API Key: " key
        echo # Add a newline after secret input
        if [[ -z "$key" ]]; then
            error_exit "Google Gemini API Key is required."
        fi
    fi
    # Set the global internal key variable
    GEMINI_API_KEY_INTERNAL="$key"
}

# --- Get Filename and Content ---
# Usage: get_filename_and_content <input_filename_arg>
# Returns: Assigns content to global variable CODE_CONTENT and filename to INPUT_FILENAME
get_filename_and_content() {
    local file_arg="$1"

    if [[ -n "$file_arg" ]]; then
        INPUT_FILENAME="$file_arg"
        echo "Using input file from argument: $INPUT_FILENAME"
    else
        read -p "Enter the filename containing the code to review: " INPUT_FILENAME
    fi

    if [[ -z "$INPUT_FILENAME" ]]; then
        error_exit "No filename provided."
    fi
    if [[ ! -f "$INPUT_FILENAME" ]]; then
        error_exit "File '$INPUT_FILENAME' not found." 2
    fi
    if [[ ! -r "$INPUT_FILENAME" ]]; then
        error_exit "Cannot read file '$INPUT_FILENAME'. Check permissions." 3
    fi

    echo "Reading content from '$INPUT_FILENAME'..."
    # Read potentially large file content
    CODE_CONTENT=$(cat "$INPUT_FILENAME")
    local read_status=$?
    if [[ $read_status -ne 0 ]]; then
       error_exit "Failed to read file content from '$INPUT_FILENAME'." 4
    fi

    if [[ -z "$CODE_CONTENT" ]]; then
        echo "Warning: File '$INPUT_FILENAME' is empty. Sending empty content to Gemini."
        # Decide if you want to exit or continue for empty files
        # error_exit "File '$INPUT_FILENAME' is empty."
    fi
}

# --- Build API Request Payload ---
# Usage: build_payload <prompt_instructions>
# Reads CODE_CONTENT from global scope via stdin piping (using --rawfile)
# Returns: Assigns JSON payload to global variable JSON_PAYLOAD
build_payload() {
    local prompt_instructions="$1" # Only the instruction text, not the code
    local payload_json

    # Construct the final prompt text within jq, combining instructions and code from stdin
    # Escaped backticks (\`) and newlines (\n) directly in the JSON string part of the jq command.
    local text_jq_expression='($prompt_instructions + "\\n\\nOriginal Code:\\n\\`\\`\\`\\n" + $code_from_stdin + "\\n\\`\\`\\`")'

    # Pipe the large CODE_CONTENT to jq's stdin.
    # Use --rawfile variable_name /dev/stdin to read the *entire* raw stdin content
    # into a jq string variable WITHOUT parsing it as JSON.
    # Use --arg name value for smaller arguments like the instructions.
    payload_json=$(echo "$CODE_CONTENT" | jq -n \
        --rawfile code_from_stdin /dev/stdin \
        --arg prompt_instructions "$prompt_instructions" \
        --argjson temp "$TEMPERATURE" \
        --argjson top_p "$TOP_P" \
        --argjson top_k "$TOP_K" \
        --argjson max_tokens "$MAX_OUTPUT_TOKENS" \
        "{
          \"contents\": [{\"parts\": [{\"text\": ${text_jq_expression} }]}],
          \"generationConfig\": {
            \"temperature\": \$temp,
            \"topK\": \$top_k,
            \"topP\": \$top_p,
            \"maxOutputTokens\": \$max_tokens,
            \"stopSequences\": []
          },
          \"safetySettings\": [
            {\"category\": \"HARM_CATEGORY_HARASSMENT\", \"threshold\": \"BLOCK_MEDIUM_AND_ABOVE\"},
            {\"category\": \"HARM_CATEGORY_HATE_SPEECH\", \"threshold\": \"BLOCK_MEDIUM_AND_ABOVE\"},
            {\"category\": \"HARM_CATEGORY_SEXUALLY_EXPLICIT\", \"threshold\": \"BLOCK_MEDIUM_AND_ABOVE\"},
            {\"category\": \"HARM_CATEGORY_DANGEROUS_CONTENT\", \"threshold\": \"BLOCK_MEDIUM_AND_ABOVE\"}
          ]
        }")

    # Check jq exit status
    if [[ $? -ne 0 ]]; then
        # Provide more context in the error message
        echo "DEBUG: CODE_CONTENT length (first 100 chars): ${CODE_CONTENT:0:100}..." >&2
        error_exit "Failed to build JSON payload using jq. Error occurred likely during JSON construction after reading raw input." 8
    fi

    # Set global variable
    JSON_PAYLOAD="$payload_json"
}


# --- Call Gemini API ---
# Usage: call_gemini <api_key> <payload>
# Reads payload from variable and pipes it to curl's stdin using --data @-
# Returns: Assigns API response to global variable API_RESPONSE
call_gemini() {
    local key="$1"
    local payload="$2"
    local response

    # Print the actual model being used (could be default or from env var)
    echo "Sending code to Gemini for review (Model: ${MODEL_NAME})..."

    # Pipe the JSON payload to curl's standard input using --data @-
    # Use printf "%s" for safer piping than echo.
    response=$(printf "%s" "$payload" | curl --silent --show-error --location \
         --request POST "${API_URL}?key=${key}" \
         --header "Content-Type: application/json" \
         --data @-) # Read data from stdin

    # Check curl exit status more reliably
    local curl_status=$?
    if [[ $curl_status -ne 0 ]]; then
        # Check if response has content even if curl exited non-zero (e.g., HTTP error code)
        if [[ -n "$response" ]]; then
             echo "Warning: curl exited with status $curl_status, but received response:" >&2
             echo "${response:0:500}..." >&2 # Show beginning of response
        fi
        error_exit "curl command failed with exit code $curl_status while reading payload from stdin. Check network, API URL (${API_URL}), API Key, or potential issues with piped data format." 5
    fi

    # Check for API-level errors reported in the JSON response
    if echo "$response" | jq -e '.error' > /dev/null; then
        echo "Error: API returned an error:" >&2
        # Print the formatted error JSON
        echo "$response" | jq '.error' >&2
        error_exit "API call failed (received error object in response)." 6 # Use a specific exit code for API errors
    fi

    # Set global variable
    API_RESPONSE="$response"
}


# --- Extract Enhanced Code from Response ---
# Usage: extract_code <api_response_json>
# Returns: Assigns extracted code to global variable ENHANCED_CODE
extract_code() {
    local response_json="$1"
    local extracted_text

    # Primary expected path for Gemini API v1beta text response
    extracted_text=$(echo "$response_json" | jq -r '.candidates[0].content.parts[0].text // ""')

    # Handle cases where the primary path might fail or be empty
    if [[ -z "$extracted_text" ]]; then
         # Attempt fallback path (less common now but good robustness)
         extracted_text=$(echo "$response_json" | jq -r '.candidates[0].text // ""')
         if [[ -n "$extracted_text" ]]; then
             echo "Warning: Extracted code using fallback path '.candidates[0].text'." >&2
         fi
    fi

    # Check if the response indicates blocked content or other issues
    local finish_reason
    finish_reason=$(echo "$response_json" | jq -r '.candidates[0].finishReason // "UNKNOWN"')
    if [[ "$finish_reason" != "STOP" && "$finish_reason" != "MAX_TOKENS" ]]; then
        echo "Warning: Gemini finishReason was '$finish_reason'. Output might be incomplete or blocked due to safety settings or other reasons." >&2
        # If the primary extraction failed AND we have a non-STOP reason, it's likely blocked.
        if [[ -z "$extracted_text" ]]; then
            echo "Error: Code extraction failed, likely due to content blocking (finishReason: $finish_reason)." >&2
            echo "Full API Response (first 500 chars):" >&2
            echo "${response_json:0:500}..." >&2
             # Check for prompt feedback block reason
             local block_reason
             block_reason=$(echo "$response_json" | jq -r '.promptFeedback.blockReason // "NONE"')
             if [[ "$block_reason" != "NONE" ]]; then
                 echo "Prompt Feedback Block Reason: $block_reason" >&2
             fi
            error_exit "Failed to get valid code from API response, potentially blocked." 7
        fi
    fi

    if [[ -z "$extracted_text" && "$finish_reason" == "STOP" ]]; then
        # If extraction is empty but finish reason is STOP, maybe the model returned nothing?
        echo "Warning: Gemini returned an empty response text but finished normally (finishReason: STOP)." >&2
        # Consider if empty response is an error for your use case
        # error_exit "Received empty response text from Gemini." 7
    elif [[ -z "$extracted_text" && "$finish_reason" != "UNKNOWN" ]]; then
        # General catch-all if extraction failed for other known non-STOP reasons
        echo "Error: Could not extract enhanced code from the API response (finishReason: $finish_reason)." >&2
        echo "Full API Response (first 500 chars):" >&2
        echo "${response_json:0:500}..." >&2
        error_exit "Failed to parse valid code from API response." 7
    elif [[ -z "$extracted_text" && "$finish_reason" == "UNKNOWN" ]]; then
         # If reason is unknown and text is empty, parsing likely failed or response was empty/malformed
        echo "Error: Could not extract enhanced code. API response might be malformed or empty." >&2
        echo "Full API Response (first 500 chars):" >&2
        echo "${response_json:0:500}..." >&2
        error_exit "Failed to parse valid code from API response (Unknown reason)." 7
    fi


    # Set global variable
    ENHANCED_CODE="$extracted_text"
}

# === Main Script Logic ===

# --- Argument Parsing ---
INPUT_FILE_ARG="$1"
OUTPUT_FILE_ARG="$2" # Optional second argument for output file

# --- Preparations ---
check_dependencies
get_api_key # Sets GEMINI_API_KEY_INTERNAL

# --- Get Input ---
# Global variables INPUT_FILENAME and CODE_CONTENT will be set by this function
get_filename_and_content "$INPUT_FILE_ARG"

# --- Prepare Prompt Instructions (Code content will be added inside build_payload) ---
PROMPT_INSTRUCTIONS="Please review the following code snippet from the file named '$INPUT_FILENAME'. Your task is to:
1. Identify potential bugs and logical errors.
2. Suggest enhancements for clarity, readability, and maintainability.
3. Optimize for performance and efficiency where applicable.
4. Ensure adherence to common best practices for the language (if discernible).
5. Apply necessary fixes and improvements directly to the code.

IMPORTANT: Return ONLY the complete, corrected, and enhanced code block. Do NOT include any introductory phrases (e.g., \"Here's the enhanced code:\"), concluding remarks, explanations outside the code (unless as comments within the code), summaries, or markdown code fences (\`\`\`) unless the code itself is markdown. Provide only the raw, modified code content."
# Note: The "Original Code:" header and backticks are now added *inside* the build_payload function.

# --- Build Payload ---
# The CODE_CONTENT variable (global) will be piped into jq within the function
build_payload "$PROMPT_INSTRUCTIONS" # Sets JSON_PAYLOAD

# --- Call API ---
call_gemini "$GEMINI_API_KEY_INTERNAL" "$JSON_PAYLOAD" # Sets API_RESPONSE

# --- Extract Code ---
extract_code "$API_RESPONSE" # Sets ENHANCED_CODE

# --- Output Result ---
echo ""
echo "----------------------------------------"
echo " Gemini Code Review Result:           "
echo "----------------------------------------"

if [[ -n "$OUTPUT_FILE_ARG" ]]; then
    # Attempt to save to file
    if printf "%s" "$ENHANCED_CODE" > "$OUTPUT_FILE_ARG"; then
        echo "Enhanced code saved successfully to: $OUTPUT_FILE_ARG"
    else
        local write_status=$?
        echo "Error: Failed to save enhanced code to '$OUTPUT_FILE_ARG' (Exit code: $write_status)." >&2
        echo "Displaying code on terminal instead:"
        echo "----------------------------------------"
        printf "%s\n" "$ENHANCED_CODE" # Use printf for safer output
        echo "----------------------------------------"
        exit 9 # Specific exit code for file write error
    fi
else
    # Print to terminal if no output file specified
    printf "%s\n" "$ENHANCED_CODE" # Use printf for potentially safer output than echo
    echo "----------------------------------------"
    echo "(To save output to a file, run as: $0 <input_file> <output_file>)"
fi

echo ""
echo "Review complete for '$INPUT_FILENAME'."
exit 0