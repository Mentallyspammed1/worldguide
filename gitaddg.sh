#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Forge GitHub Actions Structure Script (v2 Enhanced)
#
# Purpose: Sets up the directory structure and necessary files for a GitHub
#          Action that performs code reviews using Google's Gemini API.
#          Designed with Termux environment considerations.
# Features:
#   - Idempotent directory creation.
#   - Interactive file writing (overwrite/skip/abort prompt).
#   - Explicit error handling (`set -e`, `set -o pipefail`).
#   - Modular function-based design.
#   - Colorized output for better readability.
#   - Includes GitHub Workflow YAML and Python review script.
# ==============================================================================

# === Arcane Configuration ===
# Halt script immediately if any command fails
set -e
# Halt script if any command in a pipeline fails (e.g., cmd1 | cmd2)
set -o pipefail
# Treat unset variables as an error (optional, but good practice)
# set -u # Uncomment if you want stricter variable checks

# === Colors for the Mystical Script Output ===
# (Using ANSI escape codes for terminal colors)
RESET='\e[0m'
BOLD='\e[1m'
DIM='\e[2m'
RED='\e[0;31m'
GREEN='\e[0;32m'
YELLOW='\e[0;33m'
BLUE='\e[0;34m'
MAGENTA='\e[0;35m'
CYAN='\e[0;36m'
# Bright variants
BRIGHT_RED='\e[1;31m'
BRIGHT_GREEN='\e[1;32m'
BRIGHT_YELLOW='\e[1;33m'
BRIGHT_BLUE='\e[1;34m'
BRIGHT_MAGENTA='\e[1;35m'
BRIGHT_CYAN='\e[1;36m'

# === Utility Functions ===

# Function to create a directory if it doesn't exist
# Provides user feedback.
# Usage: create_directory <path>
create_directory() {
    local dir_path="$1"
    if [ -d "$dir_path" ]; then
        # Use printf for potentially better portability/consistency than echo -e
        printf "${DIM}${YELLOW}* Directory already exists:${RESET}${DIM} %s${RESET}\n" "$dir_path"
    else
        printf "${YELLOW}* Summoning directory dimension:${RESET} ${BRIGHT_YELLOW}%s${RESET}\n" "$dir_path"
        # -p creates parent directories as needed, no error if it already exists
        mkdir -p "$dir_path"
        printf "${GREEN}  Dimension materialized.${RESET}\n"
    fi
     # No explicit return needed here, rely on set -e for errors
}

# Function to safely write content to a file, checking for existence.
# Prompts the user for action if the file exists (overwrite, skip, abort).
# Reads content from standard input (designed for here-documents).
# Usage: write_file_safely <file_path> << 'HEREDOC_MARKER'
#        ... content ...
#        HEREDOC_MARKER
write_file_safely() {
    local file_path="$1"
    local overwrite_choice="" # Store user's choice

    printf "${CYAN}- Preparing to inscribe:${RESET} ${BRIGHT_CYAN}%s${RESET}\n" "$file_path"

    if [ -f "$file_path" ]; then
        # Prompt user, reading directly from the terminal
        printf "${BRIGHT_YELLOW}  ? File already exists. Overwrite? (y/N/s[kip]/a[bort]): ${RESET}"
        # Read a single character directly from the TTY, even if stdin is redirected
        read -r -n 1 overwrite_choice </dev/tty
        printf "\n" # Add a newline for cleaner output after user input

        case "$overwrite_choice" in
            y|Y)
                printf "${YELLOW}  Proceeding with overwrite.${RESET}\n"
                ;; # Continue to writing block
            s|S)
                printf "${DIM}  Skipping inscription as requested.${RESET}\n"
                return 0 # Success (skipped), continue script
                ;;
            a|A)
                printf "${BRIGHT_RED}  Aborting ritual as requested.${RESET}\n"
                exit 1 # Abort the entire script
                ;;
            *)
                # Default case (including Enter key or anything else) is No/Skip
                printf "${DIM}  Assuming 'No'. Keeping existing file.${RESET}\n"
                return 0 # Success (skipped), continue script
                ;;
        esac
    # else # File doesn't exist, implicit 'yes' to write
    fi

    # If we reached here, either the file didn't exist or user chose 'y'
    printf "${YELLOW}  Inscribing arcane text...${RESET}\n"
    # Use 'cat' to read from stdin (here-document) and redirect to the file
    # The redirection '>' truncates the file before writing.
    if cat > "$file_path"; then
        printf "${GREEN}  Inscription successful.${RESET}\n"
    else
        # This branch should ideally not be reached if cat fails due to `set -e`,
        # but explicit check adds clarity.
        printf "${BRIGHT_RED}  Error: Failed to inscribe %s!${RESET}\n" "$file_path"
        exit 1 # Critical error, halt script
    fi

    return 0 # Indicate success for this file operation
}

# === Main Ritual ===

printf "${BRIGHT_MAGENTA}${BOLD}~~~ Pyrmethus's GitHub Actions Structure Conjuration v2 (Enhanced) ~~~${RESET}\n"
# Simple check confirming script context awareness
if [[ -n "$TERMUX_VERSION" ]]; then
     printf "${DIM}Performing within the Termux environment (v%s).${RESET}\n" "$TERMUX_VERSION"
else
     printf "${DIM}Performing setup (Termux environment not explicitly detected).${RESET}\n"
fi
printf "\n"

# --- Step 1: Sculpting the Directories ---
printf "${BRIGHT_BLUE}--- Phase 1: Sculpting Dimensions ---${RESET}\n"
create_directory ".github/workflows/scripts" # Workflow scripts sub-directory
create_directory ".gemini"                  # For Gemini-related config like style guides
printf "\n"

# --- Step 2: Inscribing the Scrolls ---
printf "${BRIGHT_BLUE}--- Phase 2: Inscribing Arcane Scrolls ---${RESET}\n"

# Scroll 1: GitHub Workflow Definition (.github/workflows/gemini-code-review.yml)
# Defines the trigger, permissions, jobs, and steps for the code review action.
write_file_safely ".github/workflows/gemini-code-review.yml" << 'EOF_WORKFLOW'
# Workflow name
name: Code Review with Gemini

# Triggers: Run on pull requests targeting specific branches
on:
  pull_request:
    types: [opened, synchronize, reopened] # Events that indicate new code or reopening
    branches:
      - main      # Target branch: main
      - develop   # Target branch: develop
      # Add any other primary branches you want reviews on

# Permissions: Define the minimal permissions needed by the workflow
permissions:
  contents: read          # Required by actions/checkout to read repository code
  pull-requests: write  # Required to post review comments on the PR

jobs:
  code_review:
    name: Gemini Code Review
    runs-on: ubuntu-latest # Use the latest stable Ubuntu runner

    # Security: Prevent running on forks if secrets aren't explicitly shared/intended.
    # This ensures your GEMINI_API_KEY is not exposed or used unintentionally by forks.
    if: github.event.pull_request.head.repo.full_name == github.repository

    steps:
      # Step 1: Check out the code from the PR branch
      - name: Checkout PR Code
        uses: actions/checkout@v4 # Use specific major version tag
        with:
          # Checkout the actual commit SHA of the pull request's head branch
          ref: ${{ github.event.pull_request.head.sha }}
          # Fetch all history. Required for accurate diff generation,
          # especially if comparing against the base branch.
          fetch-depth: 0

      # Step 2: Set up the Python environment
      - name: Set up Python
        uses: actions/setup-python@v5 # Use specific major version tag
        with:
          python-version: '3.10' # Specify a consistent Python version

      # Step 3: Install Python dependencies
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          # Install required libraries for GitHub API interaction and Gemini API
          pip install PyGithub google-generativeai

      # Step 4: Execute the Python script for code review
      - name: Run Gemini Code Review Script
        # Environment variables passed to the Python script
        env:
          # GitHub token (automatically provided by Actions runner) for API auth
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # Gemini API Key (must be configured as a Repository Secret)
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          # Repository name (e.g., 'owner/repo')
          GITHUB_REPOSITORY: ${{ github.repository }}
          # Pull Request number associated with the event
          GITHUB_EVENT_NUMBER: ${{ github.event.number }}
          # Workspace path (where the code is checked out) for finding local files like style guides
          GITHUB_WORKSPACE: ${{ github.workspace }}
        run: |
          # Execute the Python script located in the specified path
          python .github/workflows/scripts/gemini_review_code.py

EOF_WORKFLOW

# Scroll 2: Python Familiar Script (.github/workflows/scripts/gemini_review_code.py)
# This script interacts with GitHub API (get diff) and Gemini API (get review),
# then posts the review back to the PR.
write_file_safely ".github/workflows/scripts/gemini_review_code.py" << 'EOF_PYTHON'
import os
import github # PyGithub library
import google.generativeai as genai # Google AI SDK
import re
import sys
import time # For delays (e.g., retries)

# === Configuration ===
# Fetch sensitive keys and context from environment variables (set in workflow)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REPO_NAME = os.getenv('GITHUB_REPOSITORY') # Format: 'owner/repo'
PR_NUMBER_STR = os.getenv('GITHUB_EVENT_NUMBER') # The PR number triggering the workflow
GITHUB_WORKSPACE = os.getenv('GITHUB_WORKSPACE', '.') # Default to current dir if not in Actions

# === Constants ===
MAX_COMMENT_LENGTH = 65000 # Approximate GitHub comment length limit (65536 bytes)
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest' # Choose model: 'gemini-1.5-flash' (faster, cheaper) or 'gemini-1.5-pro' (more powerful)
MAX_RETRIES = 2 # Number of retries for Gemini API calls
RETRY_DELAY_SECONDS = 5 # Seconds to wait between retries
STYLE_GUIDE_PATH = '.gemini/styleguide.md' # Relative path within the repo

# === Validate Environment Variables ===
def validate_environment():
    """Check if all required environment variables are set and valid."""
    print("Validating environment variables...")
    required_vars = {
        'GITHUB_TOKEN': GITHUB_TOKEN,
        'GEMINI_API_KEY': GEMINI_API_KEY,
        'GITHUB_REPOSITORY': REPO_NAME,
        'GITHUB_EVENT_NUMBER': PR_NUMBER_STR
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1) # Exit with error code

    try:
        pr_number = int(PR_NUMBER_STR)
        print(f"✅ Environment variables validated. Using PR #{pr_number}.")
        return pr_number
    except (ValueError, TypeError):
        print(f"❌ Error: Invalid GITHUB_EVENT_NUMBER: '{PR_NUMBER_STR}'. Must be an integer.")
        sys.exit(1) # Exit with error code

# === Initialize APIs ===
def initialize_apis(pr_number):
    """Initialize GitHub and Gemini API clients."""
    print("\n--- Initializing APIs ---")
    # Initialize GitHub Client
    try:
        print(f"Initializing GitHub client for repo: {REPO_NAME}")
        g = github.Github(GITHUB_TOKEN)
        repo = g.get_repo(REPO_NAME)
        print(f"Fetching Pull Request #{pr_number}...")
        pr = repo.get_pull(pr_number)
        print(f"✅ GitHub API initialized and PR #{pr_number} obtained.")
    except Exception as e:
        print(f"❌ Error initializing GitHub client or getting PR #{pr_number}: {e}")
        sys.exit(1) # Exit with error code

    # Initialize Gemini Client
    try:
        print(f"Configuring Gemini API with model: {GEMINI_MODEL_NAME}")
        genai.configure(api_key=GEMINI_API_KEY)
        # Add client options if needed, e.g., for request timeouts
        # client_options = {"api_endpoint": "generativelanguage.googleapis.com"}
        # model = genai.GenerativeModel(GEMINI_MODEL_NAME, client_options=client_options)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print("✅ Gemini API configured.")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        sys.exit(1) # Exit with error code

    return g, repo, pr, model

# === Core Functions ===
def get_pr_diff(pr):
    """Fetch the diff of the pull request from the GitHub API."""
    print("\n--- Fetching PR Diff ---")
    try:
        print(f"Requesting file diffs for PR #{pr.number}...")
        files = pr.get_files()
        diff_content = ""
        file_count = 0
        change_count = 0
        total_additions = 0
        total_deletions = 0

        for file in files:
            file_count += 1
            total_additions += file.additions
            total_deletions += file.deletions
            # We only care about the patch (diff) content for the review
            if file.patch:
                change_count += 1
                diff_content += f"--- File: {file.filename} ---\n"
                diff_content += f"Status: {file.status} (+{file.additions}/-{file.deletions})\n"
                # Add patch content - consider size limits if Gemini struggles
                # Note: Gemini models (esp. 1.5 Pro) have large context windows,
                # so truncating per file might not be necessary initially.
                # max_patch_len = 10000 # Example limit per file
                # patch_content = file.patch[:max_patch_len] + ('\n... (patch truncated)' if len(file.patch) > max_patch_len else '')
                patch_content = file.patch
                diff_content += f"Patch:\n{patch_content}\n\n"

        print(f"Found {change_count} files with changes (+{total_additions}/-{total_deletions} lines) "
              f"out of {file_count} files in the PR.")

        # Optional: Check total diff size against Gemini token limits if needed
        # max_total_diff = 300000 # Example total limit (adjust based on model)
        # if len(diff_content) > max_total_diff:
        #     print(f"⚠️ Warning: Total diff size ({len(diff_content)} chars) is large. Truncating for review.")
        #     diff_content = diff_content[:max_total_diff] + "\n\n... (Total diff truncated due to size limit)"

        return diff_content.strip() # Return stripped diff content
    except Exception as e:
        print(f"❌ Error fetching PR diff for PR #{pr.number}: {e}")
        # Attempt to post an error comment back to the PR
        try:
             pr.create_issue_comment(f"⚠️ **Gemini Review Bot Error:**\n\nCould not retrieve the diff for this pull request.\n\n`{e}`")
        except Exception as post_e:
             print(f"❌ Additionally, failed to post diff retrieval error to PR: {post_e}")
        return None # Indicate failure

def load_style_guide():
    """Loads the custom style guide content from the predefined path."""
    style_guide_full_path = os.path.join(GITHUB_WORKSPACE, STYLE_GUIDE_PATH)
    print(f"Attempting to load style guide from: {style_guide_full_path}")
    if os.path.exists(style_guide_full_path):
        try:
            with open(style_guide_full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print("✅ Loaded custom style guide.")
            return content
        except Exception as e:
            print(f"⚠️ Warning: Could not read style guide at {style_guide_full_path}: {e}")
            return "Error reading style guide file."
    else:
        print("ℹ️ No custom style guide found at specified path. Using default prompt instructions.")
        return "No custom style guide provided."

def generate_review_prompt(diff, pr_number, repo_name):
    """Generate a prompt for Gemini to review the code diff."""
    print("Generating review prompt for Gemini...")
    style_guide_content = load_style_guide()

    # Construct the prompt with clear instructions
    prompt = f"""Act as an expert code reviewer for a GitHub pull request.

**Context:**
- Repository: {repo_name}
- Pull Request #: {pr_number}

**Your Task:**
Analyze the following code diff thoroughly. Provide concise, actionable feedback focusing ONLY on potential issues **within the changed lines ('+' or '-') or their immediate surrounding context** in the provided patch. Do not review unchanged lines outside the diff context.

**Review Focus Areas:**
- **Bugs & Logic Errors:** Identify potential runtime errors, logical flaws, off-by-one errors, incorrect assumptions, race conditions, or missed edge cases introduced by the changes.
- **Security Vulnerabilities:** Look for common vulnerabilities (e.g., injection, XSS, insecure data handling, hardcoded secrets, auth issues) introduced or exacerbated by the diff.
- **Performance Issues:** Point out obvious performance bottlenecks, inefficient loops/algorithms, or excessive resource usage introduced in the changed code.
- **Maintainability & Readability:** Comment on overly complex code, poor naming, lack of necessary comments for complex logic, or violations of common style conventions (refer to the style guide below) within the changed sections.
- **Best Practices & Idioms:** Note deviations from language/framework best practices or idiomatic code relevant to the changes.
- **Redundancy:** Identify clearly redundant code introduced by the changes.
- **Testing:** Suggest missing test cases for the changed logic, if obvious.

**Output Format:**
- Structure feedback clearly. Use bullet points for multiple distinct issues.
- **MUST** reference the specific file (`File: <filename>`) for each comment.
- If possible, reference specific line numbers *within the patch context* (e.g., `Lines +10-15 in patch`). Use the `+` or `-` prefix from the diff format if relevant.
- Prioritize critical and high-impact issues.
- Be constructive. Explain *why* something is an issue and suggest improvements if applicable.
- If no significant issues are found in the diff, explicitly state: "✅ No significant issues found in the reviewed changes."

**Custom Style Guide (Consider this when evaluating Maintainability/Readability/Best Practices):**
```markdown
{style_guide_content}
```

**Code Diff to Review:**
```diff
{diff}
```

**Final Instruction:** Provide only the review feedback based **solely on the provided diff**. Do not include greetings, summaries outside the feedback, or comments on code not present in the diff. Focus on the changes.
"""
    # print(f"\n--- Generated Prompt Snippet ---\n{prompt[:500]}...\n--- End Prompt Snippet ---\n") # Debugging: print prompt start
    return prompt

def call_gemini_api_with_retry(model, prompt):
    """Sends the prompt to the Gemini API with retry logic for transient errors."""
    print("\n--- Calling Gemini API ---")
    retries = 0
    last_error = None
    while retries <= MAX_RETRIES:
        try:
            print(f"Attempting API call (Attempt {retries + 1}/{MAX_RETRIES + 1})...")
            start_api_call = time.time()
            response = model.generate_content(
                prompt,
                # Configuration for the generation process
                generation_config=genai.types.GenerationConfig(
                    # Controls randomness. Lower values (e.g., 0.2) make output more deterministic/focused.
                    temperature=0.2,
                    # Consider setting max_output_tokens if needed, based on model limits
                    # max_output_tokens=8192,
                ),
                # Optional: Define safety settings to block harmful content
                # safety_settings=[...] # Defaults are usually reasonable
            )
            end_api_call = time.time()
            print(f"Gemini API call duration: {end_api_call - start_api_call:.2f} seconds.")

            # --- Process the response ---
            if response.candidates:
                if response.candidates[0].content.parts:
                    review_text = response.candidates[0].content.parts[0].text
                    print("✅ Successfully received review from Gemini.")
                    return review_text # Success
                else:
                    # Candidate exists but has no content parts
                    last_error = "Gemini API returned a candidate with empty content."
                    print(f"⚠️ Warning: {last_error}")
                    # Allow retry for potentially transient empty responses
            else:
                # No candidates usually means content was blocked or there was an issue
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                last_error = f"Gemini API returned no candidates. Block Reason: '{block_reason}'. This might be due to safety filters or prompt issues."
                print(f"⚠️ Warning: {last_error}")
                # Do not retry if blocked, as it's likely a content issue
                if block_reason != "BLOCK_REASON_UNSPECIFIED" and block_reason != "Unknown":
                    print("Stopping retries due to content blocking.")
                    return f"Error: Gemini API blocked the request or response. Reason: {block_reason}"
                # Otherwise, allow retry for unspecified/unknown reasons

        except Exception as e:
            last_error = f"Error calling Gemini API: {e}"
            print(f"❌ Error during API call attempt {retries + 1}: {e}")
            # Consider specific error types that might warrant stopping retries early
            # e.g., authentication errors (401/403), invalid argument errors (400)

        # If we reached here, it means an error occurred or the response was problematic
        retries += 1
        if retries <= MAX_RETRIES:
            print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            print("❌ Max retries reached for Gemini API call.")
            final_error_message = f"Error: Failed to get review from Gemini API after {MAX_RETRIES + 1} attempts."
            if last_error:
                final_error_message += f"\nLast known issue: {last_error}"
            return final_error_message

    # Should not be reachable if logic is correct
    return "Error: Unexpected state reached in Gemini API call function."


def post_review_comment(pr, review_text):
    """Posts the generated review as a single comment on the GitHub PR."""
    print("\n--- Posting Review Comment ---")
    try:
        print(f"Formatting review comment for PR #{pr.number}...")
        # Add a header to the comment
        comment_header = f"### ✨ Gemini Code Review ({GEMINI_MODEL_NAME}) ✨\n\n"
        comment_body = comment_header + review_text.strip()

        # Check for and handle GitHub's comment length limit
        if len(comment_body.encode('utf-8')) > MAX_COMMENT_LENGTH: # Check byte length
            print(f"⚠️ Warning: Review text exceeds GitHub comment length limit ({MAX_COMMENT_LENGTH} bytes). Truncating.")
            # Truncate carefully to avoid breaking markdown or code blocks if possible
            # This is a simple truncation; more sophisticated methods could be used.
            truncation_marker = "\n\n...(review truncated due to length limit)"
            # Calculate allowed length for the main body
            allowed_body_length = MAX_COMMENT_LENGTH - len(comment_header.encode('utf-8')) - len(truncation_marker.encode('utf-8'))
            # Truncate the review_text part based on byte length
            truncated_review_bytes = review_text.encode('utf-8')[:allowed_body_length]
            # Decode back to string, ignoring errors in case of split multi-byte char
            truncated_review_text = truncated_review_bytes.decode('utf-8', errors='ignore')
            comment_body = comment_header + truncated_review_text + truncation_marker
            print(f"Comment truncated to {len(comment_body.encode('utf-8'))} bytes.")

        print(f"Posting comment to Pull Request #{pr.number}...")
        pr.create_issue_comment(comment_body)
        print(f"✅ Successfully posted review comment to PR #{pr.number}.")
        return True
    except Exception as e:
        print(f"❌ Error posting comment to PR #{pr.number}: {e}")
        # Fallback: Print review to logs if posting fails, so it's not lost
        print("\n--- Gemini Review Text (Failed to Post to PR) ---")
        print(review_text)
        print("--- End Review Text ---\n")
        # Decide if this should be a fatal error for the Action
        # Consider returning False instead of exiting if you want the workflow step to succeed even if posting fails.
        # sys.exit(1) # Uncomment to fail the workflow step if posting the comment fails
        return False

# === Main Execution ===
def main():
    """Main function to orchestrate the code review process."""
    overall_start_time = time.time()
    print("=============================================")
    print("--- Starting Gemini Code Review Process ---")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=============================================")

    pr_number = validate_environment()
    g, repo, pr, model = initialize_apis(pr_number)

    diff = get_pr_diff(pr)

    # Handle case where diff retrieval failed
    if diff is None:
        print("❌ Failed to retrieve PR diff. Aborting review process.")
        # Error message was already posted in get_pr_diff if possible
        sys.exit(1) # Exit with error status

    # Handle case where diff is empty (e.g., PR with no code changes)
    if not diff:
        print("\nℹ️ No code changes detected in the pull request diff.")
        post_review_comment(pr, "ℹ️ **Gemini Review Bot:** No code changes detected to review in this Pull Request.")
        print("--- Review Process Completed (No Changes) ---")
        overall_end_time = time.time()
        print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")
        sys.exit(0) # Exit successfully

    # Generate the prompt using the retrieved diff
    prompt = generate_review_prompt(diff, pr_number, REPO_NAME)

    # Call the Gemini API to get the review
    review_text = call_gemini_api_with_retry(model, prompt)

    # Process the result from the Gemini API
    if not review_text or review_text.startswith("Error:"):
        error_message = f"⚠️ **Gemini Review Bot Error:**\n\nFailed to generate a valid code review.\n\nDetails:\n```\n{review_text}\n```"
        print(f"\n❌ {error_message}")
        post_review_comment(pr, error_message) # Post the error details to the PR
        # Decide if API failure should fail the workflow step
        # sys.exit(1) # Uncomment to fail the step on API error/failure
    else:
        # Post the successful review comment
        post_review_comment(pr, review_text)

    overall_end_time = time.time()
    print("\n=============================================")
    print(f"--- Gemini Code Review Process Completed ---")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")
    print("=============================================")

if __name__ == "__main__":
    main()

EOF_PYTHON

# Scroll 3: Style Guide Template (.gemini/styleguide.md)
# A basic template for guiding Gemini's review focus. Customize as needed.
write_file_safely ".gemini/styleguide.md" << 'EOF_STYLEGUIDE'
# Custom Style Guide for Gemini Code Review

This guide helps Gemini focus its review comments towards our project's standards.

## General Principles
- **Clarity over Cleverness:** Code should be straightforward and easy for others (and future you) to understand. Avoid overly complex or obscure constructs if simpler alternatives exist.
- **Consistency:** Adhere to existing project patterns, naming conventions, and architectural choices. New code should feel like it belongs.
- **DRY (Don't Repeat Yourself):** Avoid copy-pasting code blocks. Use functions, classes, or other abstractions to promote reuse.
- **Focused Changes:** Pull requests should ideally represent a single logical change or feature.
- **Tested Code:** New logic or bug fixes should be accompanied by relevant unit or integration tests. (Gemini primarily reviews the diff, but can sometimes spot missing test coverage).

## Python Specifics (If applicable)
- **PEP 8:** Follow standard Python style guidelines (e.g., via linters like Flake8, Black, Ruff). Pay attention to line length, naming conventions (snake_case for functions/variables, PascalCase for classes), imports order, and whitespace.
- **Type Hinting:** Use type hints (Python 3.6+) for function signatures (arguments and return types) to improve code clarity and enable static analysis.
- **Docstrings:** Provide clear and concise docstrings for all public modules, classes, functions, and methods (e.g., following Google, NumPy, or reStructuredText style). Explain the *what* and *why*, not just the *how*.
- **Error Handling:** Use specific exception types rather than broad `except Exception:`. Handle potential errors gracefully and provide informative error messages. Log errors appropriately.
- **Resource Management:** Use `with` statements for managing resources like files, network connections, locks, etc., to ensure they are properly released even if errors occur.
- **List Comprehensions/Generator Expressions:** Prefer these over `map()`/`filter()` or simple `for` loops when they improve readability and conciseness for creating lists or iterables.
- **Logging:** Use the `logging` module for application logging instead of `print()` statements for better control over levels and output destinations.

## Security Focus (Critical)
- **Input Validation/Sanitization:** Treat ALL external input (user input, API responses, file content, environment variables) as potentially malicious. Validate formats, ranges, and types. Sanitize data appropriately before using it in queries, commands, or HTML output (prevent XSS, Injection).
- **Secrets Management:** **ABSOLUTELY NO** hardcoded API keys, passwords, certificates, or other sensitive credentials in the codebase. Use GitHub Secrets, environment variables injected securely, or a dedicated secrets management service.
- **Least Privilege:** Ensure code runs with the minimum necessary permissions. File permissions and access controls should be appropriately restrictive.
- **Dependency Security:** Keep dependencies up-to-date to patch known vulnerabilities. Check if the PR introduces new dependencies or updates existing ones, considering their security implications. Use tools like `pip-audit` or GitHub Dependabot alerts.
- **Authentication & Authorization:** Verify that changes correctly implement or respect authentication and authorization checks.

## Performance Considerations
- **Algorithmic Efficiency:** Be mindful of algorithm complexity (e.g., avoid O(n^2) loops if O(n) or O(n log n) is feasible).
- **Database Queries:** Avoid N+1 query problems in ORMs. Fetch data efficiently using appropriate joins or prefetching. Ensure database queries are indexed where necessary.
- **Resource Usage:** Be mindful of memory consumption and CPU usage, especially in loops or data processing tasks. Avoid blocking operations in asynchronous code.
- **Caching:** Consider caching for expensive computations or frequently accessed, rarely changing data where appropriate.

EOF_STYLEGUIDE

printf "\n"
# --- Completion ---
printf "${BRIGHT_GREEN}${BOLD}*** GitHub Actions Structure Conjuration Complete! ***${RESET}\n"
printf "${CYAN}The directories and files have been materialized or updated based on your choices.${RESET}\n"
printf "${CYAN}Review the generated structure (especially the Python script and Workflow file).${RESET}\n"
printf "${CYAN}Then, proceed with your Git rituals (add, commit, push).${RESET}\n"
printf "\n"
printf "${BRIGHT_YELLOW}${BOLD}⚡️ Crucial Reminder: Configure Secrets! ⚡️${RESET}\n"
printf "${CYAN}Ensure the ${BRIGHT_GREEN}GEMINI_API_KEY${CYAN} secret is configured in your GitHub repository settings under:${RESET}\n"
printf "${YELLOW}  Settings > Security > Secrets and variables > Actions > Repository secrets${RESET}\n"
printf "${CYAN}Without this secret, the code review Action will fail.${RESET}\n"

exit 0 # Explicitly exit with success code
