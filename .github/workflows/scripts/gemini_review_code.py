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

