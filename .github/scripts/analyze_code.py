# .github/scripts/analyze_code.py
import os
import json
import requests
import google.generativeai as genai
from colorama import init, Fore, Style, Back

init(autoreset=True)

MAX_CHUNK_TOKENS = 3800
GEMINI_MODEL_NAME = '${GEMINI_MODEL}'
GITHUB_API_BASE_URL = "https://api.github.com"

def print_wizard_message(message, color=Fore.CYAN):
    print(color + Style.BRIGHT + "✨ Pyrmethus whispers: " + Style.RESET_ALL + color + message + Style.RESET_ALL)

def print_error_message(message):
    print(Back.RED + Fore.WHITE + Style.BRIGHT + "⚠️ Arcane Anomaly: " + Style.RESET_ALL + Fore.RED + message + Style.RESET_ALL)

def chunk_diff(diff_text, max_tokens=MAX_CHUNK_TOKENS):
    print_wizard_message(f"Chunking the diff (max ~{max_tokens} tokens per chunk)...", Fore.MAGENTA)
    max_chars = max_tokens * 4
    lines = diff_text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_char_count = 0
    current_file = None

    for line in lines:
        if line.startswith('+++ b/'):
            current_file = line[6:]
        line_len = len(line) + 1
        if current_char_count + line_len > max_chars and current_chunk_lines:
            chunks.append(("\n".join(current_chunk_lines), current_file))
            current_chunk_lines = []
            current_char_count = 0
            if line.startswith('+++ b/'):
                current_chunk_lines.append(line)
                current_char_count += line_len
            elif not line.startswith('--- a/'):
                current_chunk_lines.append(line)
                current_char_count += line_len
        elif not (line.startswith('--- a/') and not current_chunk_lines):
            current_chunk_lines.append(line)
            current_char_count += line_len

    if current_chunk_lines:
        if not current_file:
            for line in reversed(current_chunk_lines):
                if line.startswith('+++ b/'):
                    current_file = line[6:]
                    break
        chunks.append(("\n".join(current_chunk_lines), current_file if current_file else "unknown_file"))

    print_wizard_message(f"Diff split into {Fore.YELLOW}{len(chunks)}{Fore.MAGENTA} chunks.", Fore.MAGENTA)
    return chunks

def analyze_code_chunk_with_gemini(diff_chunk, model):
    prompt = f"""
    You are an expert code reviewer AI assistant, Pyrmethus's familiar.
    Analyze the following code diff strictly for potential bugs, security vulnerabilities, and significant logical errors within the changes presented.
    Ignore stylistic suggestions unless they indicate a potential bug (e.g., variable shadowing).
    Focus ONLY on the added or modified lines (usually starting with '+').

    For each distinct issue found, provide:
    1. `file_path`: The full path of the file where the issue occurs (e.g., "src/utils/helpers.py"). Infer this from the '+++ b/...' line in the diff chunk.
    2. `line_number`: The approximate line number *in the new file version* where the issue begins. Estimate based on the '@@ ... +start,count @@' hunk header and '+' lines.
    3. `description`: A concise explanation of the potential bug or vulnerability.
    4. `affected_code`: The specific line(s) from the diff (lines starting with '+') that contain the issue.
    5. `suggested_fix`: A concrete code snippet demonstrating how to fix the issue. If the fix involves removing lines, indicate that clearly.

    If no significant issues are found in this chunk, respond with an empty JSON array: `[]`.

    Respond ONLY with a valid JSON array containing objects matching the structure described above.

    Diff Chunk:
    ```diff
    {diff_chunk}
    ```
    """
    print_wizard_message(f"Sending chunk to Gemini ({GEMINI_MODEL_NAME})...", Fore.BLUE)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        if not response_text:
            print_wizard_message("Gemini found no issues in this chunk.", Fore.GREEN)
            return []
        issues = json.loads(response_text)
        print_wizard_message(f"Gemini identified {Fore.YELLOW}{len(issues)}{Fore.GREEN} potential issue(s) in this chunk.", Fore.GREEN)
        return issues
    except json.JSONDecodeError as json_err:
        print_error_message(f"Failed to decode Gemini's JSON response.")
        print(Fore.RED + f"Raw Response Text: {response.text[:500]}...")
        print(Fore.RED + f"Error details: {json_err}")
        return []
    except Exception as e:
        print_error_message(f"An error occurred while querying Gemini: {e}")
        return []

def post_github_comment(repo_name, pr_number, commit_sha, github_token, issue_report):
    api_url = f"{GITHUB_API_BASE_URL}/repos/{repo_name}/pulls/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    file_path = issue_report.get('file_path', 'unknown_file')
    line_number = issue_report.get('line_number', 1)
    description = issue_report.get('description', 'N/A')
    affected_code = issue_report.get('affected_code', 'N/A')
    suggested_fix = issue_report.get('suggested_fix', 'N/A')
    comment_body = f"""
:mage: **Pyrmethus's Insight** :crystal_ball:

**Issue Detected:** {description}
**File:** `{file_path}` (approx. line: {line_number})

**Affected Code Snippet:**
```diff
{affected_code}
```

**Suggested Enchantment (Fix):**
```python
{suggested_fix}
```
*(Note: Line number is an estimate. Please verify the context.)*
"""
    payload = {
        "body": comment_body,
        "commit_id": commit_sha,
        "path": file_path,
        "line": int(line_number),
        "side": "RIGHT"
    }
    print_wizard_message(f"Posting comment for issue in {Fore.YELLOW}{file_path}:{line_number}{Fore.CYAN}...", Fore.CYAN)
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        print_wizard_message(f"Comment posted successfully!", Fore.GREEN)
    except requests.exceptions.RequestException as e:
        print_error_message(f"Failed to post GitHub comment for {file_path}:{line_number}.")
        if e.response is not None:
            print(Fore.RED + f"Status Code: {e.response.status_code}")
            print(Fore.RED + f"Response Body: {e.response.text}")

def main():
    print_wizard_message("Starting the Gemini Code Review Spell...", Fore.MAGENTA + Style.BRIGHT)
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    github_token = os.getenv('GITHUB_TOKEN')
    repo_name = os.getenv('REPO_NAME')
    pr_number = os.getenv('PR_NUMBER')
    pr_diff = os.getenv('PR_DIFF')
    commit_sha = os.getenv('COMMIT_SHA')
    if not all([gemini_api_key, github_token, repo_name, pr_number, pr_diff, commit_sha]):
        print_error_message("One or more required environment variables are missing!")
        exit(1)
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print_wizard_message("Gemini Oracle configured successfully.", Fore.GREEN)
    except Exception as e:
        print_error_message(f"Failed to configure Gemini API: {e}")
        exit(1)
    diff_chunks = chunk_diff(pr_diff)
    total_issues_found = 0
    for chunk, file_context in diff_chunks:
        if not chunk.strip():
            continue
        print_wizard_message(f"Analyzing chunk related to file: {Fore.YELLOW}{file_context or 'Multiple/Unknown'}", Fore.BLUE)
        issues = analyze_code_chunk_with_gemini(chunk, model)
        for issue in issues:
            if 'file_path' not in issue or not issue['file_path']:
                issue['file_path'] = file_context if file_context else 'unknown_file'
            post_github_comment(repo_name, pr_number, commit_sha, github_token, issue)
            total_issues_found += 1
    print_wizard_message(f"Code review spell complete. Found {Fore.YELLOW}{total_issues_found}{Fore.MAGENTA} potential issues.", Fore.MAGENTA + Style.BRIGHT)

if __name__ == "__main__":
    main()
