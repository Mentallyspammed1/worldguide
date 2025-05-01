#!/usr/bin/env bash

# Pyrmethus's Enhanced Repository Setup Spell for Gemini Code Review

# Exit on errors
set -e

# --- Configuration ---
LOG_FILE="pyrmethus_setup.log"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-1.5-flash-latest}"
BRANCH_NAME="add-gemini-code-review"
WORKFLOW_FILENAME="${WORKFLOW_FILENAME:-gemini_code_review.yml}"

# --- Color Codes for Enchantment ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_CYAN='\033[0;36m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_RED='\033[0;31m'
C_MAGENTA='\033[0;35m'

# --- Wizardly Functions ---
wizard_echo() {
  echo -e "${C_MAGENTA}${C_BOLD}✨ Pyrmethus whispers:${C_RESET} ${C_CYAN}$1${C_RESET}" | tee -a "$LOG_FILE"
}

success_echo() {
  echo -e "${C_GREEN}${C_BOLD}✅ Success:${C_RESET} ${C_GREEN}$1${C_RESET}" | tee -a "$LOG_FILE"
}

warning_echo() {
  echo -e "${C_YELLOW}${C_BOLD}⚠️ Caution:${C_RESET} ${C_YELLOW}$1${C_RESET}" | tee -a "$LOG_FILE"
}

error_echo() {
  echo -e "${C_RED}${C_BOLD}❌ Arcane Anomaly:${C_RESET} ${C_RED}$1${C_RESET}" | tee -a "$LOG_FILE"
  exit 1
}

prompt_input() {
  local prompt_text="$1"
  local var_name="$2"
  read -p "$(echo -e ${C_YELLOW}${C_BOLD}${prompt_text}:${C_RESET} ${C_YELLOW})" $var_name
  if [[ -z "${!var_name}" ]]; then
    error_echo "Input cannot be empty. Aborting spell."
  fi
  echo | tee -a "$LOG_FILE"
}

prompt_yes_no() {
  local prompt_text="$1"
  local response
  read -p "$(echo -e ${C_YELLOW}${C_BOLD}${prompt_text} [y/N]:${C_RESET} ${C_YELLOW})" response
  case "$response" in
    [Yy]*) return 0 ;;
    *) return 1 ;;
  esac
}

progress_bar() {
  local duration=$1
  local message=$2
  if [[ -t 1 ]]; then
    local cols=$(tput cols 2>/dev/null || echo 80)
    local width=$((cols - 20))
    for ((i=0; i<=duration; i++)); do
      local progress=$((i * width / duration))
      local percent=$((i * 100 / duration))
      local bar=$(printf "%${progress}s" | tr ' ' '#')
      printf "\r${C_CYAN}${message}: [${bar}%${width}s] %d%%" "$percent"
      sleep 0.1
    done
    echo
  else
    wizard_echo "$message..."
  fi
}

# --- Initialize Logging ---
echo "Pyrmethus Setup Log - $(date)" > "$LOG_FILE"
wizard_echo "Initiating the Enhanced Gemini Code Review setup spell..."

# --- Parse Arguments ---
DRY_RUN=0
FORCE=0
while [[ "$1" ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      wizard_echo "Casting in dry-run mode. No changes will be made."
      ;;
    --force)
      FORCE=1
      wizard_echo "Force mode enabled. Overwrite prompts will be skipped."
      ;;
    *)
      error_echo "Unknown argument: $1"
      ;;
  esac
  shift
done

# --- Spell Preparation ---
wizard_echo "Verifying prerequisites..."

# Check for git
if ! command -v git &> /dev/null; then
  error_echo "'git' command not found. Please install it first: pkg install git"
fi
wizard_echo "Verified 'git' presence."

# Check for curl
if ! command -v curl &> /dev/null; then
  error_echo "'curl' command not found. Please install it first: pkg install curl"
fi
wizard_echo "Verified 'curl' presence."

# Check for gh
if ! command -v gh &> /dev/null; then
  warning_echo "'gh' (GitHub CLI) not found. Manual workflow triggering may require alternative methods."
fi
wizard_echo "Verified environment tools."

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
  error_echo "You must run this script from within your local git repository clone."
fi
wizard_echo "Confirmed presence within a git repository."

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
  warning_echo "Uncommitted changes detected. Please commit or stash them before proceeding."
  if [[ $FORCE -eq 0 ]] && ! prompt_yes_no "Continue anyway?"; then
    error_echo "Aborted due to uncommitted changes."
  fi
fi

# --- Gather Ingredients (User Input) ---
prompt_input "Enter your GitHub Username" GITHUB_USER
prompt_input "Enter your GitHub Repository Name" GITHUB_REPO

# Sanitize inputs
GITHUB_USER=$(echo "$GITHUB_USER" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
GITHUB_REPO=$(echo "$GITHUB_REPO" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

# Validate repository against git remote
wizard_echo "Verifying repository alignment with git remote..."
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+)/([^/.]+)(\.git)?$ ]]; then
  REMOTE_USER=${BASH_REMATCH[1]}
  REMOTE_REPO=${BASH_REMATCH[2]}
  if [[ "$REMOTE_USER" != "$GITHUB_USER" || "$REMOTE_REPO" != "$GITHUB_REPO" ]]; then
    warning_echo "Local repository is linked to https://github.com/$REMOTE_USER/$REMOTE_REPO RICH USER/$REMOTE_REPO, but you entered $GITHUB_USER/$GITHUB_REPO."
    if [[ $FORCE -eq 0 ]] && ! prompt_yes_no "Continue with $GITHUB_USER/$GITHUB_REPO?"; then
      error_echo "Aborted due to repository mismatch."
    fi
  fi
else
  warning_echo "Could not parse GitHub repository from remote URL: $REMOTE_URL"
fi

# Validate repository existence
wizard_echo "Consulting the GitHub Oracle to verify repository..."
if ! curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/$GITHUB_USER/$GITHUB_REPO" | grep -q "200"; then
  error_echo "Repository https://github.com/$GITHUB_USER/$GITHUB_REPO does not exist or is inaccessible."
fi
success_echo "Repository verified."

# Check branch tracking
wizard_echo "Verifying branch status..."
if git show-ref --quiet "refs/heads/$BRANCH_NAME"; then
  TRACKING_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name "$BRANCH_NAME@{upstream}" 2>/dev/null)
  if [[ "$TRACKING_BRANCH" != "origin/main" && "$TRACKING_BRANCH" != "origin/$BRANCH_NAME" ]]; then
    warning_echo "Branch $BRANCH_NAME is tracking $TRACKING_BRANCH, not origin/main or origin/$BRANCH_NAME."
  fi
fi

# --- Define File Paths ---
WORKFLOW_DIR=".github/workflows"
SCRIPT_DIR=".github/scripts"
WORKFLOW_FILE="$WORKFLOW_DIR/$WORKFLOW_FILENAME"
SCRIPT_FILE="$SCRIPT_DIR/analyze_code.py"

# --- Backup Existing Files ---
backup_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local backup="${file}.bak.$(date +%Y%m%d%H%M%S)"
    wizard_echo "Backing up existing $file to $backup..."
    cp "$file" "$backup" || error_echo "Failed to backup $file."
    success_echo "Backup created: $backup"
  fi
}

if [[ $DRY_RUN -eq 0 ]]; then
  backup_file "$WORKFLOW_FILE"
  backup_file "$SCRIPT_FILE"
fi

# --- Prompt for Overwrite ---
if [[ $FORCE -eq 0 && (-f "$WORKFLOW_FILE" || -f "$SCRIPT_FILE") ]]; then
  warning_echo "Existing workflow or script files detected."
  if ! prompt_yes_no "Overwrite existing files?"; then
    error_echo "Aborted to preserve existing files."
  fi
fi

# --- The Incantation (File Creation) ---
wizard_echo "Creating arcane directory structures..."
if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$WORKFLOW_DIR" "$SCRIPT_DIR" || error_echo "Failed to create directories."
fi
success_echo "Directories ${C_YELLOW}$WORKFLOW_DIR/${C_GREEN} and ${C_YELLOW}$SCRIPT_DIR/${C_GREEN} ensured."

wizard_echo "Weaving the ${C_YELLOW}$WORKFLOW_FILENAME${C_CYAN} workflow spell..."
if [[ $DRY_RUN -eq 0 ]]; then
  progress_bar 10 "Inscribing workflow"
  cat << 'EOF' > "$WORKFLOW_FILE"
# .github/workflows/gemini_code_review.yml
name: Gemini Code Review Spell
on:
  pull_request:
    types: [opened, synchronize]
  workflow_dispatch:
    inputs:
      base_branch:
        description: 'Base branch for diff comparison (default: main)'
        required: false
        default: 'main'

jobs:
  gemini-review:
    name: Cast Gemini Review Spell
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Get PR Diff
        id: get_diff
        shell: bash
        run: |
          echo "Forging the diff..."
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            BASE_BRANCH="${{ github.event.inputs.base_branch || 'main' }}"
            echo "Manual run: Comparing HEAD with $BASE_BRANCH"
            git fetch origin $BASE_BRANCH:$BASE_BRANCH
            DIFF_CONTENT=$(git diff $BASE_BRANCH HEAD -- . ':(exclude).github/*')
          else
            echo "PR run: Comparing ${{ github.event.pull_request.base.ref }} with PR head"
            git fetch origin ${{ github.event.pull_request.base.ref }}:${{ github.event.pull_request.base.ref }}
            git fetch origin ${{ github.event.pull_request.head.ref }}:${{ github.event.pull_request.head.ref }}
            DIFF_CONTENT=$(git diff "origin/${{ github.event.pull_request.base.ref }}" "${{ github.event.pull_request.head.sha }}" -- . ':(exclude).github/*')
          fi
          if [ -z "$DIFF_CONTENT" ]; then
            echo "No code changes detected to analyze."
            echo "diff_content=" >> $GITHUB_OUTPUT
          else
            echo "Diff captured successfully."
            echo "diff_content<<EOF" >> $GITHUB_OUTPUT
            echo "$DIFF_CONTENT" >> $GITHUB_OUTPUT
            echo "EOF" >> $GITHUB_OUTPUT
          fi
        env:
          GITHUB_BASE_REF: ${{ github.event.pull_request.base.ref }}
          GITHUB_HEAD_REF: ${{ github.event.pull_request.head.ref }}

      - name: Set up Python Environment
        if: steps.get_diff.outputs.diff_content != ''
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Dependencies
        if: steps.get_diff.outputs.diff_content != ''
        run: |
          python -m pip install --upgrade pip
          pip install google-generativeai requests colorama

      - name: Analyze Code with Gemini API
        if: steps.get_diff.outputs.diff_content != ''
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          REPO_NAME: ${{ github.repository }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
          PR_DIFF: ${{ steps.get_diff.outputs.diff_content }}
          COMMIT_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
          EVENT_NAME: ${{ github.event_name }}
        run: |
          python .github/scripts/analyze_code.py
          cat review_report.json

      - name: Upload Review Report
        if: steps.get_diff.outputs.diff_content != ''
        uses: actions/upload-artifact@v4
        with:
          name: gemini-review-report
          path: review_report.json
          retention-days: 7
EOF
  # Verify workflow file integrity
  if [[ ! -s "$WORKFLOW_FILE" ]]; then
    error_echo "Workflow file ${C_YELLOW}$WORKFLOW_FILE${C_RED} is empty or not created."
  fi
fi
success_echo "Workflow file ${C_YELLOW}$WORKFLOW_FILE${C_GREEN} inscribed."

wizard_echo "Conjuring the ${C_YELLOW}analyze_code.py${C_CYAN} Python familiar..."
if [[ $DRY_RUN -eq 0 ]]; then
  progress_bar 10 "Summoning Python familiar"
  cat << 'EOF' > "$SCRIPT_FILE"
# .github/scripts/analyze_code.py
import os
import json
import requests
import google.generativeai as genai
from colorama import init, Fore, Style, Back

init(autoreset=True)

MAX_CHUNK_TOKENS = 3800
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
GITHUB_API_BASE_URL = "https://api.github.com"
REPORT_FILE = "review_report.json"

def print_wizard_message(message, color=Fore.CYAN):
    print(color + Style.BRIGHT + "✨ Pyrmethus whispers: " + Style.RESET_ALL + color + message + Style.RESET_ALL)

def print_error_message(message):
    print(Back.RED + Fore.WHITE + Style.BRIGHT + "⚠️ Arcane Anomaly: " + Style.RESET_ALL + Fore.RED + message + Style.RESET_ALL)

def save_report(issues):
    """Save analysis results to a report file."""
    report = {"total_issues": len(issues), "issues": issues}
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print_wizard_message(f"Review report saved to {REPORT_FILE}", Fore.GREEN)

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
        print_error_message(f"Failed to decode Gemini's JSON response: {json_err}")
        return [{"error": "JSON decode error", "details": str(json_err), "raw_response": response.text[:500]}]
    except Exception as e:
        print_error_message(f"Gemini API error: {e}")
        return [{"error": "API error", "details": str(e)}]

def post_github_comment(repo_name, pr_number, commit_sha, github_token, issue_report):
    if not pr_number:
        print_wizard_message("No pull request number provided (manual workflow run?). Skipping comment posting.", Fore.YELLOW)
        return
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
        print_error_message(f"Failed to post GitHub comment for {file_path}:{line_number}: {e}")
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
    event_name = os.getenv('EVENT_NAME')

    if not gemini_api_key:
        print_error_message("GEMINI_API_KEY is missing. Please set it in GitHub Secrets.")
        save_report([{"error": "Missing GEMINI_API_KEY"}])
        exit(1)
    if not github_token:
        print_error_message("GITHUB_TOKEN is missing. Please ensure the workflow has permissions.")
        save_report([{"error": "Missing GITHUB_TOKEN"}])
        exit(1)
    if not repo_name:
        print_error_message("REPO_NAME is missing. Please ensure the workflow is run in a GitHub repository.")
        save_report([{"error": "Missing REPO_NAME"}])
        exit(1)
    if not pr_diff:
        print_error_message("PR_DIFF is missing. No code changes to analyze.")
        save_report([{"error": "No code changes detected"}])
        exit(1)
    if event_name == 'workflow_dispatch':
        print_wizard_message("Manual workflow run detected. Analysis will proceed, but comments require a pull request.", Fore.YELLOW)
        pr_number = None
        commit_sha = commit_sha or 'unknown'

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print_wizard_message("Gemini Oracle configured successfully.", Fore.GREEN)
    except Exception as e:
        print_error_message(f"Failed to configure Gemini API: {e}")
        save_report([{"error": "Gemini API configuration failed", "details": str(e)}])
        exit(1)

    diff_chunks = chunk_diff(pr_diff)
    all_issues = []
    for chunk, file_context in diff_chunks:
        if not chunk.strip():
            continue
        print_wizard_message(f"Analyzing chunk related to file: {Fore.YELLOW}{file_context or 'Multiple/Unknown'}", Fore.BLUE)
        issues = analyze_code_chunk_with_gemini(chunk, model)
        for issue in issues:
            if 'file_path' not in issue or not issue['file_path']:
                issue['file_path'] = file_context if file_context else 'unknown_file'
            all_issues.append(issue)
            post_github_comment(repo_name, pr_number, commit_sha, github_token, issue)

    save_report(all_issues)
    print_wizard_message(f"Code review spell complete. Found {Fore.YELLOW}{len(all_issues)}{Fore.MAGENTA} potential issues.", Fore.MAGENTA + Style.BRIGHT)

if __name__ == "__main__":
    main()
EOF
  # Verify Python script integrity
  if [[ ! -s "$SCRIPT_FILE" ]]; then
    error_echo "Python script ${C_YELLOW}$SCRIPT_FILE${C_RED} is empty or not created."
  fi
  if ! grep -q "def main():" "$SCRIPT_FILE"; then
    error_echo "Python script ${C_YELLOW}$SCRIPT_FILE${C_RED} appears corrupted. Check $LOG_FILE for details."
  fi
fi
success_echo "Python familiar ${C_YELLOW}$SCRIPT_FILE${C_GREEN} conjured."

# --- Committing the Spell to the Repository Grimoire ---
if [[ $DRY_RUN -eq 0 ]]; then
  wizard_echo "Binding the spell to your local repository..."
  
  # Create or update branch
  if [[ $FORCE -eq 0 ]] && prompt_yes_no "Create or update branch ($BRANCH_NAME) for these changes?"; then
    wizard_echo "Crafting or updating branch ${C_YELLOW}$BRANCH_NAME${C_CYAN}..."
    if git show-ref --quiet "refs/heads/$BRANCH_NAME"; then
      git checkout "$BRANCH_NAME" || error_echo "Failed to switch to branch $BRANCH_NAME."
    else
      git checkout -b "$BRANCH_NAME" || error_echo "Failed to create branch $BRANCH_NAME."
    fi
    success_echo "On branch $BRANCH_NAME."
  fi

  git add "$WORKFLOW_DIR/" "$SCRIPT_DIR/" || error_echo "Failed to stage changes."
  git commit -m "feat: Enhance Gemini Code Review GitHub Action workflow

Conjured by Pyrmethus's script:
- Updates workflow file $WORKFLOW_FILE with workflow_dispatch inputs and artifact upload
- Enhances Python script .github/scripts/analyze_code.py with report generation
- Fixes here-document issues for Termux compatibility" || error_echo "Failed to commit changes."
  success_echo "Changes committed locally."
fi

# --- Sending the Spell to the GitHub Ether ---
if [[ $DRY_RUN -eq 0 ]]; then
  CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  wizard_echo "Dispatching the spell to the GitHub ether (origin/${C_YELLOW}${CURRENT_BRANCH}${C_CYAN})..."
  git push --set-upstream origin "$CURRENT_BRANCH" || error_echo "Failed to push changes to remote 'origin'. Check your connection and permissions."
  success_echo "Spell successfully dispatched to GitHub!"
fi

# --- Final Words of Power ---
wizard_echo "The setup is complete! Key steps to proceed:"
warning_echo "Ensure the ${C_YELLOW}GEMINI_API_KEY${C_YELLOW} secret is set in your repository settings on GitHub."
wizard_echo "Go to: ${C_BOLD}https://github.com/$GITHUB_USER/$GITHUB_REPO/settings/secrets/actions${C_RESET}"
wizard_echo "Create or verify a ${C_BOLD}'New repository secret'${C_RESET} ${C_CYAN}named ${C_YELLOW}GEMINI_API_KEY${C_CYAN} with your API key value.${C_RESET}"
wizard_echo "To manually trigger the workflow:"
wizard_echo "${C_BOLD}gh workflow run $WORKFLOW_FILENAME --ref $BRANCH_NAME --field base_branch=main${C_RESET}"
wizard_echo "To view the latest run: ${C_BOLD}gh run list --workflow=$WORKFLOW_FILENAME${C_RESET}"
wizard_echo "To inspect a run: ${C_BOLD}gh run view <RUN_ID>${C_RESET}"
wizard_echo "Log of this conjuration saved to ${C_YELLOW}$LOG_FILE${C_CYAN}."
wizard_echo "May your code be ever insightful!"