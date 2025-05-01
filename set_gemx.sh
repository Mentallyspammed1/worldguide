#!/data/data/com.termux/files/usr/bin/env bash

# Pyrmethus's Repository Setup Spell

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
  echo -e "${C_MAGENTA}${C_BOLD}✨ Pyrmethus whispers:${C_RESET} ${C_CYAN}$1${C_RESET}"
}

success_echo() {
  echo -e "${C_GREEN}${C_BOLD}✅ Success:${C_RESET} ${C_GREEN}$1${C_RESET}"
}

warning_echo() {
  echo -e "${C_YELLOW}${C_BOLD}⚠️ Caution:${C_RESET} ${C_YELLOW}$1${C_RESET}"
}

error_echo() {
  echo -e "${C_RED}${C_BOLD}❌ Arcane Anomaly:${C_RESET} ${C_RED}$1${C_RESET}"
}

prompt_input() {
  local prompt_text="$1"
  local var_name="$2"
  read -p "$(echo -e ${C_YELLOW}${C_BOLD}${prompt_text}:${C_RESET} ${C_YELLOW})" $var_name
  if [[ -z "${!var_name}" ]]; then
      error_echo "Input cannot be empty. Aborting spell."
      exit 1
  fi
  echo
}

# --- Spell Preparation ---
wizard_echo "Preparing the Gemini Code Review setup spell..."

# Check for git
if ! command -v git &> /dev/null; then
    error_echo "'git' command not found. Please install it first: pkg install git"
    exit 1
fi
wizard_echo "Verified 'git' presence."

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    error_echo "You must run this script from within your local git repository clone."
    exit 1
fi
wizard_echo "Confirmed presence within a git repository."

# --- Gather Ingredients (User Input) ---
prompt_input "Enter your GitHub Username" GITHUB_USER
prompt_input "Enter your GitHub Repository Name" GITHUB_REPO

# --- Define File Paths ---
WORKFLOW_DIR=".github/workflows"
SCRIPT_DIR=".github/scripts"
WORKFLOW_FILE="$WORKFLOW_DIR/gemini_code_review.yml"
SCRIPT_FILE="$SCRIPT_DIR/analyze_code.py"

# --- The Incantation (File Creation) ---
wizard_echo "Creating arcane directory structures..."
mkdir -p "$WORKFLOW_DIR"
mkdir -p "$SCRIPT_DIR"
success_echo "Directories ${C_YELLOW}$WORKFLOW_DIR/${C_GREEN} and ${C_YELLOW}$SCRIPT_DIR/${C_GREEN} ensured."

wizard_echo "Weaving the ${C_YELLOW}gemini_code_review.yml${C_CYAN} workflow spell..."
cat << 'EOF' > "$WORKFLOW_FILE"
# .github/workflows/gemini_code_review.yml
name: Gemini Code Review Spell
on:
  pull_request:
    types: [opened, synchronize]

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
          git fetch origin ${{ github.event.pull_request.base.ref }}:${{ github.event.pull_request.base.ref }}
          git fetch origin ${{ github.event.pull_request.head.ref }}:${{ github.event.pull_request.head.ref }}
          DIFF_CONTENT=$(git diff "origin/${{ github.event.pull_request.base.ref }}" "${{ github.event.pull_request.head.sha }}" -- . ':(exclude).github/*')
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
          python-version: '3.10'

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
          COMMIT_SHA: ${{ github.event.pull_request.head.sha }}
        run: |
          python .github/scripts/analyze_code.py
EOF
success_echo "Workflow file ${C_YELLOW}$WORKFLOW_FILE${C_GREEN} inscribed."

wizard_echo "Conjuring the ${C_YELLOW}analyze_code.py${C_CYAN} Python familiar..."
cat << 'EOF' > "$SCRIPT_FILE"
# .github/scripts/analyze_code.py
import os
import json
import requests
import google.generativeai as genai
from colorama import init, Fore, Style, Back

init(autoreset=True)

MAX_CHUNK_TOKENS = 3800
GEMINI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25'
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
    1.  `file_path`: The full path of the file where the issue occurs (e.g., "src/utils/helpers.py"). Infer this from the '+++ b/...' line in the diff chunk.
    2.  `line_number`: The approximate line number *in the new file version* where the issue begins. Estimate based on the '@@ ... +start,count @@' hunk header and '+' lines.
    3.  `description`: A concise explanation of the potential bug or vulnerability.
    4.  `affected_code`: The specific line(s) from the diff (lines starting with '+') that contain the issue.
    5.  `suggested_fix`: A concrete code snippet demonstrating how to fix the issue. If the fix involves removing lines, indicate that clearly.

    If no significant issues are found in this chunk, respond with an empty JSON array: `[]`.

    Respond ONLY with a valid JSON array containing objects matching the structure described above.

    Diff Chunk:
    ```diff
    {diff_chunk}