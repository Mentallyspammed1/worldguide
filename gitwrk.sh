#!/bin/bash

# setup_workflows.sh
# Automates the setup of GitHub Actions workflows (pyrmin.yml, lint-and-format.yml)
# and xfix_files.py for a Python coding repository.

set -e

# Configuration variables
REPO_DIR="$(pwd)"
WORKFLOWS_DIR="$REPO_DIR/.github/workflows"
XFIX_FILE="$REPO_DIR/xfix_files.py"
PYPROJECT_FILE="$REPO_DIR/pyproject.toml"
GITHUB_API_URL="https://api.github.com"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REPO_NAME="$(basename "$REPO_DIR")"
REPO_OWNER="$(git config --get remote.origin.url | sed -n 's#.*github.com[:/]\(.*\)/.*\.git#\1#p')"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print messages
print_message() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${NC}"
}

# Check if running in a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    print_message "$RED" "Error: This is not a Git repository. Please run in a Git repository."
    exit 1
fi

# Check if curl is installed (needed for GitHub API)
if ! command -v curl >/dev/null 2>&1; then
    print_message "$RED" "Error: curl is required but not installed. Please install curl."
    exit 1
fi

# Create workflows directory
print_message "$YELLOW" "Creating .github/workflows directory..."
mkdir -p "$WORKFLOWS_DIR"

# Write pyrmin.yml
print_message "$YELLOW" "Creating .github/workflows/pyrmin.yml..."
cat > "$WORKFLOWS_DIR/pyrmin.yml" << 'EOF'
name: Code Enhancement Workflow

on:
  workflow_dispatch:
    inputs:
      base_directory:
        description: 'Base directory to search for Python files'
        required: true
        default: '.'
      file_pattern:
        description: 'Glob pattern for Python files (e.g., "**/*.py")'
        required: true
        default: '**/*.py'
      google_api_key:
        description: 'Google API Key (leave blank to use repository secret)'
        required: false
      max_api_calls:
        description: 'Maximum API calls per minute (1-60)'
        required: false
        default: '59'
      commit_message:
        description: 'Custom commit message for the enhancement changes'
        required: false
        default: 'Apply automated code enhancements'

jobs:
  enhance_python_code:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install google-generativeai aiohttp

      - name: Validate Inputs
        run: |
          FILE_PATTERN="${{ inputs.file_pattern }}"
          if [[ ! "$FILE_PATTERN" =~ \.py$ ]] || [[ "$FILE_PATTERN" =~ \[.*\] ]] || [[ "$FILE_PATTERN" =~ [[:space:]]or[[:space:]] ]]; then
            echo "Error: file_pattern must target .py files (e.g., '*.py', '**/*.py') and not contain square brackets or 'or'"
            echo "Invalid pattern provided: $FILE_PATTERN"
            exit 1
          fi
          MAX_CALLS="${{ inputs.max_api_calls }}"
          if ! [[ "$MAX_CALLS" =~ ^[0-9]+$ ]] || [ "$MAX_CALLS" -lt 1 ] || [ "$MAX_CALLS" -gt 60 ]; then
            echo "Error: max_api_calls must be a number between 1 and 60"
            exit 1
          fi
          if [ -z "${{ inputs.google_api_key || secrets.GOOGLE_API_KEY }}" ]; then
            echo "Error: Google API key not provided and GOOGLE_API_KEY secret not set"
            exit 1
          fi
          if [ ! -d "${{ inputs.base_directory }}" ]; then
            echo "Error: Base directory '${{ inputs.base_directory }}' does not exist"
            exit 1
          fi

      - name: Ensure Enhancement Script
        run: |
          if [ ! -f xfix_files.py ]; then
            echo "Error: xfix_files.py not found in repository root"
            exit 1
          fi
          chmod +x xfix_files.py

      - name: List Repository Files
        run: |
          echo "Repository contents:" | tee -a enhancement_log.txt
          find . -type f | tee -a enhancement_log.txt

      - name: List Matched Python Files
        run: |
          echo "Matched files for pattern ${{ inputs.file_pattern }} in directory ${{ inputs.base_directory }}:" | tee matched_files.txt
          find "${{ inputs.base_directory }}" -type f -path "${{ inputs.file_pattern }}" | tee -a matched_files.txt
          if ! grep -q "[^[:space:]]" matched_files.txt; then
            echo "Warning: No Python files matched the pattern ${{ inputs.file_pattern }} in directory ${{ inputs.base_directory }}" | tee -a matched_files.txt
          fi

      - name: Run Code Enhancement Script
        env:
          GOOGLE_API_KEY: ${{ inputs.google_api_key || secrets.GOOGLE_API_KEY }}
          MAX_API_CALLS: ${{ inputs.max_api_calls }}
        run: |
          touch enhancement_log.txt
          ./xfix_files.py "${{ inputs.base_directory }}" "${{ inputs.file_pattern }}" >> enhancement_log.txt 2>&1 || {
            echo "Error: Enhancement script failed:" >> enhancement_log.txt
            cat enhancement_log.txt
            exit 1
          }

      - name: Debug Git Status
        run: |
          git status | tee -a enhancement_log.txt
          echo "Staged changes:" | tee -a enhancement_log.txt
          git diff --cached || echo "No staged changes" | tee -a enhancement_log.txt

      - name: Check for Changes
        id: git_status
        run: |
          git add .
          if ! git diff --cached --quiet; then
            echo "Changes detected" | tee -a enhancement_log.txt
            echo "changes_made=true" >> $GITHUB_OUTPUT
          else
            echo "No changes detected" | tee -a enhancement_log.txt
            echo "changes_made=false" >> $GITHUB_OUTPUT
          fi

      - name: Configure Git
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Create New Branch and Commit Changes
        if: steps.git_status.outputs.changes_made == 'true'
        id: commit-changes
        run: |
          BRANCH_NAME="enhance-$(date +%Y%m%d-%H%M%S)-${{ github.run_id }}"
          echo "Creating branch: $BRANCH_NAME" | tee -a enhancement_log.txt
          git checkout -b "$BRANCH_NAME" || {
            echo "Error: Failed to create branch $BRANCH_NAME" | tee -a enhancement_log.txt
            exit 1
          }
          git commit -m "${{ inputs.commit_message }}" || {
            echo "Error: Failed to commit changes" | tee -a enhancement_log.txt
            git status
            exit 1
          }
          echo "branch_name=$BRANCH_NAME" >> $GITHUB_OUTPUT

      - name: Push Changes
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          BRANCH_NAME="${{ steps.commit-changes.outputs.branch_name }}"
          echo "Pushing branch $BRANCH_NAME to origin" | tee -a enhancement_log.txt
          git push origin "$BRANCH_NAME" --force-with-lease || {
            echo "Error: Failed to push branch $BRANCH_NAME" | tee -a enhancement_log.txt
            git status
            git remote -v
            exit 1
          }

      - name: Create Pull Request
        if: steps.git_status.outputs.changes_made == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          branch: ${{ steps.commit-changes.outputs.branch_name }}
          base: ${{ github.ref_name }}
          title: "Automated Code Enhancements"
          body: |
            This PR contains automated code enhancements for Python (.py) files using the Gemini API.

            **Details:**
            - Base directory: ${{ inputs.base_directory }}
            - File pattern: ${{ inputs.file_pattern }}
            - API calls per minute: ${{ inputs.max_api_calls }}
            - Commit message: ${{ inputs.commit_message }}

            Please review the changes and the attached `enhancement_log.txt` and `matched_files.txt` artifacts.
          labels: |
            enhancement
            automated
          assignees: ${{ github.actor }}
          commit-message: "${{ inputs.commit_message }}"
          delete-branch: true

      - name: Upload Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: enhancement-artifacts
          path: |
            matched_files.txt
            enhancement_log.txt
          retention-days: 7
          if-no-files-found: warn
EOF

# Write lint-and-format.yml
print_message "$YELLOW" "Creating .github/workflows/lint-and-format.yml..."
cat > "$WORKFLOWS_DIR/lint-and-format.yml" << 'EOF'
name: Lint and Format Python Code

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    branches:
      - main
      - develop
  workflow_dispatch:
    inputs:
      base_directory:
        description: 'Base directory to search for Python files'
        required: true
        default: '.'
      file_pattern:
        description: 'Glob pattern for Python files (e.g., "**/*.py")'
        required: true
        default: '**/*.py'
      commit_message:
        description: 'Custom commit message for formatting changes'
        required: false
        default: 'Apply automated linting and formatting'

jobs:
  lint-and-format:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install ruff mypy
          # Install project dependencies if pyproject.toml or requirements.txt exists
          if [ -f pyproject.toml ]; then
            pip install .
          elif [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi

      - name: Validate Inputs
        run: |
          FILE_PATTERN="${{ inputs.file_pattern }}"
          if [[ ! "$FILE_PATTERN" =~ \.py$ ]] || [[ "$FILE_PATTERN" =~ \[.*\] ]] || [[ "$FILE_PATTERN" =~ [[:space:]]or[[:space:]] ]]; then
            echo "Error: file_pattern must target .py files (e.g., '*.py', '**/*.py') and not contain square brackets or 'or'"
            echo "Invalid pattern provided: $FILE_PATTERN"
            exit 1
          fi
          if [ ! -d "${{ inputs.base_directory }}" ]; then
            echo "Error: Base directory '${{ inputs.base_directory }}' does not exist"
            exit 1
          fi

      - name: Initialize Lint Report
        run: |
          echo "Linting and Formatting Report" > lint_report.txt
          echo "=============================" >> lint_report.txt

      - name: List Repository Files
        run: |
          echo "Repository contents:" >> lint_report.txt
          find . -type f >> lint_report.txt

      - name: List Matched Python Files
        run: |
          echo "Matched files for pattern ${{ inputs.file_pattern }} in directory ${{ inputs.base_directory }}:" >> lint_report.txt
          find "${{ inputs.base_directory }}" -type f -path "${{ inputs.file_pattern }}" | tee -a lint_report.txt
          if ! grep -q "[^[:space:]]" lint_report.txt; then
            echo "Warning: No Python files matched the pattern ${{ inputs.file_pattern }} in directory ${{ inputs.base_directory }}" >> lint_report.txt
          fi

      - name: Run Ruff Linting
        run: |
          echo "Running Ruff linting..." >> lint_report.txt
          ruff check "${{ inputs.base_directory }}/${{ inputs.file_pattern }}" --output-format=full >> lint_report.txt 2>&1 || {
            echo "Ruff linting found issues, but continuing to formatting" >> lint_report.txt
          }

      - name: Run Ruff Formatting
        run: |
          echo "Running Ruff formatting..." >> lint_report.txt
          ruff format "${{ inputs.base_directory }}/${{ inputs.file_pattern }}" >> lint_report.txt 2>&1 || {
            echo "Error: Ruff formatting failed" >> lint_report.txt
            exit 1
          }

      - name: Run Mypy Type Checking
        run: |
          echo "Running Mypy type checking..." >> lint_report.txt
          mypy "${{ inputs.base_directory }}/${{ inputs.file_pattern }}" >> lint_report.txt 2>&1 || {
            echo "Mypy found type issues, but continuing" >> lint_report.txt
          }

      - name: Debug Git Status
        run: |
          git status >> lint_report.txt
          echo "Staged changes:" >> lint_report.txt
          git diff --cached || echo "No staged changes" >> lint_report.txt

      - name: Check for Changes
        id: git_status
        run: |
          git add .
          if ! git diff --cached --quiet; then
            echo "Changes detected" >> lint_report.txt
            echo "changes_made=true" >> $GITHUB_OUTPUT
          else
            echo "No changes detected" >> lint_report.txt
            echo "changes_made=false" >> $GITHUB_OUTPUT
          fi

      - name: Configure Git
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Create New Branch and Commit Changes
        if: steps.git_status.outputs.changes_made == 'true'
        id: commit-changes
        run: |
          BRANCH_NAME="lint-format-$(date +%Y%m%d-%H%M%S)-${{ github.run_id }}"
          echo "Creating branch: $BRANCH_NAME" >> lint_report.txt
          git checkout -b "$BRANCH_NAME" || {
            echo "Error: Failed to create branch $BRANCH_NAME" >> lint_report.txt
            exit 1
          }
          git commit -m "${{ inputs.commit_message }}" || {
            echo "Error: Failed to commit changes" >> lint_report.txt
            git status
            exit 1
          }
          echo "branch_name=$BRANCH_NAME" >> $GITHUB_OUTPUT

      - name: Push Changes
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          BRANCH_NAME="${{ steps.commit-changes.outputs.branch_name }}"
          echo "Pushing branch $BRANCH_NAME to origin" >> lint_report.txt
          git push origin "$BRANCH_NAME" --force-with-lease || {
            echo "Error: Failed to push branch $BRANCH_NAME" >> lint_report.txt
            git status
            git remote -v
            exit 1
          }

      - name: Create Pull Request
        if: steps.git_status.outputs.changes_made == 'true'
        uses: peter-evans/create-pull-request@v6
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          branch: ${{ steps.commit-changes.outputs.branch_name }}
          base: ${{ github.ref_name }}
          title: "Automated Linting and Formatting"
          body: |
            This PR contains automated linting and formatting changes for Python (.py) files using Ruff and Mypy.

            **Details:**
            - Base directory: ${{ inputs.base_directory }}
            - File pattern: ${{ inputs.file_pattern }}
            - Commit message: ${{ inputs.commit_message }}

            Please review the changes and the attached `lint_report.txt` artifact for details.
          labels: |
            linting
            formatting
            automated
          assignees: ${{ github.actor }}
          commit-message: "${{ inputs.commit_message }}"
          delete-branch: true

      - name: Upload Lint Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: lint-report
          path: lint_report.txt
          retention-days: 7
          if-no-files-found: warn
EOF

# Write xfix_files.py
print_message "$YELLOW" "Creating xfix_files.py..."
cat > "$XFIX_FILE" << 'EOF'
#!/usr/bin/env python3
import os
import glob
import google.generativeai as genai
import time
import logging
from pathlib import Path
import sys
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhancement_log.txt', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def configure_api():
    """Configure the Gemini API with the provided API key and safety settings."""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25', safety_settings=safety_settings)

def get_python_files(base_dir, pattern):
    """Get list of Python files matching the pattern in the base directory."""
    try:
        base_path = Path(base_dir)
        files = [str(f) for f in base_path.glob(pattern) if f.is_file()]
        with open('matched_files.txt', 'w', encoding='utf-8') as f:
            f.write(f"Matched files for pattern '{pattern}' in directory '{base_dir}':\n")
            if files:
                f.write('\n'.join(files) + '\n')
            else:
                f.write("No Python files matched the pattern\n")
        logger.info(f"Found {len(files)} Python files matching pattern '{pattern}' in '{base_dir}'")
        return sorted(files)
    except Exception as e:
        logger.error(f"Invalid file pattern '{pattern}' in directory '{base_dir}': {str(e)}")
        with open('matched_files.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: Invalid file pattern '{pattern}' in directory '{base_dir}': {str(e)}\n")
        sys.exit(1)

def read_file(file_path):
    """Read content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Successfully read {file_path}")
            return content
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {str(e)}")
        return None

def write_file(file_path, content):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote enhanced code to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {str(e)}")
        return False

def enhance_code(model, code, file_path):
    """Enhance Python code using Gemini API."""
    prompt = f"""You are an expert Python code reviewer. Analyze the following Python code and suggest improvements focusing on:
1. Code readability and documentation (e.g., docstrings, comments)
2. Performance optimization (e.g., efficient algorithms, reducing redundancy)
3. Error handling (e.g., try-except blocks, input validation)
4. Python best practices and PEP 8 compliance (e.g., naming conventions, code structure)
5. Type hints where appropriate (e.g., for function parameters and return types)

Provide the enhanced code wrapped in ```python
```

File: {file_path}
```python
{code}
```

Return only the improved code within ```python``` blocks, with comments explaining major changes."""
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            logger.warning(f"No enhancement suggestions for {file_path}")
            return None
        
        # Extract code from response
        code_match = re.search(r'```python\n(.*?)```', response.text, re.DOTALL)
        if not code_match:
            logger.error(f"No valid code block in response for {file_path}")
            return None
            
        enhanced_code = code_match.group(1).strip()
        if enhanced_code == code.strip():
            logger.info(f"No changes suggested for {file_path}")
            return None
            
        logger.info(f"Generated enhancements for {file_path}")
        return enhanced_code
    except Exception as e:
        logger.error(f"API call failed for {file_path}: {str(e)}")
        return None

def main(base_dir, file_pattern):
    """Main function to process and enhance Python files."""
    # Ensure enhancement log exists
    with open('enhancement_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"Code enhancement log for directory '{base_dir}' and pattern '{file_pattern}'\n")
    
    max_api_calls = int(os.environ.get('MAX_API_CALLS', 59))
    logger.info(f"Starting enhancement process with max API calls: {max_api_calls}")
    model = configure_api()
    files = get_python_files(base_dir, file_pattern)
    
    if not files:
        logger.warning("No Python files found to enhance")
        with open('enhancement_log.txt', 'a', encoding='utf-8') as f:
            f.write("No Python files found to enhance\n")
        return
    
    # Calculate delay to respect API rate limit (calls per minute)
    delay = 60.0 / max_api_calls if max_api_calls > 0 else 0
    logger.info(f"API rate limit delay: {delay} seconds per call")
    
    modified_files = 0
    for file_path in files:
        logger.info(f"Processing {file_path}")
        
        original_code = read_file(file_path)
        if original_code is None:
            continue
            
        enhanced_code = enhance_code(model, original_code, file_path)
        if enhanced_code and write_file(file_path, enhanced_code):
            modified_files += 1
        
        # Respect rate limit
        if delay > 0:
            time.sleep(delay)
    
    with open('enhancement_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"Processed {len(files)} files, modified {modified_files} files\n")
    logger.info(f"Completed processing: {len(files)} files processed, {modified_files} files modified")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: ./xfix_files.py <base_directory> <file_pattern>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    file_pattern = sys.argv[2]
    
    if not os.path.isdir(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        sys.exit(1)
    
    main(base_dir, file_pattern)
EOF

# Make xfix_files.py executable
chmod +x "$XFIX_FILE"
print_message "$GREEN" "xfix_files.py created and made executable."

# Create or update pyproject.toml with ruff and mypy configurations
print_message "$YELLOW" "Creating/updating pyproject.toml with ruff and mypy configurations..."
if [ -f "$PYPROJECT_FILE" ]; then
    print_message "$YELLOW" "pyproject.toml exists, checking for ruff and mypy sections..."
    # Check if ruff and mypy sections exist, append if not
    if ! grep -q "\[tool.ruff\]" "$PYPROJECT_FILE"; then
        cat >> "$PYPROJECT_FILE" << 'EOF'

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "D"]
ignore = ["D203"]
EOF
    fi
    if ! grep -q "\[tool.mypy\]" "$PYPROJECT_FILE"; then
        cat >> "$PYPROJECT_FILE" << 'EOF'

[tool.mypy]
strict = true
ignore_missing_imports = true
EOF
    fi
else
    cat > "$PYPROJECT_FILE" << 'EOF'
[project]
name = "my-project"
version = "0.1.0"
dependencies = []

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "D"]
ignore = ["D203"]

[tool.mypy]
strict = true
ignore_missing_imports = true
EOF
fi
print_message "$GREEN" "pyproject.toml configured with ruff and mypy."

# Configure GitHub Actions permissions (requires GITHUB_TOKEN)
if [ -n "$GITHUB_TOKEN" ]; then
    print_message "$YELLOW" "Configuring GitHub Actions permissions..."
    # Check current permissions
    response=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "$GITHUB_API_URL/repos/$REPO_OWNER/$REPO_NAME/actions/permissions")
    
    current_permission=$(echo "$response" | grep -o '"workflow": "[^"]*"' | cut -d'"' -f4)
    
    if [ "$current_permission" != "write" ]; then
        print_message "$YELLOW" "Updating GitHub Actions permissions to read and write..."
        curl -s -X PUT \
            -H "Authorization: Bearer $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$GITHUB_API_URL/repos/$REPO_OWNER/$REPO_NAME/actions/permissions" \
            -d '{"enabled": true, "allowed_actions": "all", "workflow": "write"}' >/dev/null
        if [ $? -eq 0 ]; then
            print_message "$GREEN" "GitHub Actions permissions updated successfully."
        else
            print_message "$RED" "Failed to update GitHub Actions permissions. Please set manually in repository settings."
        fi
    else
        print_message "$GREEN" "GitHub Actions permissions already set to write."
    fi
else
    print_message "$YELLOW" "GITHUB_TOKEN not provided. Please manually ensure GitHub Actions has read and write permissions in repository settings."
fi

# Stage changes
print_message "$YELLOW" "Staging changes..."
git add "$WORKFLOWS_DIR/pyrmin.yml" "$WORKFLOWS_DIR/lint-and-format.yml" "$XFIX_FILE" "$PYPROJECT_FILE"

# Check for changes
if git diff --cached --quiet; then
    print_message "$YELLOW" "No new changes to commit."
else
    print_message "$YELLOW" "Committing changes..."
    git commit -m "Add GitHub Actions workflows for code enhancement and linting" || {
        print_message "$RED" "Error: Failed to commit changes."
        exit 1
    }
    print_message "$GREEN" "Changes committed successfully."
fi

# Instructions for next steps
print_message "$GREEN" "Setup complete! Next steps:"
echo "1. Push changes to your repository:"
echo "   git push origin main"
echo "2. Set up the GOOGLE_API_KEY secret in your GitHub repository:"
echo "   - Go to Settings > Secrets and variables > Actions > New repository secret"
echo "   - Name: GOOGLE_API_KEY"
echo "   - Value: Your Google Gemini API key"
echo "3. Verify GitHub Actions permissions:"
echo "   - Go to Settings > Actions > General > Workflow permissions"
echo "   - Ensure 'Read and write permissions' is selected"
echo "4. Test the workflows:"
echo "   - Go to the Actions tab and run 'Code Enhancement Workflow' or 'Lint and Format Python Code'"
echo "   - Use default inputs (base_directory: '.', file_pattern: '**/*.py')"
echo "5. Ensure your repository contains Python (.py) files for the workflows to process."

exit 0