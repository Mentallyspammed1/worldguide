#!/data/data/com.termux/files/usr/bin/bash

# setup_pyrmethus.sh: Create and commit Pyrmethus files in Termux

# ANSI color codes for vibrant output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

# Log file
LOG_FILE="pyrmethus_setup.log"

# Function to log messages to file and console
log_message() {
    local level="$1"
    local message="$2"
    local color="$3"
    echo -e "${color}[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [$level] $message${RESET}" | tee -a "$LOG_FILE"
}

# Initialize log
echo -e "${MAGENTA}# Pyrmethus Setup Spell Initiated${RESET}" | tee "$LOG_FILE"
log_message "INFO" "Starting setup in Termux environment" "$CYAN"

# Check if running in Termux
if [ ! -d "/data/data/com.termux/files/home" ]; then
    log_message "ERROR" "This script is designed for Termux. Exiting." "$RED"
    exit 1
fi

# Ensure Git is installed
if ! command -v git >/dev/null 2>&1; then
    log_message "INFO" "Installing Git..." "$YELLOW"
    pkg install git -y || {
        log_message "ERROR" "Failed to install Git. Exiting." "$RED"
        exit 1
    }
fi

# Repository directory (current working directory by default)
REPO_DIR="$PWD"
log_message "INFO" "Using repository directory: $REPO_DIR" "$BLUE"

# Create directories
log_message "INFO" "Creating directory structure..." "$CYAN"
mkdir -p "$REPO_DIR/.github/workflows" || {
    log_message "ERROR" "Failed to create .github/workflows directory. Exiting." "$RED"
    exit 1
}

# Write xfix_files.py
XFIX_FILE="$REPO_DIR/xfix_files.py"
log_message "INFO" "Writing xfix_files.py to $XFIX_FILE..." "$BLUE"
cat > "$XFIX_FILE" << 'EOF'
#!/data/data/com.termux/files/usr/bin/env python
from colorama import init, Fore, Style
import google.generativeai as genai
import os
import sys
import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
import diff_match_patch as dmp_module
import tempfile
import subprocess

# Initialize Colorama for vibrant terminal output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('enhancement.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Token-bucket rate limiter for API calls."""
    def __init__(self, calls_per_minute):
        self.rate = calls_per_minute / 60.0  # Tokens per second
        self.capacity = calls_per_minute
        self.tokens = self.capacity
        self.last_refill = time.time()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.rate
            logger.info(Fore.YELLOW + f"# Pausing for {wait_time:.2f}s to respect rate limit..." + Style.RESET_ALL)
            await asyncio.sleep(wait_time)
            self.tokens = min(self.capacity, self.tokens + wait_time * self.rate)
            self.last_refill = time.time()

        self.tokens -= 1

def configure_api(api_key):
    """Configure the Google Generative AI API."""
    if not api_key:
        logger.error(Fore.RED + "Error: GOOGLE_API_KEY is not set." + Style.RESET_ALL)
        sys.exit(1)
    genai.configure(api_key=api_key)
    logger.info(Fore.CYAN + "# API configured successfully." + Style.RESET_ALL)

def load_config(config_path):
    """Load configuration from a JSON file."""
    default_config = {
        "model": "gemini-2.5-pro-exp-03-25",
        "max_calls_per_minute": 59,
        "enhancement_mode": "comprehensive",
        "backup_dir": ".enhancement_backups",
        "max_retries": 3
    }
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            default_config.update(config)
            logger.info(Fore.CYAN + f"# Loaded configuration from {config_path}." + Style.RESET_ALL)
        except Exception as e:
            logger.warning(Fore.YELLOW + f"# Failed to load config: {e}. Using defaults." + Style.RESET_ALL)
    return default_config

def create_backup(file_path, original_code, backup_dir):
    """Create a backup of the original file."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_code)
    logger.info(Fore.MAGENTA + f"# Backed up {file_path} to {backup_path}." + Style.RESET_ALL)
    return backup_path

def compute_diff(original, enhanced):
    """Compute a human-readable diff between original and enhanced code."""
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(original, enhanced)
    dmp.diff_cleanupSemantic(diffs)
    diff_text = []
    for op, data in diffs:
        if op == 0:
            diff_text.append(data)
        elif op == 1:
            diff_text.append(Fore.GREEN + f"+ {data}" + Style.RESET_ALL)
        elif op == -1:
            diff_text.append(Fore.RED + f"- {data}" + Style.RESET_ALL)
    return '\n'.join(diff_text)

def validate_syntax(code, file_path):
    """Validate Python code syntax using py_compile."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    try:
        result = subprocess.run(
            ['python3', '-m', 'py_compile', temp_file_path],
            capture_output=True, text=True, check=True
        )
        os.unlink(temp_file_path)
        return True, ""
    except subprocess.CalledProcessError as e:
        os.unlink(temp_file_path)
        error_message = e.stderr or str(e)
        logger.error(Fore.RED + f"# Syntax error in {file_path}: {error_message}" + Style.RESET_ALL)
        return False, error_message

async def enhance_code(file_path, model_name, mode, rate_limiter, max_retries, dry_run=False):
    """Enhance a Python file with retries and syntax validation."""
    try:
        logger.info(Fore.BLUE + f"# Summoning enhancements for {file_path} (Mode: {mode})..." + Style.RESET_ALL)
        if not os.path.exists(file_path):
            logger.error(Fore.RED + f"Error: File {file_path} does not exist." + Style.RESET_ALL)
            return False

        # Read the original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # Define enhancement prompts based on mode
        prompts = {
            "readability": """Fix all syntax errors and enhance the Python code for readability and maintainability. Follow PEP 8 strictly, add clear docstrings and comments, and use meaningful variable names. Ensure the code is syntactically correct and preserves functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "performance": """Fix all syntax errors and optimize the Python code for performance. Reduce time complexity, minimize memory usage, and use efficient data structures. Ensure the code is syntactically correct and preserves functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "type_hints": """Fix all syntax errors and add type hints to the Python code per PEP 484, ensuring type safety. Ensure the code is syntactically correct and preserves functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "comprehensive": """Fix all syntax errors and enhance the Python code comprehensively: ensure strict PEP 8 compliance, add docstrings and comments, optimize performance (efficient algorithms and data structures), and add type hints per PEP 484. Ensure the code is syntactically correct and preserves functionality. Return only the enhanced code wrapped in ```python ... ```."""
        }
        prompt_template = prompts.get(mode, prompts["comprehensive"])
        prompt = f"{prompt_template}\n\n```python\n{original_code}\n```"

        enhanced_code = None
        for attempt in range(max_retries):
            logger.info(Fore.CYAN + f"# Attempt {attempt + 1}/{max_retries} for {file_path}..." + Style.RESET_ALL)
            await rate_limiter.acquire()

            # Call the AI model
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if not response or not hasattr(response, 'text') or not response.text:
                    logger.warning(Fore.YELLOW + f"# No valid response from AI on attempt {attempt + 1}." + Style.RESET_ALL)
                    continue
            except Exception as e:
                logger.warning(Fore.YELLOW + f"# API error on attempt {attempt + 1}: {str(e)}" + Style.RESET_ALL)
                continue

            # Extract enhanced code
            candidate_code = response.text.strip()
            if candidate_code.startswith('```python') and candidate_code.endswith('```'):
                candidate_code = candidate_code[10:-3].strip()
            else:
                logger.warning(Fore.YELLOW + f"# Response not properly formatted on attempt {attempt + 1}." + Style.RESET_ALL)
                continue

            # Validate syntax
            is_valid, error_message = validate_syntax(candidate_code, file_path)
            if is_valid:
                enhanced_code = candidate_code
                break
            else:
                logger.error(Fore.RED + f"# Syntax error on attempt {attempt + 1}: {error_message}" + Style.RESET_ALL)
                if attempt < max_retries - 1:
                    logger.info(Fore.YELLOW + f"# Retrying enhancement for {file_path}..." + Style.RESET_ALL)

        if enhanced_code is None:
            logger.error(Fore.RED + f"# Failed to produce valid code for {file_path} after {max_retries} attempts." + Style.RESET_ALL)
            return False

        # Compute and log diff
        diff = compute_diff(original_code, enhanced_code)
        logger.info(Fore.CYAN + f"# Diff for {file_path}:\n{diff}" + Style.RESET_ALL)

        if dry_run:
            logger.info(Fore.YELLOW + f"# Dry run: Changes not applied to {file_path}." + Style.RESET_ALL)
            return True

        # Create backup
        backup_path = create_backup(file_path, original_code, ".enhancement_backups")

        # Write enhanced code
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_code)

        logger.info(Fore.GREEN + f"Successfully enhanced {file_path}." + Style.RESET_ALL)
        logger.info(Fore.YELLOW + f"# Original length: {len(original_code)} chars, Enhanced length: {len(enhanced_code)} chars" + Style.RESET_ALL)
        return True

    except Exception as e:
        logger.error(Fore.RED + f"Error enhancing {file_path}: {str(e)}" + Style.RESET_ALL)
        return False

async def main():
    """Main function to parse arguments and enhance files."""
    parser = argparse.ArgumentParser(description="Pyrmethus Advanced Code Enhancer")
    parser.add_argument("file_path", help="Path to the Python file to enhance")
    parser.add_argument("--model", default="gemini-2.5-pro-exp-03-25", help="AI model to use")
    parser.add_argument("--max-calls", type=int, default=59, help="Max API calls per minute (1-60)")
    parser.add_argument("--mode", choices=["readability", "performance", "type_hints", "comprehensive"], default="comprehensive", help="Enhancement mode")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    args = parser.parse_args()

    logger.info(Fore.MAGENTA + "# Pyrmethus Advanced Code Enhancer Initialized" + Style.RESET_ALL)

    # Load configuration
    config = load_config(args.config)
    model_name = args.model or config["model"]
    max_calls = args.max_calls or config["max_calls_per_minute"]
    mode = args.mode or config["enhancement_mode"]
    max_retries = config.get("max_retries", 3)

    # Validate max_calls
    if not 1 <= max_calls <= 60:
        logger.error(Fore.RED + f"Error: max-calls must be between 1 and 60, got {max_calls}." + Style.RESET_ALL)
        sys.exit(1)

    # Configure API
    api_key = os.getenv("GOOGLE_API_KEY")
    configure_api(api_key)

    # Initialize rate limiter
    rate_limiter = RateLimiter(max_calls)

    # Enhance file with progress feedback
    with tqdm(total=1, desc="Enhancing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        success = await enhance_code(args.file_path, model_name, mode, rate_limiter, max_retries, args.dry_run)
        pbar.update(1)

    if success:
        logger.info(Fore.GREEN + "# Enhancement spell completed successfully." + Style.RESET_ALL)
        sys.exit(0)
    else:
        logger.error(Fore.RED + "# Enhancement spell failed." + Style.RESET_ALL)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

if [ $? -eq 0 ]; then
    log_message "INFO" "Successfully created xfix_files.py" "$GREEN"
else
    log_message "ERROR" "Failed to create xfix_files.py. Exiting." "$RED"
    exit 1
fi

# Make xfix_files.py executable
log_message "INFO" "Making xfix_files.py executable..." "$YELLOW"
chmod +x "$XFIX_FILE" || {
    log_message "ERROR" "Failed to make xfix_files.py executable. Exiting." "$RED"
    exit 1
}

# Write pyrmethus-code-fixer.yml
WORKFLOW_FILE="$REPO_DIR/.github/workflows/pyrmethus-code-fixer.yml"
log_message "INFO" "Writing pyrmethus-code-fixer.yml to $WORKFLOW_FILE..." "$BLUE"
cat > "$WORKFLOW_FILE" << 'EOF'
name: Pyrmethus Code Fixer Workflow

on:
  workflow_dispatch:
    inputs:
      base_directory:
        description: 'Base directory to search for Python files (e.g., ., src, volume).'
        required: true
        default: '.'
      file_pattern:
        description: 'Glob pattern for Python files (e.g., *.py, **/*.py, volume/**/*.py).'
        required: true
        default: '**/*.py'
      enhancement_script_path:
        description: 'Path to xfix_files.py (e.g., ./xfix_files.py).'
        required: true
        default: './xfix_files.py'
      enhancement_script_args:
        description: 'Arguments for xfix_files.py (e.g., --model gemini-2.5-pro-exp-03-25 --mode comprehensive).'
        required: false
        default: ''
      google_api_key:
        description: 'Google API Key (leave blank to use secret GOOGLE_API_KEY).'
        required: false
      max_api_calls_per_minute:
        description: 'Max API calls per minute (1-60).'
        required: false
        default: '59'
      batch_size:
        description: 'Number of files per batch.'
        required: false
        default: '10'
      commit_message:
        description: 'Commit message for fixes.'
        required: false
        default: 'Apply automated code fixes and enhancements via Pyrmethus'
      debug_mode:
        description: 'Enable debug mode to log detailed syntax errors (true/false).'
        required: false
        default: 'true'
      target_branch:
        description: 'Branch to commit fixes to.'
        required: true
        default: 'main'

jobs:
  fix_python_code:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    outputs:
      files_processed_count: ${{ steps.run-enhancement.outputs.processed_count }}
      files_failed_enhancement_count: ${{ steps.run-enhancement.outputs.failed_enhancement_count }}
      syntax_error_count: ${{ steps.validate-enhanced-files.outputs.syntax_error_count }}

    steps:
      - name: Initialize Logging
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Pyrmethus Code Fixer Initiated" > enhancement_log.txt
          echo "Workflow Run ID: ${{ github.run_id }}" >> enhancement_log.txt

      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          fetch-depth: 0
          ref: ${{ inputs.target_branch }}

      - name: Clean Working Directory
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Cleaning working directory..." >> enhancement_log.txt
          git reset --hard HEAD
          git clean -fd
          echo "Working directory cleaned." >> enhancement_log.txt
          git status >> enhancement_log.txt

      - name: Sync Target Branch
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Syncing with remote ${{ inputs.target_branch }}..." >> enhancement_log.txt
          git status >> enhancement_log.txt
          if ! git pull origin ${{ inputs.target_branch }} --rebase; then
            echo "Error: Failed to sync with remote ${{ inputs.target_branch }}." >> enhancement_log.txt
            git status >> enhancement_log.txt
            exit 1
          fi
          echo "Synced with remote ${{ inputs.target_branch }}." >> enhancement_log.txt

      - name: Debug Repository Contents
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Listing repository contents..." >> enhancement_log.txt
          ls -R >> enhancement_log.txt
          if [ -f "${{ inputs.enhancement_script_path }}" ]; then
            echo "Script found at '${{ inputs.enhancement_script_path }}'." >> enhancement_log.txt
          else
            echo "Error: Script not found at '${{ inputs.enhancement_script_path }}'." >> enhancement_log.txt
            ls -R >> enhancement_log.txt
            exit 1
          fi

      - name: Validate Inputs
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Validating inputs..." >> enhancement_log.txt
          if [[ ! "${{ inputs.batch_size }}" =~ ^[0-9]+$ ]] || [ "${{ inputs.batch_size }}" -lt 1 ]; then
            echo "Error: batch_size must be a positive integer, got '${{ inputs.batch_size }}'." | tee -a enhancement_log.txt
            exit 1
          fi
          if [[ ! "${{ inputs.max_api_calls_per_minute }}" =~ ^[0-9]+$ ]] || [ "${{ inputs.max_api_calls_per_minute }}" -lt 1 ] || [ "${{ inputs.max_api_calls_per_minute }}" -gt 60 ]; then
            echo "Error: max_api_calls_per_minute must be between 1 and 60, got '${{ inputs.max_api_calls_per_minute }}'." | tee -a enhancement_log.txt
            exit 1
          fi
          if [ ! -f "${{ inputs.enhancement_script_path }}" ]; then
            echo "Error: Enhancement script '${{ inputs.enhancement_script_path }}' not found." | tee -a enhancement_log.txt
            ls -R >> enhancement_log.txt
            exit 1
          fi
          echo "Input validation completed successfully." >> enhancement_log.txt

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Installing dependencies..." >> enhancement_log.txt
          python -m pip install --upgrade pip
          pip install google-generativeai aiohttp colorama tqdm diff-match-patch
          pip list >> enhancement_log.txt

      - name: Make Enhancement Script Executable
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Setting executable permissions for ${{ inputs.enhancement_script_path }}..." >> enhancement_log.txt
          chmod +x "${{ inputs.enhancement_script_path }}"
          echo "Script is executable." >> enhancement_log.txt

      - name: List Matched Python Files
        id: list_files
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Listing Python files in '${{ inputs.base_directory }}' with pattern '${{ inputs.file_pattern }}'..." >> enhancement_log.txt
          normalized_base_dir=$(echo "${{ inputs.base_directory }}" | sed 's:/*$::')
          if [ "$normalized_base_dir" == "." ] || [ -z "$normalized_base_dir" ]; then
            full_pattern="${{ inputs.file_pattern }}"
          else
            full_pattern="${normalized_base_dir}/${{ inputs.file_pattern }}"
          fi
          full_pattern=$(echo "$full_pattern" | sed 's#^\./##' | sed 's#//\+#/#g')
          echo "Effective pattern: '$full_pattern'" | tee -a enhancement_log.txt matched_files.txt

          git ls-files -- "$full_pattern" | grep '\.py$' | sort > matched_files_list.txt
          file_count=$(wc -l < matched_files_list.txt)
          if [ "$file_count" -eq 0 ]; then
            echo "Warning: No Python files matched the pattern '$full_pattern'." | tee -a enhancement_log.txt matched_files.txt
          else
            echo "Found $file_count Python file(s)." | tee -a enhancement_log.txt matched_files.txt
            cat matched_files_list.txt >> matched_files.txt
          fi
          echo "file_count=$file_count" >> "$GITHUB_OUTPUT"

      - name: Prepare Files for Batching
        id: prepare_batching
        if: steps.list_files.outputs.file_count > 0
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Preparing files for batching..." >> enhancement_log.txt
          mv matched_files_list.txt files_to_process.txt
          if [ ! -s files_to_process.txt ]; then
            echo "No files available for processing." | tee -a enhancement_log.txt
            echo "files_available_for_batching=false" >> $GITHUB_OUTPUT
          else
            echo "Files prepared for batching." | tee -a enhancement_log.txt
            echo "files_available_for_batching=true" >> $GITHUB_OUTPUT
          fi

      - name: Split Files into Batches
        if: steps.prepare_batching.outputs.files_available_for_batching == 'true'
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Splitting files into batches of ${{ inputs.batch_size }}..." >> enhancement_log.txt
          split -l ${{ inputs.batch_size }} files_to_process.txt batch_
          ls batch_* > batch_list.txt
          batch_count=$(wc -l < batch_list.txt)
          echo "Created $batch_count batches." | tee -a enhancement_log.txt

      - name: Run Code Enhancement Script
        id: run-enhancement
        if: steps.prepare_batching.outputs.files_available_for_batching == 'true'
        env:
          GOOGLE_API_KEY: ${{ inputs.google_api_key || secrets.GOOGLE_API_KEY }}
          MAX_API_CALLS_PER_MINUTE: ${{ inputs.max_api_calls_per_minute }}
          ENHANCEMENT_SCRIPT_ARGS: ${{ inputs.enhancement_script_args }}
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Running enhancement script on batches..." >> enhancement_log.txt
          if [ -z "$GOOGLE_API_KEY" ]; then
            echo "Error: GOOGLE_API_KEY is not set." | tee -a enhancement_log.txt
            exit 1
          fi
          processed_count=0
          failed_enhancement_count=0
          > failed_enhancement_files.txt

          while read -r batch_file; do
            if [ ! -s "$batch_file" ]; then
              echo "Skipping empty batch: $batch_file" | tee -a enhancement_log.txt
              continue
            fi
            echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Processing batch: $batch_file" >> enhancement_log.txt
            while IFS= read -r file_path || [[ -n "$file_path" ]]; do
              if [ -z "$file_path" ]; then continue; fi
              echo "Enhancing: $file_path" | tee -a enhancement_log.txt
              if ! python3 "${{ inputs.enhancement_script_path }}" "$file_path" $ENHANCEMENT_SCRIPT_ARGS >> enhancement_log.txt 2>&1; then
                echo "Failed to enhance: $file_path" | tee -a enhancement_log.txt
                echo "$file_path" >> failed_enhancement_files.txt
                failed_enhancement_count=$((failed_enhancement_count + 1))
              else
                processed_count=$((processed_count + 1))
              fi
            done < "$batch_file"
          done < batch_list.txt

          echo "Processed $processed_count files, $failed_enhancement_count failed." | tee -a enhancement_log.txt
          echo "processed_count=$processed_count" >> $GITHUB_OUTPUT
          echo "failed_enhancement_count=$failed_enhancement_count" >> $GITHUB_OUTPUT

      - name: Validate Enhanced Files
        id: validate-enhanced-files
        if: steps.prepare_batching.outputs.files_available_for_batching == 'true' && steps.run-enhancement.outputs.processed_count > 0
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Validating Python files for syntax errors..." >> enhancement_log.txt
          syntax_error_count=0
          > syntax_error_files.txt
          while IFS= read -r file_path || [[ -n "$file_path" ]]; do
            if [ -z "$file_path" ]; then continue; fi
            if [ -f "$file_path" ]; then
              is_failed=$(grep -Fx "$file_path" failed_enhancement_files.txt && echo "true" || echo "false")
              echo "Validating: $file_path (Enhancement failed: $is_failed)" | tee -a enhancement_log.txt
              if ! python3 -m py_compile "$file_path" >> syntax_validation.log 2>&1; then
                echo "Syntax error in: $file_path" | tee -a enhancement_log.txt
                echo "$file_path" >> syntax_error_files.txt
                if [ "${{ inputs.debug_mode }}" == "true" ]; then
                  echo "Error details for $file_path:" >> enhancement_log.txt
                  cat syntax_validation.log >> enhancement_log.txt
                fi
                syntax_error_count=$((syntax_error_count + 1))
              fi
            else
              echo "Warning: File $file_path not found." | tee -a enhancement_log.txt
            fi
          done < files_to_process.txt

          echo "syntax_error_count=$syntax_error_count" >> $GITHUB_OUTPUT
          if [ "$syntax_error_count" -gt 0 ]; then
            echo "Warning: $syntax_error_count syntax errors found. Proceeding with valid files." | tee -a enhancement_log.txt
          else
            echo "No syntax errors detected." | tee -a enhancement_log.txt
          fi

      - name: Check for Changes
        id: git_status
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Checking for changes..." >> enhancement_log.txt
          git add .
          git status >> enhancement_log.txt
          if ! git diff --cached --quiet; then
            echo "Changes detected." | tee -a enhancement_log.txt
            echo "changes_made=true" >> "$GITHUB_OUTPUT"
          else
            echo "No changes to commit." | tee -a enhancement_log.txt
            echo "changes_made=false" >> "$GITHUB_OUTPUT"
          fi

      - name: Configure Git User
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Configuring Git user..." >> enhancement_log.txt
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Commit Changes
        if: steps.git_status.outputs.changes_made == 'true'
        id: commit-changes
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Committing changes..." >> enhancement_log.txt
          git commit -m "${{ inputs.commit_message }}"
          echo "Committed changes to ${{ inputs.target_branch }}." >> enhancement_log.txt

      - name: Push Changes
        if: steps.git_status.outputs.changes_made == 'true'
        run: |
          echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') - Pushing changes to ${{ inputs.target_branch }}..." >> enhancement_log.txt
          git push origin "${{ inputs.target_branch }}"
          echo "Changes pushed." >> enhancement_log.txt

      - name: Upload Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: pyrmethus-artifacts-${{ github.run_id }}
          path: |
            matched_files.txt
            enhancement_log.txt
            files_to_process.txt
            batch_*
            batch_list.txt
            failed_enhancement_files.txt
            syntax_error_files.txt
            syntax_validation.log
          if-no-files-found: ignore
          retention-days: 7
EOF

if [ $? -eq 0 ]; then
    log_message "INFO" "Successfully created pyrmethus-code-fixer.yml" "$GREEN"
else
    log_message "ERROR" "Failed to create pyrmethus-code-fixer.yml. Exiting." "$RED"
    exit 1
fi

# Initialize Git repository if not already initialized
if [ ! -d "$REPO_DIR/.git" ]; then
    log_message "INFO" "Initializing Git repository..." "$YELLOW"
    git init "$REPO_DIR"
    git checkout -b main
    log_message "INFO" "Git repository initialized on main branch" "$GREEN"
fi

# Configure Git user if not set
log_message "INFO" "Configuring Git user..." "$CYAN"
git config user.name "Termux User" || {
    log_message "ERROR" "Failed to configure Git user name. Exiting." "$RED"
    exit 1
}
git config user.email "termux.user@example.com" || {
    log_message "ERROR" "Failed to configure Git user email. Exiting." "$RED"
    exit 1
}

# Check for remote repository
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
    log_message "WARNING" "No remote repository set. Please provide the remote URL (e.g., https://github.com/username/repo.git):" "$YELLOW"
    read -p "Remote URL: " REMOTE_URL
    if [ -z "$REMOTE_URL" ]; then
        log_message "ERROR" "No remote URL provided. Exiting." "$RED"
        exit 1
    fi
    git remote add origin "$REMOTE_URL" || {
        log_message "ERROR" "Failed to add remote origin. Exiting." "$RED"
        exit 1
    }
    log_message "INFO" "Added remote origin: $REMOTE_URL" "$GREEN"
else
    log_message "INFO" "Using existing remote: $REMOTE_URL" "$BLUE"
fi

# Stage files
log_message "INFO" "Staging files for commit..." "$CYAN"
git add "$XFIX_FILE" "$WORKFLOW_FILE" || {
    log_message "ERROR" "Failed to stage files. Exiting." "$RED"
    exit 1
}

# Commit files
log_message "INFO" "Committing files..." "$YELLOW"
git commit -m "Add Pyrmethus code fixer script and workflow" || {
    log_message "ERROR" "Failed to commit files. Exiting." "$RED"
    exit 1
}
log_message "INFO" "Files committed successfully" "$GREEN"

# Push to main branch
log_message "INFO" "Pushing to main branch..." "$CYAN"
if ! git push origin main; then
    log_message "INFO" "Initial push failed, trying to set upstream..." "$YELLOW"
    git push --set-upstream origin main || {
        log_message "ERROR" "Failed to push to main branch. Ensure remote is accessible and credentials are set. Exiting." "$RED"
        exit 1
    }
fi
log_message "INFO" "Successfully pushed to main branch" "$GREEN"

# Final message
log_message "INFO" "Pyrmethus setup complete. Files created and committed." "$MAGENTA"
log_message "INFO" "Check $LOG_FILE for details." "$BLUE"
echo -e "${GREEN}Spell successfully cast! Your repository is ready for code enhancement.${RESET}"
exit 0
