#!/data/data/com.termux/files/usr/bin/bash

# resolve_git_changes.sh: Resolve unstaged changes and commit Pyrmethus files in Termux

# ANSI color codes for vibrant output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

# Log file
LOG_FILE="git_resolve.log"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    local color="$3"
    echo -e "${color}[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] [$level] $message${RESET}" | tee -a "$LOG_FILE"
}

# Initialize log
echo -e "${MAGENTA}# Pyrmethus Git Resolution Spell Initiated${RESET}" | tee "$LOG_FILE"
log_message "INFO" "Starting Git changes resolution in Termux" "$CYAN"

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

# Repository directory
REPO_DIR="$PWD"
log_message "INFO" "Using repository directory: $REPO_DIR" "$BLUE"

# Check if in a Git repository
if [ ! -d "$REPO_DIR/.git" ]; then
    log_message "ERROR" "Not a Git repository. Initialize with 'git init' or clone first. Exiting." "$RED"
    exit 1
fi

# Configure Git user if not set
log_message "INFO" "Configuring Git user..." "$CYAN"
git config user.name >/dev/null || git config user.name "Termux User" || {
    log_message "ERROR" "Failed to configure Git user name. Exiting." "$RED"
    exit 1
}
git config user.email >/dev/null || git config user.email "termux.user@example.com" || {
    log_message "ERROR" "Failed to configure Git user email. Exiting." "$RED"
    exit 1
}

# Check for remote
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

# Check Git status
log_message "INFO" "Checking Git status..." "$CYAN"
git status --short > git_status.txt
if [ -s git_status.txt ]; then
    log_message "WARNING" "Unstaged changes detected:" "$YELLOW"
    cat git_status.txt | while read -r line; do
        log_message "WARNING" "  $line" "$YELLOW"
    done
else
    log_message "INFO" "No unstaged changes. Repository is clean." "$GREEN"
fi

# Prompt user for action
log_message "INFO" "Choose action for unstaged changes (select a number):" "$BLUE"
echo -e "${BLUE}1) Commit changes (recommended for Pyrmethus files)${RESET}"
echo -e "${BLUE}2) Stash changes (save for later)${RESET}"
echo -e "${BLUE}3) Reset changes (discard all unstaged changes)${RESET}"
echo -e "${BLUE}4) Exit without changes${RESET}"
read -p "Enter choice (1-4): " choice

case "$choice" in
    1)
        log_message "INFO" "Committing unstaged changes..." "$YELLOW"
        git add . || {
            log_message "ERROR" "Failed to stage changes. Exiting." "$RED"
            exit 1
        }
        git commit -m "Commit unstaged changes for Pyrmethus setup" || {
            log_message "ERROR" "Failed to commit changes. Exiting." "$RED"
            exit 1
        }
        log_message "INFO" "Changes committed successfully." "$GREEN"
        ;;
    2)
        log_message "INFO" "Stashing unstaged changes..." "$YELLOW"
        git stash push -m "Pyrmethus setup stash" || {
            log_message "ERROR" "Failed to stash changes. Exiting." "$RED"
            exit 1
        }
        log_message "INFO" "Changes stashed. Run 'git stash pop' to restore later." "$GREEN"
        ;;
    3)
        log_message "INFO" "Resetting unstaged changes..." "$YELLOW"
        git reset --hard HEAD || {
            log_message "ERROR" "Failed to reset changes. Exiting." "$RED"
            exit 1
        }
        git clean -fd || {
            log_message "ERROR" "Failed to clean untracked files. Exiting." "$RED"
            exit 1
        }
        log_message "INFO" "Repository reset to clean state." "$GREEN"
        ;;
    4)
        log_message "INFO" "Exiting without modifying changes." "$BLUE"
        exit 0
        ;;
    *)
        log_message "ERROR" "Invalid choice. Exiting." "$RED"
        exit 1
        ;;
esac

# Verify Pyrmethus files
XFIX_FILE="$REPO_DIR/xfix_files.py"
WORKFLOW_FILE="$REPO_DIR/.github/workflows/pyrmethus-code-fixer.yml"
log_message "INFO" "Verifying Pyrmethus files..." "$CYAN"

if [ -f "$XFIX_FILE" ]; then
    log_message "INFO" "xfix_files.py found." "$GREEN"
else
    log_message "WARNING" "xfix_files.py not found. Ensure it’s created by setup_pyrmethus.sh." "$YELLOW"
fi

if [ -f "$WORKFLOW_FILE" ]; then
    log_message "INFO" "pyrmethus-code-fixer.yml found." "$GREEN"
else
    log_message "WARNING" "pyrmethus-code-fixer.yml not found. Ensure it’s created by setup_pyrmethus.sh." "$YELLOW"
fi

# Stage Pyrmethus files (if present)
log_message "INFO" "Staging Pyrmethus files..." "$CYAN"
[ -f "$XFIX_FILE" ] && git add "$XFIX_FILE"
[ -f "$WORKFLOW_FILE" ] && git add "$WORKFLOW_FILE"

# Commit if there are staged changes
if ! git diff --cached --quiet; then
    log_message "INFO" "Committing Pyrmethus files..." "$YELLOW"
    git commit -m "Ensure Pyrmethus files for code enhancement workflow" || {
        log_message "ERROR" "Failed to commit Pyrmethus files. Exiting." "$RED"
        exit 1
    }
    log_message "INFO" "Pyrmethus files committed successfully." "$GREEN"
else
    log_message "INFO" "No new changes to commit." "$BLUE"
fi

# Pull with rebase to sync
log_message "INFO" "Syncing with remote main..." "$CYAN"
if ! git pull origin main --rebase; then
    log_message "ERROR" "Failed to pull with rebase. Resolve conflicts manually and try again. Exiting." "$RED"
    git status >> "$LOG_FILE"
    exit 1
fi
log_message "INFO" "Repository synced with remote main." "$GREEN"

# Push to main
log_message "INFO" "Pushing to main branch..." "$CYAN"
if ! git push origin main; then
    log_message "INFO" "Initial push failed, trying to set upstream..." "$YELLOW"
    git push --set-upstream origin main || {
        log_message "ERROR" "Failed to push to main. Ensure remote is accessible and credentials are set. Exiting." "$RED"
        exit 1
    }
fi
log_message "INFO" "Successfully pushed to main branch." "$GREEN"

# Clean up
rm -f git_status.txt
log_message "INFO" "Pyrmethus Git resolution complete. Repository is clean and synced." "$MAGENTA"
log_message "INFO" "Check $LOG_FILE for details." "$BLUE"
echo -e "${GREEN}Spell successfully cast! Your repository is ready for the Pyrmethus workflow.${RESET}"
exit 0
