#!/data/data/com.termux/files/usr/bin/bash

# resolve_git_changes.sh: Resolve unstaged changes and sync Pyrmethus repository in Termux

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

# Configure Git user
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
    read18read -p "Remote URL: " REMOTE_URL
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

# Update .gitignore to exclude logs and temp files
log_message "INFO" "Updating .gitignore to exclude logs..." "$CYAN"
cat > .gitignore << 'EOF'
# Pyrmethus logs and temp files
git_resolve.log
git_status.txt
pyrmethus_setup.log
enhancement_log.txt
syntax_validation.log
*.bak
# Other common ignores
*.o
*.swp
*.pyc
__pycache__/
EOF
git add .gitignore
git commit -m "Update .gitignore to exclude Pyrmethus logs and temp files" --allow-empty || {
    log_message "WARNING" "No changes in .gitignore to commit." "$YELLOW"
}
# Refresh Git index
git update-index --refresh
log_message "INFO" "Git index refreshed." "$CYAN"

# Check Git status
log_message "INFO" "Checking Git status..." "$CYAN"
git status --short > git_status.txt
if [ -s git_status.txt ]; then
    log_message "WARNING" "Unstaged or untracked changes detected:" "$YELLOW"
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
        git stash push -m "Pyrmethus setup stash" --include-untracked || {
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

# Clean working directory
log_message "INFO" "Cleaning working directory to ensure no unstaged changes..." "$CYAN"
git reset --hard HEAD || {
    log_message "ERROR" "Failed to reset working directory. Exiting." "$RED"
    exit 1
}
git clean -fd || {
    log_message "ERROR" "Failed to clean untracked files. Exiting." "$RED"
    exit 1
}
# Brief delay to ensure file system sync
sleep 1
# Refresh Git index again
git update-index --refresh
log_message "INFO" "Working directory cleaned and index refreshed." "$GREEN"

# Verify no unstaged changes
log_message "INFO" "Verifying repository state..." "$CYAN"
git status --short > git_status.txt
if [ -s git_status.txt ]; then
    log_message "ERROR" "Unstaged changes remain after cleaning:" "$RED"
    cat git_status.txt >> "$LOG_FILE"
    log_message "ERROR" "Detailed Git status:" "$RED"
    git status >> "$LOG_FILE"
    exit 1
fi
log_message "INFO" "Repository is clean. No unstaged changes." "$GREEN"

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

# Stage Pyrmethus files
log_message "INFO" "Staging Pyrmethus files..." "$CYAN"
[ -f "$XFIX_FILE" ] && git add "$XFIX_FILE"
[ -f "$WORKFLOW_FILE" ] && git add "$WORKFLOW_FILE"

# Commit if staged changes exist
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

# Pull with rebase (retry up to 3 times)
log_message "INFO" "Syncing with remote main..." "$CYAN"
for attempt in {1..3}; do
    log_message "INFO" "Pull attempt $attempt/3..." "$CYAN"
    if git pull origin main --rebase; then
        log_message "INFO" "Repository synced with remote main." "$GREEN"
        break
    else
        log_message "WARNING" "Pull failed on attempt $attempt. Resetting and retrying..." "$YELLOW"
        git rebase --abort 2>/dev/null || true
        git reset --hard HEAD
        git clean -fd
        git update-index --refresh
        if [ $attempt -eq 3 ]; then
            log_message "ERROR" "Failed to pull with rebase after 3 attempts. Resolve conflicts manually and run 'git rebase --continue'." "$RED"
            git status >> "$LOG_FILE"
            exit 1
        fi
        sleep 1
    fi
done

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
