#!/data/data/com.termux/files/usr/bin/bash

# Colors for mystical terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

# Logging setup
LOG_DIR="$HOME/.termux-git-logs"
LOG_FILE="$LOG_DIR/merge_latest_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"
echo -e "${CYAN}# Forging the Git spirits' chronicle at $LOG_FILE...${RESET}" | tee -a "$LOG_FILE"

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to check Termux storage
check_storage() {
    local available=$(df -k "$HOME" | awk 'NR==2 {print $4}')
    if [ "$available" -lt 102400 ]; then
        log "${RED}Low storage (${available}KB available)! Free up space.${RESET}"
        termux-toast -b red "Low storage!"
        exit 1
    fi
}

# Function to detect repository platform
detect_platform() {
    local remote_url=$(git remote get-url origin 2>/dev/null)
    if echo "$remote_url" | grep -qi "github.com"; then
        echo "github"
    elif echo "$remote_url" | grep -qi "gitlab.com"; then
        echo "gitlab"
    else
        echo "unknown"
    fi
}

# Function to retry git push
retry_push() {
    local branch=$1
    local attempts=3
    local delay=2
    for ((i=1; i<=attempts; i++)); do
        log "${CYAN}# Attempt $i: Sending essence to origin...${RESET}"
        git push origin "$branch" 2>>"$LOG_FILE"
        if [ $? -eq 0 ]; then
            return 0
        fi
        log "${YELLOW}Push attempt $i failed, retrying in ${delay}s...${RESET}"
        sleep $delay
        delay=$((delay * 2))
    done
    log "${RED}Push failed after $attempts attempts!${RESET}"
    termux-toast -b red "Push failed!"
    return 1
}

log "${CYAN}# Summoning the Git spirits to merge the latest pull request...${RESET}"

# Ensure git is installed
if ! command -v git &> /dev/null; then
    log "${RED}Git not found! Install it with: pkg install git${RESET}"
    termux-toast -b red "Git not installed!"
    exit 1
fi

# Check storage
check_storage

# Check if inside a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    log "${RED}Not a git repository! Initialize one first.${RESET}"
    termux-toast -b red "Not a git repository!"
    exit 1
fi

# Verify remote exists
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$REMOTE_URL" ]; then
    log "${RED}No valid remote 'origin' found! Set it with: git remote add origin <url>${RESET}"
    termux-toast -b red "No valid remote 'origin'!"
    exit 1
fi
log "${YELLOW}Remote origin: $REMOTE_URL${RESET}"

# Stage all changes
log "${CYAN}# Forcing the essence of changes into the repository...${RESET}"
git add -A 2>>"$LOG_FILE"
if [ $? -ne 0 ]; then
    log "${RED}Failed to stage changes!${RESET}"
    termux-toast -b red "Failed to stage changes!"
    exit 1
fi

# Commit with a forceful message
git commit -m "Forceful commit by Pyrmethus' will" --no-verify 2>>"$LOG_FILE"
if [ $? -ne 0 ]; then
    log "${YELLOW}Nothing to commit, the ether is calm...${RESET}"
fi

# Switch to main branch
MAIN_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
git checkout "$MAIN_BRANCH" 2>>"$LOG_FILE"
if [ $? -ne 0 ]; then
    log "${RED}Failed to switch to $MAIN_BRANCH!${RESET}"
    termux-toast -b red "Failed to switch to $MAIN_BRANCH!"
    exit 1
fi

# Create backup branch
BACKUP_BRANCH="backup-$(date +%Y%m%d_%H%M%S)"
log "${CYAN}# Crafting a backup branch: $BACKUP_BRANCH...${RESET}"
git branch "$BACKUP_BRANCH" 2>>"$LOG_FILE"
if [ $? -eq 0 ]; then
    log "${GREEN}Backup created at $BACKUP_BRANCH${RESET}"
    termux-toast -b green "Backup: $BACKUP_BRANCH"
else
    log "${YELLOW}Failed to create backup branch, proceeding...${RESET}"
fi

# Detect platform
PLATFORM=$(detect_platform)
log "${YELLOW}Detected platform: $PLATFORM${RESET}"

# Fetch the most recent pull request
if [ "$PLATFORM" = "github" ]; then
    PR_REF="refs/pull/*/head"
    PR_REGEX="refs/pull/[0-9]+/head"
elif [ "$PLATFORM" = "gitlab" ]; then
    PR_REF="refs/merge-requests/*/head"
    PR_REGEX="refs/merge-requests/[0-9]+/head"
else
    log "${RED}Unsupported platform! Assuming GitHub-style refs...${RESET}"
    PR_REF="refs/pull/*/head"
    PR_REGEX="refs/pull/[0-9]+/head"
fi

log "${CYAN}# Seeking the most recent pull request...${RESET}"
PR_NUMBER=$(git ls-remote origin "$PR_REF" | grep -oE "$PR_REGEX" | sort -t'/' -k3 -nr | head -n1 | grep -oE '[0-9]+')
if [ -z "$PR_NUMBER" ]; then
    log "${RED}No pull requests found or repository does not support $PR_REF${RESET}"
    log "${YELLOW}Ensure this is a GitHub/GitLab repository.${RESET}"
    termux-toast -b red "No pull requests found!"
    exit 1
fi
log "${GREEN}Found most recent pull request #$PR_NUMBER${RESET}"
termux-toast -b green "Found PR #$PR_NUMBER"

# Verify PR exists
if [ "$PLATFORM" = "github" ]; then
    PR_HEAD="refs/pull/$PR_NUMBER/head"
else
    PR_HEAD="refs/merge-requests/$PR_NUMBER/head"
fi
if ! git ls-remote origin "$PR_HEAD" &>/dev/null; then
    log "${RED}Pull request #$PR_NUMBER does not exist or is inaccessible!${RESET}"
    termux-toast -b red "PR #$PR_NUMBER not found!"
    exit 1
fi

# Fetch the pull request
git fetch origin "$PR_HEAD:pr-$PR_NUMBER" 2>>"$LOG_FILE"
if [ $? -ne 0 ]; then
    log "${RED}Failed to fetch pull request #$PR_NUMBER!${RESET}"
    termux-toast -b red "Failed to fetch PR #$PR_NUMBER!"
    exit 1
fi

# Attempt merge with strategy to favor PR changes
log "${CYAN}# Merging PR #$PR_NUMBER with the power to overwrite conflicts...${RESET}"
git merge pr-$PR_NUMBER --strategy-option theirs -m "Force merge PR #$PR_NUMBER by Pyrmethus" --no-commit 2>>"$LOG_FILE"

# Check for conflicts
if git status --porcelain | grep -E '^(DD|AU|UD|UA|DU|AA|UU)' >/dev/null; then
    log "${YELLOW}# Conflicts detected, resolving by favoring PR #$PR_NUMBER...${RESET}"
    
    # Resolve conflicts: prioritize PR's deletions and changes
    git status --porcelain | grep -E '^(DD|AU|UD|UA|DU|AA|UU)' | awk '{print $2}' | sort -u | while read -r file; do
        if ! git ls-tree pr-$PR_NUMBER -- "$file" >/dev/null; then
            log "${YELLOW}Removing $file (deleted in PR #$PR_NUMBER)${RESET}"
            git rm --force "$file" 2>>"$LOG_FILE"
        else
            log "${YELLOW}Accepting PR #$PR_NUMBER's version of $file${RESET}"
            git checkout --theirs -- "$file" 2>>"$LOG_FILE"
            git add "$file" 2>>"$LOG_FILE"
        fi
    done

    # Handle additional deletions in PR
    git diff --name-status pr-$PR_NUMBER | grep '^D' | awk '{print $2}' | while read -r file; do
        log "${YELLOW}Removing $file (deleted in PR #$PR_NUMBER)${RESET}"
        git rm --force "$file" 2>>"$LOG_FILE"
    done

    # Commit resolved merge
    git commit -m "Force merge PR #$PR_NUMBER by Pyrmethus (conflicts resolved)" --no-verify 2>>"$LOG_FILE"
    if [ $? -ne 0 ]; then
        log "${RED}Failed to commit resolved merge for PR #$PR_NUMBER!${RESET}"
        termux-toast -b red "Failed to commit PR #$PR_NUMBER!"
        git merge --abort 2>>"$LOG_FILE"
        git branch -D pr-$PR_NUMBER 2>>"$LOG_FILE"
        exit 1
    fi
else
    # No conflicts, finalize merge
    git commit --no-edit 2>>"$LOG_FILE"
fi

# Push changes
if ! retry_push "$MAIN_BRANCH"; then
    git branch -D pr-$PR_NUMBER 2>>"$LOG_FILE"
    exit 1
fi

# Clean up temporary branch
git branch -D pr-$PR_NUMBER 2>>"$LOG_FILE"
log "${GREEN}# PR #$PR_NUMBER merged successfully.${RESET}"
termux-toast -b green "Merged PR #$PR_NUMBER!"

log "${GREEN}# Incantation complete. The repository is aligned with PR #$PR_NUMBER.${RESET}"
termux-toast -b green "Merge complete!"
