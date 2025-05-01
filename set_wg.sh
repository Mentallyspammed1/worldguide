#!/data/data/com.termux/files/usr/bin/bash

# === Pyrmethus's Enhanced GitHub Repository Conjuration ===
# Purpose: Clones or updates the worldguide repository in Termux.
# Version: 2.1 (Refined status check)

# --- Strict Mode & Error Handling ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines return the exit status of the last command to exit non-zero.
set -o pipefail

# --- Arcane Colors & Logging Functions ---
RESET='\033[0m'
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'

log_info() { echo -e "${BLUE}INFO:${RESET} $1"; }
log_warn() { echo -e "${YELLOW}WARN:${RESET} $1"; }
# Errors go to stderr, ensuring they don't pollute stdout for potential scripting use
log_error() { echo -e "${RED}${BOLD}ERROR:${RESET}${RED} $1${RESET}" >&2; }
log_success() { echo -e "${GREEN}SUCCESS:${RESET} $1"; }
log_step() { echo -e "\n${CYAN}${BOLD}>>> $1${RESET}"; }

# --- Configuration ---
REPO_URL="https://github.com/Mentallyspammed1/worldguide"
# Derive directory name from URL (simple version, assumes standard GitHub URL structure)
REPO_DIR=$(basename "$REPO_URL" .git)
# Ensure REPO_DIR is not empty if basename fails unexpectedly
if [[ -z "$REPO_DIR" ]]; then
    log_error "Could not determine repository directory name from URL: ${REPO_URL}"
    exit 1
fi
TARGET_DIR="$HOME/$REPO_DIR" # Clone into home directory

# --- Helper Functions ---

check_dependency() {
    local cmd="$1"
    local pkg="$2"
    log_info "Checking for dependency: ${BOLD}${cmd}${RESET}"
    # Use 'command -v' for POSIX compatibility and checking builtins/functions too
    if ! command -v "$cmd" &> /dev/null; then
        log_warn "${BOLD}${cmd}${RESET} command not found."
        # Use printf for potentially safer prompt formatting, though echo -e is fine here
        printf "${BLUE}Attempt to install package '%s' via pkg? (y/N):${RESET} " "$pkg"
        read -r install_pkg # Use -r to prevent backslash interpretation
        if [[ "$install_pkg" =~ ^[Yy]$ ]]; then
            log_info "Installing ${pkg}..."
            # Capture output of pkg install in case of errors, though set -e handles exit code
            if pkg install -y "$pkg"; then
                log_success "Package ${pkg} installed."
                # Verify command again after installation
                if ! command -v "$cmd" &> /dev/null; then
                     log_error "Installation reported success, but command '${cmd}' still not found. Aborting."
                     exit 1
                fi
            else
                # pkg install should have already printed errors
                log_error "Failed to install package ${pkg}. Please install it manually and retry. Aborting."
                exit 1
            fi
        else
            log_error "Dependency '${cmd}' (package '${pkg}') is required. Please install manually. Aborting."
            exit 1
        fi
    else
        log_success "${BOLD}${cmd}${RESET} is available."
    fi
}

check_network() {
    log_info "Checking network connectivity to github.com..."
    # Use ping with a count of 1 and a timeout of 5 seconds. Redirect output.
    if ping -c 1 -W 5 github.com &> /dev/null; then
        log_success "Network connection to github.com seems OK."
    else
        # Exit code of ping was non-zero
        log_warn "Could not ping github.com. This might be due to network issues or ICMP being blocked."
        # Ask user if they want to continue despite potential network issues
        printf "${BLUE}Continue anyway? (Cloning/pulling might fail) (y/N):${RESET} "
        read -r continue_anyway
        if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
            log_info "Aborting due to potential network issue."
            exit 1 # User chose not to continue
        fi
        log_info "Proceeding despite ping failure..."
    fi
}

clone_repo() {
    log_info "Cloning repository from ${BOLD}${REPO_URL}${RESET} into ${BOLD}${TARGET_DIR}${RESET}..."
    # Use --depth=1 for faster initial clone (fetches only the latest commit)
    # Use --quiet/-q to reduce verbose git output unless there's an error
    if git clone --depth=1 --quiet "$REPO_URL" "$TARGET_DIR"; then
        log_success "Repository cloned successfully."
    else
        # set -e would normally exit, but explicitly catching error allows for a better message
        log_error "Failed to clone repository. Check URL, network connection, and permissions. Aborting."
        exit 1
    fi
}

update_repo() {
    log_info "Repository already exists at ${BOLD}${TARGET_DIR}${RESET}."
    # Ensure we are in a git repository directory before running git commands
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        log_error "Directory ${TARGET_DIR} exists but does not appear to be a valid git repository. Aborting update check."
        # Decide if this should be a fatal error or just skip update? Let's make it fatal for now.
        exit 1
    fi

    printf "${BLUE}Do you want to attempt to update it with 'git pull'? (y/N):${RESET} "
    read -r pull_changes
    if [[ "$pull_changes" =~ ^[Yy]$ ]]; then
        log_info "Attempting to update repository..."

        # Check for uncommitted changes using the output of git status --porcelain
        # This command outputs status lines if there are changes, and nothing if clean.
        local git_status
        git_status=$(git status --porcelain) # Capture the output

        if [[ -n "$git_status" ]]; then
             log_warn "Repository has uncommitted changes or untracked files:"
             # Optionally show the changes for more context (can be long)
             # echo "$git_status" | while IFS= read -r line; do echo -e "${YELLOW}  $line${RESET}"; done
             printf "${BLUE}Attempt 'git pull' anyway? (May cause conflicts or overwrite local work) (y/N):${RESET} "
             read -r pull_anyway
             if [[ ! "$pull_anyway" =~ ^[Yy]$ ]]; then
                  log_info "Skipping update due to local changes."
                  return 0 # Exit the function successfully, skipping the pull
             fi
             log_info "Proceeding with pull despite local changes..."
        fi

        # Fetch latest changes from the remote 'origin' for the current branch
        local current_branch
        current_branch=$(git rev-parse --abbrev-ref HEAD)
        log_info "Pulling changes for branch '${current_branch}' from origin..."
        # Use --quiet/-q to reduce verbose git output unless there's an error
        if git pull --quiet origin "$current_branch"; then
            log_success "Repository updated successfully."
        else
            # git pull failed (e.g., merge conflicts, network error)
            log_error "Failed to update repository. There might be merge conflicts or network issues. Please resolve manually in ${TARGET_DIR}"
            # Don't exit the script, just report the error. User is already in the directory context.
            return 1 # Indicate update failure functionally if needed elsewhere
        fi
    else
        log_info "Skipping update as requested."
    fi
    return 0 # Indicate update skipped or succeeded
}

display_guidance() {
    log_step "Guidance for the Seeker"
    # Use pwd -P to resolve symlinks for the canonical path if necessary
    local current_dir
    current_dir=$(pwd -P)
    log_info "The repository's essence now resides in: ${CYAN}${BOLD}${current_dir}${RESET}"
    log_info "Explore its contents. Seek scrolls like ${BOLD}README.md${RESET} or ${BOLD}README.rst${RESET} for primary instructions."

    local found_guidance=false
    # Check for common project setup/dependency files
    if [[ -f "requirements.txt" ]]; then
        log_warn "Python dependencies found (${BOLD}requirements.txt${RESET}). Consider installing them:"
        echo -e "${CYAN}  cd \"${current_dir}\"${RESET}" # Remind user to be in the correct directory
        echo -e "${CYAN}  python -m venv .venv && source .venv/bin/activate${RESET}  (Optional: Use a virtual environment)"
        echo -e "${CYAN}  pip install -r requirements.txt${RESET}"
        check_dependency "python" "python" # Check if python/pip are available
        check_dependency "pip" "python"
        found_guidance=true
    fi
    if [[ -f "package.json" ]]; then
        log_warn "Node.js dependencies found (${BOLD}package.json${RESET}). Consider installing them:"
        echo -e "${CYAN}  cd \"${current_dir}\"${RESET}"
        echo -e "${CYAN}  npm install${RESET}  (or 'yarn install' if yarn.lock exists)"
        check_dependency "node" "nodejs" # Check if node is installed if package.json exists
        check_dependency "npm" "nodejs" # Check npm too
        found_guidance=true
    fi
    if [[ -f "setup.py" ]] || [[ -f "pyproject.toml" ]]; then
        log_warn "Python package setup found (${BOLD}setup.py/pyproject.toml${RESET}). It might be installable:"
        echo -e "${CYAN}  cd \"${current_dir}\"${RESET}"
        echo -e "${CYAN}  pip install .${RESET}"
        check_dependency "pip" "python"
        found_guidance=true
    fi
    if [[ -f "Makefile" ]]; then
        log_warn "A ${BOLD}Makefile${RESET} was found. Common commands might be:"
        echo -e "${CYAN}  cd \"${current_dir}\"${RESET}"
        echo -e "${CYAN}  make help${RESET} (if available)"
        echo -e "${CYAN}  make install${RESET}"
        echo -e "${CYAN}  make build${RESET}"
        echo -e "${CYAN}  make test${RESET}"
        check_dependency "make" "make"
        found_guidance=true
    fi
    if [[ -f "docker-compose.yml" ]] || [[ -f "Dockerfile" ]]; then
         log_warn "Docker configuration found (${BOLD}Dockerfile/docker-compose.yml${RESET}). You might use Docker commands if installed on your system (less common directly in Termux)."
         # No dependency check here, as Docker setup in Termux is non-standard
         found_guidance=true
    fi
    # Add check for composer.json (PHP)
    if [[ -f "composer.json" ]]; then
        log_warn "PHP dependencies found (${BOLD}composer.json${RESET}). Consider installing them:"
        echo -e "${CYAN}  cd \"${current_dir}\"${RESET}"
        echo -e "${CYAN}  pkg install php composer${RESET} (If not installed)"
        echo -e "${CYAN}  composer install${RESET}"
        check_dependency "php" "php"
        check_dependency "composer" "composer"
        found_guidance=true
    fi

    if ! $found_guidance; then
        log_info "No common dependency or build files detected (like requirements.txt, package.json, Makefile). Consult the README or project documentation for setup instructions."
    fi
}

# --- Main Execution ---
echo -e "${CYAN}${BOLD}~~~ Pyrmethus's Enhanced GitHub Conjuration Ritual v2.1 ~~~${RESET}"

log_step "Checking Prerequisites"
check_dependency "git" "git"
# ping is usually in inetutils or termux-tools, let's assume inetutils first
check_dependency "ping" "inetutils"

log_step "Checking Network"
check_network

log_step "Checking Local Manifestation: ${TARGET_DIR}"
# Check if the target exists
if [ -d "$TARGET_DIR" ]; then
    # Directory exists, attempt to update
    # Change directory *before* calling update_repo
    if cd "$TARGET_DIR"; then
        log_info "Entered existing directory: ${BOLD}$(pwd)${RESET}"
        update_repo # Function handles user prompts and pulling
    else
        log_error "Target directory ${TARGET_DIR} exists but could not enter it. Check permissions. Aborting."
        exit 1
    fi
else
    # Directory does not exist, clone it
    clone_repo # Function clones into TARGET_DIR
    # Change directory *after* successful clone
    if cd "$TARGET_DIR"; then
         log_info "Entered newly cloned directory: ${BOLD}$(pwd)${RESET}"
    else
         log_error "Successfully cloned to ${TARGET_DIR} but could not enter it. Check permissions. Aborting."
         exit 1
    fi
fi

# From here, we should be inside the TARGET_DIR

log_step "Displaying Contents"
log_info "Current directory: ${BOLD}$(pwd)${RESET}"
log_info "Top-level contents (excluding .git):"
# Use ls -A to show hidden files (like .env.example) but not . or ..
ls -A

display_guidance # Function provides next-step suggestions

log_step "Ritual Complete"
log_success "The '${REPO_DIR}' repository is ready in ${BOLD}$(pwd)${RESET}"

exit 0 # Explicitly exit with success
