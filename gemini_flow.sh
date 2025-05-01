#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting. - Disabled as it might break existing logic relying on empty strings.
# Exit status of the last command that threw a non-zero exit code is returned.
# set -eu
set -o pipefail

# --- Configuration & Initial Defaults ---
CUSTOM_CONFIG_PATH=""
DRY_RUN_MODE=false
REPO_INPUT="" # Renamed from REPO_URL to clarify it's the initial input
REPO_URL=""   # Will hold the actual URL if cloning
TARGET_REPO_DIR="" # Will hold the absolute path to the target repo directory
IS_LOCAL_PATH=false # Flag to check if input is treated as a local path
INTERACTIVE_MODE=false
DEFAULT_COMMIT_MSG="Add Gemini Code Assist configuration"
COMMIT_MSG_PROVIDED=false # Flag to track if --commit-msg was used

# --- Detect Interactivity ---
# Checks if file descriptor 0 (stdin) is connected to a terminal
if [[ -t 0 ]]; then
   INTERACTIVE_MODE=true
fi

# --- Helper: Logging ---
# Consistent logging prefix
log() {
    echo "[INFO] $@"
}
warn() {
    echo "[WARN] $@" >&2
}
error() {
    echo "[ERROR] $@" >&2
}

# --- Function: Display Usage Instructions ---
usage() {
   cat << EOF
Usage: $0 <repository_url_or_path> [OPTIONS]

Arguments:
  repository_url_or_path  Required. The URL of the Git repository to clone,
                          or the local path to an existing Git repository.

Options:
  -c, --config FILE       Optional. Path to a custom config.yaml file to use.
                          Relative paths are resolved from the directory where
                          the script is run. In interactive mode, prompts if omitted.
  -d, --dry-run           Optional. Simulate actions without making changes
                          (implies interactive prompts might show what *would* happen).
                          In interactive mode, prompts if omitted.
  --commit-msg 'MESSAGE'  Optional. Specify the git commit message.
                          If omitted in interactive mode, prompts if needed.
                          Default: "$DEFAULT_COMMIT_MSG"
  -h, --help              Show this help message and exit.

Behavior:
  Interactive Mode (Terminal): Prompts for missing required info and optional flags.
  Non-Interactive Mode (Workflow/Pipe): Requires repository_url_or_path argument.
                                        Uses defaults or arguments for options.
                                        Does not prompt for input.

Examples:
  # Interactive setup for a remote repo
  $0 https://github.com/owner/repo.git

  # Non-interactive setup using a local path and custom config/message
  $0 /path/to/local/repo -c ../my-gemini-config.yaml --commit-msg "Setup Gemini via script"

  # Dry run for a remote repo in a workflow
  ./setup_gemini.sh https://github.com/owner/repo.git -d
EOF
   exit 1
}

# --- Argument Parsing ---
# Use getopt for robust parsing
PARSED_ARGS=$(getopt -o c:dh -l config:,dry-run,commit-msg:,help -- "$@")
if [[ $? -ne 0 ]]; then
   usage
fi

# Set positional parameters based on getopt output
eval set -- "$PARSED_ARGS"

COMMIT_MSG="$DEFAULT_COMMIT_MSG" # Initialize with default

while true; do
   case "$1" in
      -c | --config)
         CUSTOM_CONFIG_PATH="$2"
         shift 2
         ;;
      -d | --dry-run)
         DRY_RUN_MODE=true
         shift
         ;;
      --commit-msg)
         COMMIT_MSG="$2"
         COMMIT_MSG_PROVIDED=true # Mark that it was explicitly provided
         shift 2
         ;;
      -h | --help)
         usage
         ;;
      --) # End of options
         shift
         break
         ;;
      *) # Should not happen with getopt
         error "Internal parsing error!"
         usage
         ;;
   esac
done

# Handle the remaining argument (repository URL/path)
if [[ $# -gt 0 ]]; then
   REPO_INPUT="$1"
   shift
else
   # REPO_INPUT is required unless in interactive mode (where we prompt)
   if [[ "$INTERACTIVE_MODE" = false ]]; then
      error "Repository URL/path must be provided as the first argument in non-interactive mode."
      usage
   fi
fi

if [[ $# -gt 0 ]]; then
    warn "Ignoring extra arguments starting with '$1'."
fi

# --- Interactive Prompts (Only if INTERACTIVE_MODE is true) ---
if [[ "$INTERACTIVE_MODE" = true ]]; then
   # Prompt for Repository URL/Path if not provided
   if [[ -z "$REPO_INPUT" ]]; then
      log "No repository URL or path provided via command line."
      read -rp "Enter the target GitHub repository URL or local path: " REPO_INPUT
      if [[ -z "$REPO_INPUT" ]]; then
         error "Repository URL/path cannot be empty."
         exit 1
      fi
   fi

   # Prompt for Custom Config (optional) - only if not already set by -c
   if [[ -z "$CUSTOM_CONFIG_PATH" ]]; then
      read -rp "Enter path to custom config.yaml (or press Enter for default): " CUSTOM_CONFIG_PATH
   fi

   # Prompt for Dry Run (optional) - only if not already set by -d
   if [[ "$DRY_RUN_MODE" = false ]]; then
        read -rp "Run in dry-run mode? (Simulate only) [y/N]: " response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            DRY_RUN_MODE=true
        fi
   fi

   # Note: Commit message prompt is deferred until we know a commit is needed
fi

# --- Prerequisites Check ---
if ! command -v git &> /dev/null; then
   error "Git is not installed. Please install Git and ensure it's in your PATH."
   exit 1
fi

# Ensure realpath is available (common on Linux/macOS)
# Define a basic fallback if needed, though complex paths might fail
if ! command -v realpath &> /dev/null; then
   warn "'realpath' command not found. Path resolution might be less robust."
   realpath() {
       readlink -f "$@" || (cd "$(dirname "$@")" && echo "$(pwd)/$(basename "$@")")
   }
fi


# --- Helper Function for Command Execution ---
# Takes a description string, followed by the command and its arguments
run() {
    local cmd_desc="$1" # Description of the action
    shift             # Remove description from arguments
    local cmd=("$@")    # Remaining arguments form the command

    log "--> Action: $cmd_desc"
    if [[ "$DRY_RUN_MODE" = true ]]; then
        # Safely print command arguments for dry run
        printf "[DRY-RUN] Would execute: %q " "${cmd[@]}"
        printf "\n"
        # Simulate success in dry run for flow control, unless it's a check command
        case "${cmd[0]}" in
            git)
                case "${cmd[1]}" in
                    # Simulate checks failing if needed, otherwise success
                    diff|status|rev-parse) return 1 ;; # Simulate no changes/not repo etc.
                    *) return 0 ;; # Simulate clone/add/commit/push success
                esac
                ;;
            test|\[\[|\[) return 1 ;; # Simulate checks failing by default
            *) return 0 ;; # Simulate other commands succeeding
        esac
    else
        # Execute the command with proper quoting
        "${cmd[@]}"
        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            error "Command failed with exit code $exit_code: $(printf "'%q' " "${cmd[@]}")"
            # Attempt to return to original directory if we are inside the target repo
            if [[ -v ORIGINAL_DIR && -v TARGET_REPO_DIR && "$PWD" == "$TARGET_REPO_DIR" && "$PWD" != "$ORIGINAL_DIR" ]]; then
                log "Attempting to return to original directory $ORIGINAL_DIR..."
                cd "$ORIGINAL_DIR" || warn "Failed to cd back to $ORIGINAL_DIR"
            fi
            exit $exit_code
        fi
        return $exit_code
    fi
}

# --- Main Script Logic ---
ORIGINAL_DIR=$(pwd) # Store original directory
log "Script initiated from: $ORIGINAL_DIR"
log "Dry run mode: $DRY_RUN_MODE"
log "Interactive mode: $INTERACTIVE_MODE"

# --- Determine Target Directory and Prepare Repository ---

# Heuristic: Does it look like a URL? (basic check)
if [[ "$REPO_INPUT" == *"://"* || "$REPO_INPUT" == *"@"* ]]; then
   IS_LOCAL_PATH=false
   REPO_URL="$REPO_INPUT"
else
   # Assume it *might* be a local path
   IS_LOCAL_PATH=true
fi

if [[ "$IS_LOCAL_PATH" = true ]]; then
   log "Input '$REPO_INPUT' appears to be a local path. Attempting to use it."
   # Try to resolve the path relative to the original directory
   RESOLVED_PATH=$(realpath "$ORIGINAL_DIR/$REPO_INPUT" 2>/dev/null || echo "$ORIGINAL_DIR/$REPO_INPUT") # Fallback if realpath fails/doesn't exist

   if [[ "$DRY_RUN_MODE" = true ]]; then
       log "[DRY-RUN] Would check if '$RESOLVED_PATH' is a valid Git repository directory."
       TARGET_REPO_DIR="$RESOLVED_PATH" # Assume it's valid for dry-run flow
       # We still need to change directory in dry-run to simulate correctly
       run "Change directory to target (simulation)" cd "$TARGET_REPO_DIR"
   else
       if [[ ! -d "$RESOLVED_PATH" ]]; then
           error "Provided path '$REPO_INPUT' (resolved to '$RESOLVED_PATH') is not a valid directory."
           exit 1
       fi
       if [[ ! -d "$RESOLVED_PATH/.git" ]]; then
           error "Directory '$RESOLVED_PATH' does not appear to be a Git repository (.git missing)."
           exit 1
       fi
       TARGET_REPO_DIR="$RESOLVED_PATH"
       log "Using existing local repository at: $TARGET_REPO_DIR"
       run "Change directory to target repository" cd "$TARGET_REPO_DIR"
   fi
else
   # Input looks like a URL, proceed with cloning logic
   REPO_NAME=$(basename "$REPO_URL" .git)
   # Clone target relative to the original directory
   POTENTIAL_TARGET_DIR="$ORIGINAL_DIR/$REPO_NAME"

   log "Input '$REPO_INPUT' appears to be a URL."

   if [[ -d "$POTENTIAL_TARGET_DIR/.git" ]]; then
       log "Repository '$REPO_NAME' already exists locally at '$POTENTIAL_TARGET_DIR'."
       TARGET_REPO_DIR="$POTENTIAL_TARGET_DIR"
       run "Change directory to existing repository" cd "$TARGET_REPO_DIR"
       # Optional: Consider adding 'git pull' here if desired, maybe only in interactive mode?
       # if [[ "$INTERACTIVE_MODE" = true ]]; then read -p "Pull latest changes? [y/N]: " pull_resp ... fi
       # run "Pulling latest changes" git pull
   else
       log "Cloning repository..."
       # Perform the clone operation itself outside the 'run' function initially
       # to correctly capture TARGET_REPO_DIR even in dry-run
       CLONE_TARGET="$REPO_NAME" # Clone into a subdir named after the repo

       if [[ "$DRY_RUN_MODE" = true ]]; then
           log "[DRY-RUN] Would clone '$REPO_URL' into '$ORIGINAL_DIR/$CLONE_TARGET'"
           TARGET_REPO_DIR="$POTENTIAL_TARGET_DIR" # Assume clone would succeed
           run "Simulate changing directory to cloned repo" cd "$TARGET_REPO_DIR" # Simulate cd
       else
            # Check if target directory exists but isn't a repo - avoid clobbering
            if [[ -e "$POTENTIAL_TARGET_DIR" && ! -d "$POTENTIAL_TARGET_DIR/.git" ]]; then
                 error "Target directory '$POTENTIAL_TARGET_DIR' exists but is not a git repository. Cannot clone."
                 exit 1
            fi
           # Execute clone relative to ORIGINAL_DIR
           (cd "$ORIGINAL_DIR" && \
               git clone "$REPO_URL" "$CLONE_TARGET")
           clone_exit_code=$?
           if [[ $clone_exit_code -ne 0 ]]; then
               error "Failed to clone repository from '$REPO_URL'."
               # No need to cd back, still in ORIGINAL_DIR
               exit $clone_exit_code
           fi
           TARGET_REPO_DIR="$POTENTIAL_TARGET_DIR"
           run "Change directory to cloned repository" cd "$TARGET_REPO_DIR"
       fi
   fi
fi

log "Target repository directory set to: $TARGET_REPO_DIR"

# --- Define Paths Relative to Repository Root ---
# (Now guaranteed to be inside TARGET_REPO_DIR)
GEMINI_DIR=".gemini"
CONFIG_FILE="$GEMINI_DIR/config.yaml"
BACKUP_FILE="$CONFIG_FILE.backup.$(date +%Y%m%d%H%M%S)" # Timestamped backup

# --- Configuration File Handling ---
CONFIG_ACTION="none" # Possible values: none, create, overwrite

# Check if config file or directory already exists
# Use -e to check for file/dir/symlink existence at the path
if [[ -e "$CONFIG_FILE" || -e "$GEMINI_DIR" ]]; then
   log "Existing '.gemini' configuration found."
   if [[ "$INTERACTIVE_MODE" = true ]]; then
      if [[ "$DRY_RUN_MODE" = true ]]; then
         log "[DRY-RUN] Would prompt to overwrite existing configuration (backing up $CONFIG_FILE if it exists)."
         CONFIG_ACTION="overwrite" # Assume yes for dry-run flow
      else
         read -rp "Overwrite existing configuration? (Will backup $CONFIG_FILE if it exists) [y/N]: " response
         if [[ "$response" =~ ^[Yy]$ ]]; then
            CONFIG_ACTION="overwrite"
         else
            log "Skipping configuration setup as requested."
            cd "$ORIGINAL_DIR" # Go back before exiting
            exit 0
         fi
      fi
   else
      # Non-interactive: Default action is to overwrite safely
      log "Non-interactive mode: Proceeding with overwrite (backup will be created if $CONFIG_FILE exists)."
      CONFIG_ACTION="overwrite"
   fi

   if [[ "$CONFIG_ACTION" = "overwrite" ]]; then
      if [[ -f "$CONFIG_FILE" ]]; then
            run "Backing up existing configuration to $BACKUP_FILE" cp "$CONFIG_FILE" "$BACKUP_FILE"
      elif [[ -e "$CONFIG_FILE" ]]; then
            warn "Existing '$CONFIG_FILE' is not a regular file. Cannot back up automatically. Overwriting."
      fi
      # Ensure the directory exists (might exist already, -p handles it)
      run "Ensuring $GEMINI_DIR directory exists" mkdir -p "$GEMINI_DIR"
   fi
else
   # No existing config found, create it
   CONFIG_ACTION="create"
   run "Creating '$GEMINI_DIR' directory" mkdir -p "$GEMINI_DIR"
fi


# --- Create or Copy config.yaml inside .gemini ---
if [[ "$CONFIG_ACTION" = "create" || "$CONFIG_ACTION" = "overwrite" ]]; then

    # Double-check directory exists before writing (only in non-dry-run)
    if [[ "$DRY_RUN_MODE" = false && ! -d "$GEMINI_DIR" ]]; then
        error "Failed to create or access $GEMINI_DIR directory."
        cd "$ORIGINAL_DIR"
        exit 1
    fi

    if [[ -n "$CUSTOM_CONFIG_PATH" ]]; then
        # Resolve custom config path relative to the ORIGINAL_DIR where the script was run
        ABS_CUSTOM_CONFIG_PATH=$(realpath "$ORIGINAL_DIR/$CUSTOM_CONFIG_PATH" 2>/dev/null || echo "$ORIGINAL_DIR/$CUSTOM_CONFIG_PATH")
        log "Using custom configuration file: $ABS_CUSTOM_CONFIG_PATH"

        if [[ "$DRY_RUN_MODE" = false ]]; then
             if [[ ! -f "$ABS_CUSTOM_CONFIG_PATH" ]]; then
                error "Custom configuration file not found: $ABS_CUSTOM_CONFIG_PATH (searched relative to $ORIGINAL_DIR)"
                cd "$ORIGINAL_DIR"
                exit 1
             fi
        fi
        run "Copying custom configuration to $CONFIG_FILE" cp "$ABS_CUSTOM_CONFIG_PATH" "$CONFIG_FILE"
    else
        log "Creating default $CONFIG_FILE"
        # Define default content using a heredoc
        DEFAULT_CONFIG=$(cat << 'EOF'
# Gemini Code Assist Configuration - Default
# See documentation for all options: <Link to Docs if available>

# Example: Enable fun mode (if applicable)
have_fun: true

# Code Review Settings
code_review:
  # Completely disable code review feature
  disable: false

  # Minimum severity level for comments to be posted (LOW, MEDIUM, HIGH, CRITICAL)
  comment_severity_threshold: MEDIUM

  # Max number of review comments per run (-1 for unlimited)
  max_review_comments: -1

  # Actions triggered when a pull request is opened or synchronized
  pull_request_opened:
    # Provide helpful hints via comments
    help: false
    # Generate a summary comment of the PR changes
    summary: true
    # Perform code review on the PR changes
    code_review: true

# Other potential top-level features (add as needed)
# code_explanation:
#   disable: true
EOF
)
        if [[ "$DRY_RUN_MODE" = true ]]; then
            log "[DRY-RUN] Would create default $CONFIG_FILE with content:"
            echo "$DEFAULT_CONFIG" # Print content in dry run
        else
            # Write the default config to the file
            echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
            write_exit_code=$?
             if [[ $write_exit_code -ne 0 ]]; then
                error "Failed to write default configuration to $CONFIG_FILE."
                cd "$ORIGINAL_DIR"
                exit 1
            fi
        fi
    fi

    # --- Git Operations ---
    run "Staging $CONFIG_FILE for commit" git add "$CONFIG_FILE"

    COMMIT_NEEDED=false
    if [[ "$DRY_RUN_MODE" = true ]]; then
        log "[DRY-RUN] Would check for staged changes."
        # Assume changes exist for dry-run commit/push simulation
        COMMIT_NEEDED=true
    else
        # Check if there are changes staged for commit
        # Use plumbing command for scriptability, exit code 0 if no changes
        if ! git diff --cached --quiet; then
            log "Changes detected in $CONFIG_FILE."
            COMMIT_NEEDED=true
        else
            log "No changes staged for $CONFIG_FILE. Nothing to commit."
            COMMIT_NEEDED=false
        fi
    fi

    if [[ "$COMMIT_NEEDED" = true ]]; then
        # Prompt for commit message only if interactive AND user didn't provide --commit-msg via flag
        if [[ "$INTERACTIVE_MODE" = true && "$COMMIT_MSG_PROVIDED" = false ]]; then
            read -rp "Enter commit message (default: '$COMMIT_MSG'): " USER_COMMIT_MSG
            # Use user message if provided, otherwise stick with the default (already in COMMIT_MSG)
            COMMIT_MSG="${USER_COMMIT_MSG:-$COMMIT_MSG}"
        fi

        run "Committing configuration with message '$COMMIT_MSG'" git commit -m "$COMMIT_MSG"
        # Only attempt push if not a local-only path scenario? Or always try?
        # Let's always try, git will handle remote config.
        run "Pushing changes to origin" git push
    fi
else
    # This case happens if user chose not to overwrite in interactive mode
    log "No configuration changes were made."
fi


# --- Completion & Post-Setup ---
echo ""
# Get repo name again from the current directory for the final message
FINAL_REPO_NAME=$(basename "$PWD")
log "Gemini Code Assist configuration setup completed for '$FINAL_REPO_NAME'."
echo ""
cat << EOF
================ IMPORTANT NEXT STEPS ================
1. Ensure the Gemini Code Assist GitHub App is installed and has access
   to the '$FINAL_REPO_NAME' repository.
   - Visit Repository Settings > Integrations (or Applications).
   - Install from Marketplace if needed: https://github.com/marketplace/gemini-code-assist
   - Configure the App to grant access to this specific repository.
2. Review the '$CONFIG_FILE' file created or updated in the repository
   to ensure the settings meet your needs.
====================================================
EOF
echo ""

# --- Cleanup ---
log "Returning to original directory: $ORIGINAL_DIR"
cd "$ORIGINAL_DIR"

exit 0
```

---

**Separated and Enhanced GitHub Actions Workflow File:**

This YAML should be saved as a separate file, typically `.github/workflows/setup-gemini.yml`.

```yaml
# .github/workflows/setup-gemini.yml
name: Setup Gemini Code Assist

# Allow manual triggering from the GitHub UI
on:
  workflow_dispatch:
    inputs:
      repository_url_or_path:
        description: 'Required: Full HTTPS/SSH URL or local path (relative to runner workspace) of the target repository.'
        required: true
        type: string
      custom_config_path:
        description: 'Optional: Path (relative to *this* repo root where the workflow runs) to a custom config.yaml file to copy.'
        required: false
        type: string
      commit_message:
        description: 'Optional: Git commit message for the config file. Default used if empty.'
        required: false
        type: string
        default: 'chore: Add Gemini Code Assist configuration [Workflow]' # Default for workflow runs
      dry_run:
        description: 'Run in dry-run mode (simulate only)?'
        required: false
        type: boolean
        default: false

jobs:
  setup-gemini:
    runs-on: ubuntu-latest
    permissions:
      # Required to clone/push to the target repo IF it's the SAME repo the workflow runs in
      # OR if the target repo is public (for cloning). Needs 'write' to push the commit.
      contents: write
      # NOTE: If targeting a DIFFERENT PRIVATE repo, 'contents: write' is NOT enough.
      # You will need a Personal Access Token (PAT) with 'repo' scope (classic)
      # or fine-grained access ('contents: write') to the target repo.
      # See PAT configuration step below.

    steps:
      - name: Checkout Workflow Repo Code
        uses: actions/checkout@v4
        # Set persist-credentials to false IF using a PAT for a different repo (Option 2 below)
        # This prevents the default GITHUB_TOKEN from interfering with the PAT.
        # with:
        #   persist-credentials: false

      - name: Set up Git User for Commits
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"

      # --- Git Authentication (Choose ONE option) ---

      # Option 1: Target repo is the SAME as this workflow repo OR target is PUBLIC
      # Uses the default GITHUB_TOKEN granted by 'permissions.contents: write'.
      # No extra step needed here if using Option 1.

      # Option 2: Target repo is DIFFERENT and PRIVATE
      # Requires a PAT stored as a secret (e.g., CROSS_REPO_PAT).
      # Uncomment and configure the following step if using Option 2.
      # - name: Configure Git for Private Repo Access (using PAT)
      #   if: github.event.inputs.repository_url_or_path != '' # Basic check if URL is provided
      #   run: |
      #     echo "Configuring git to use PAT for cross-repo access..."
      #     # Ensure the default credential helper set by actions/checkout is removed if persist-credentials was true or default
      #     git config --global --unset-all credential.helper || true
      #     # Add the PAT helper - configure HTTPS access
      #     git config --global url."https://x-access-token:${{ secrets.CROSS_REPO_PAT }}@github.com/".insteadOf "https://github.com/"
      #     # If using SSH URLs, configure SSH key instead (more complex setup)
      #   env:
      #     # Ensure CROSS_REPO_PAT is defined in this repository's Actions secrets
      #     # The PAT needs 'repo' scope (classic) or fine-grained 'contents: write' for the target repo
      #     CROSS_REPO_PAT: ${{ secrets.CROSS_REPO_PAT }}

      - name: Make Script Executable
        run: chmod +x ./gem_setup.sh # Adjust path if script is elsewhere

      - name: Run Gemini Setup Script (Non-Interactive)
        run: |
          # Construct the arguments array for the script
          # This handles spaces and special characters in inputs safely
          ARGS=()
          ARGS+=("${{ inputs.repository_url_or_path }}") # Required positional argument

          if [[ "${{ inputs.dry_run }}" == "true" ]]; then
            ARGS+=("-d")
          fi

          if [[ -n "${{ inputs.custom_config_path }}" ]]; then
            # First, verify the custom config file exists in the *checked out workflow repo* before passing it
            if [[ ! -f "${{ inputs.custom_config_path }}" ]]; then
               echo "::error::Custom config file specified ('${{ inputs.custom_config_path }}') but not found in workflow checkout directory ($PWD)."
               exit 1
            fi
            echo "Custom config file '${{ inputs.custom_config_path }}' found in workflow repo."
            ARGS+=("-c" "${{ inputs.custom_config_path }}")
          fi

          # Use the commit message if provided (default is handled by workflow input default)
          if [[ -n "${{ inputs.commit_message }}" ]]; then
            ARGS+=("--commit-msg" "${{ inputs.commit_message }}")
          fi

          echo "Running script with arguments:"
          # Print args safely using printf %q for logging verification
          printf " > ./gem_setup.sh"
          printf " %q" "${ARGS[@]}"
          echo "" # Newline after printing command

          # Execute the script with the constructed arguments
          # The script runs non-interactively because stdin is not a TTY in Actions
          ./gem_setup.sh "${ARGS[@]}"
        env:
          # Pass the default GITHUB_TOKEN. The script's internal git commands
          # (clone, push) will use this if the repo URL is HTTPS and it's the
          # SAME repo OR if PAT configured globally (Option 2).
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          # If using PAT (Option 2), it's configured globally via git config,
          # so no need to pass CROSS_REPO_PAT env var here unless script specifically needed it.

      # Optional: Add steps here to clean up cloned repo if needed,
      # although the runner workspace is ephemeral anyway.
      # The script attempts to cd back to the original directory ($GITHUB_WORKSPACE).
      # - name: List workspace contents after script
      #   run: ls -la $GITHUB_WORKSPACE