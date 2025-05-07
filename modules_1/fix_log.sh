#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
# set -u
# The return value of a pipeline is the status of the last command to exit with a non-zero status
set -o pipefail

# --- Configuration ---
DRY_RUN=false
BACKUP_SUFFIX=".bak_$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR=""

# --- Helper Functions ---
log() {
  echo "[INFO] $@"
}

warn() {
  echo "[WARN] $@" >&2
}

error_exit() {
  echo "[ERROR] $@" >&2
  exit 1
}

# --- Fix Definitions ---

# Fix 1: Remove Pandas FutureWarning and its associated 'result_data' line
# These warnings come in a pair:
#   /data/data/com.termux/files/home/worldguide/modules_1/analysis.py:329: FutureWarning: Setting an item of incompatible dtype...
#     result_data = ta_func_obj(**ta_input_args, **current_params_for_ta_func)
apply_fix_remove_pandas_warnings_pair() {
  local file="$1"
  log "Applying Fix: Remove Pandas FutureWarning and associated 'result_data' line from $file"

  # Awk script to find the FutureWarning line and then check if the *next* line
  # is the associated 'result_data' line. If so, skip both.
  # If FutureWarning is found but not the specific next line, only the FutureWarning is skipped.
  local awk_script='
  # Match the FutureWarning line (adjust path if necessary)
  /^\/data\/data\/com\.termux\/files\/home\/worldguide\/modules_1\/analysis\.py:.* FutureWarning: Setting an item of incompatible dtype is deprecated/ {
    # Hold the current (FutureWarning) line, then try to read the next line
    # If next line matches the "result_data" pattern, we skip both.
    # Otherwise, we skip the FutureWarning line but print the "next_line" we consumed.
    if (getline next_line > 0) { # Successfully read the next line
      # Match the "result_data" line that follows
      if (next_line ~ /^  result_data = ta_func_obj\(\*\*ta_input_args, \*\*current_params_for_ta_func\)$/) {
        # Both lines match the pattern, skip them by doing nothing here.
        # The 'next' statement will skip the current (FutureWarning) line,
        # and the 'next_line' was consumed by getline, so it is also skipped
        # from the main processing loop of awk.
        next;
      } else {
        # FutureWarning matched, but the next line did not.
        # We will skip the FutureWarning line (by using 'next' below).
        # We must print 'next_line' as it was consumed by getline and is not part of the pair.
        print next_line;
        next;
      }
    } else {
      # FutureWarning matched, but it is the last line of the file. Skip it.
      next;
    }
  }
  # Default action: print the current line if it was not skipped by a 'next' statement above.
  1 {print}
  '

  if [ "$DRY_RUN" = "true" ]; then
    log "  DRY-RUN: Would apply awk script to remove Pandas warnings from $file"
    # To see the changes without modifying, you can use awk and diff:
    echo "    --- Proposed changes for Pandas warnings (diff -u old new) ---"
    awk "$awk_script" "$file" | diff -u "$file" - || true # || true because diff exits 1 on differences
    echo "    --- End of proposed changes ---"
  else
    local tmp_file="${file}.awk_tmp_pandas"
    if awk "$awk_script" "$file" > "$tmp_file"; then
      if cmp -s "$file" "$tmp_file"; then # -s for silent
        log "  No Pandas warning pairs found or file unchanged by this fix."
        rm "$tmp_file"
      else
        mv "$tmp_file" "$file"
        log "  Applied. Pandas warning pairs removed."
      fi
    else
      rm -f "$tmp_file" # Clean up temp file on awk error
      error_exit "  AWK script for Pandas warnings failed for $file"
    fi
  fi
}

# Fix 2: Remove consecutive duplicate lines
apply_fix_remove_consecutive_duplicates() {
  local file="$1"
  log "Applying Fix: Remove consecutive duplicate lines using uniq from $file"

  if [ "$DRY_RUN" = "true" ]; then
    log "  DRY-RUN: Would run 'uniq' on $file"
    # To see the changes:
    echo "    --- Proposed changes for duplicate lines (diff -u old new) ---"
    uniq "$file" | diff -u "$file" - || true
    echo "    --- End of proposed changes ---"
  else
    local tmp_file="${file}.uniq_tmp"
    if uniq "$file" > "$tmp_file"; then
      if cmp -s "$file" "$tmp_file"; then
        log "  No consecutive duplicates found or file unchanged by uniq."
        rm "$tmp_file"
      else
        mv "$tmp_file" "$file"
        log "  Applied. Consecutive duplicates removed."
      fi
    else
      rm -f "$tmp_file"
      error_exit "  uniq command failed for $file"
    fi
  fi
}

# --- Main Processing Logic ---
process_file() {
  local file="$1"

  if [ ! -f "$file" ]; then
    warn "File '$file' not found. Skipping."
    return
  fi

  log "Processing file: $file"

  if [ "$DRY_RUN" = "false" ]; then
    local backup_path
    if [ -n "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR/$(dirname "$file")" # Ensure subdirectories exist in backup dir
        backup_path="$BACKUP_DIR/$file$BACKUP_SUFFIX"
    else
        backup_path="$file$BACKUP_SUFFIX"
    fi

    if cp -p "$file" "$backup_path"; then
      log "  Backed up '$file' to '$backup_path'"
    else
      error_exit "Failed to backup '$file'. Aborting for this file."
      # return 1 # Not needed due to set -e
    fi
  else
    log "  Dry Run: Original file '$file' would be backed up."
  fi

  # --- Call your fix functions here in order ---
  # It's often better to remove specific complex patterns first,
  # then general cleanups like duplicate lines.

  apply_fix_remove_pandas_warnings_pair "$file"
  apply_fix_remove_consecutive_duplicates "$file"

  log "Finished processing $file"
  echo # Add a blank line for readability
}

# --- Script Entry Point ---
usage() {
  echo "Usage: $0 [options] <file1> [file2 ...]"
  echo "       $0 [options] --find <find_command_args>"
  echo ""
  echo "Applies predefined fixes to log files (Pandas warnings, duplicate lines)."
  echo ""
  echo "Options:"
  echo "  --dry-run             Show what would be done without making changes."
  echo "  --backup-suffix SUFFIX Override default backup suffix (default: .bak_YYYYMMDD_HHMMSS)."
  echo "  --backup-dir    DIR   Create backups in this directory, preserving original paths."
  echo "  --find ARGS           Use 'find . ARGS -print0' to get a list of files."
  echo "                        Example: --find '-name \"*.log\" -type f'"
  echo "  -h, --help            Show this help message."
}

files_to_process=()
find_args=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --backup-suffix)
      BACKUP_SUFFIX="$2"
      shift; shift
      ;;
    --backup-dir)
      BACKUP_DIR="$2"
      shift; shift
      ;;
    --find)
      find_args="$2"
      shift; shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" == -* ]]; then
        error_exit "Unknown option: $1. Use --help for usage."
      fi
      files_to_process+=("$1")
      shift
      ;;
  esac
done

if [ "$DRY_RUN" = "true" ]; then
    log "Dry run mode enabled. No files will be modified."
fi

if [ -n "$find_args" ]; then
  if [ ${#files_to_process[@]} -gt 0 ]; then
    error_exit "Cannot use --find and specify individual files at the same time."
  fi
  log "Finding files with: find . ${find_args} -print0"
  while IFS= read -r -d $'\0' file; do
    # Remove leading ./ if present, for cleaner paths if using BACKUP_DIR
    cleaned_file="${file#./}"
    files_to_process+=("$cleaned_file")
  done < <(eval "find . ${find_args} -print0")
elif [ ${#files_to_process[@]} -eq 0 ]; then
  usage
  error_exit "No files specified. Provide file paths or use --find."
fi

if [ ${#files_to_process[@]} -eq 0 ]; then
  log "No files found to process."
  exit 0
fi

if [ "$DRY_RUN" = "false" ] && [ -n "$BACKUP_DIR" ]; then
    if mkdir -p "$BACKUP_DIR"; then
        log "Backups will be stored in '$BACKUP_DIR'"
    else
        error_exit "Could not create backup directory '$BACKUP_DIR'"
    fi
fi

log "Starting batch file processing..."
for file_path in "${files_to_process[@]}"; do
  process_file "$file_path"
done

log "All specified fixes applied."
if [ "$DRY_RUN" = "true" ]; then
  log "DRY RUN COMPLETE: No actual changes were made."
else
  log "Processing complete. Check backups if necessary."
  if [ -n "$BACKUP_DIR" ]; then
    log "General backups are in: $BACKUP_DIR"
  else
    log "General backups have suffix: $BACKUP_SUFFIX (in the same directory as original files)"
  fi
fi