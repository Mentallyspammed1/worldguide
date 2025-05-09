# ~/.p10k.zsh
#
# Powerlevel10k configuration file.
#
# Original generated by Powerlevel10k configuration wizard on 2025-03-28.
# Based on romkatv/powerlevel10k/config/p10k-rainbow.zsh.
# Original wizard options: nerdfont-v3 + powerline, small icons, rainbow, unicode, 12h time,
# slanted separators, blurred heads, blurred tails, 2 lines, dotted, right frame,
# dark-ornaments, compact, few icons, concise, transient_prompt, instant_prompt=quiet.
#
# This configuration has been reviewed and enhanced by Pyrmethus, focusing on:
# - Readability and structure.
# - Termux environment considerations (screen space, relevant segments like battery).
# - Minor visual tweaks (e.g., branch icon, time format).
# - Adding comments explaining choices and how to customize further.
#
# To apply changes: source ~/.p10k.zsh
# To re-run the wizard: p10k configure
#
# Tip: Looking for a nice color? Here's a one-liner to print colormap.
#
#   for i in {0..255}; do print -Pn "%K{$i}  %k%F{$i}${(l:3::0:)i}%f " ${${(M)$((i%6)):#3}:+$'\n'}; done
#
# Tip: Ensure you have a Nerd Font installed and configured in your Termux terminal settings.
#      (e.g., FiraCode Nerd Font, JetBrainsMono Nerd Font)

#===============================================================================
# Initialization and Safety Checks
#===============================================================================

# Temporarily change options.
'builtin' 'local' '-a' 'p10k_config_opts'
[[ ! -o 'aliases'         ]] || p10k_config_opts+=('aliases')
[[ ! -o 'sh_glob'         ]] || p10k_config_opts+=('sh_glob')
[[ ! -o 'no_brace_expand' ]] || p10k_config_opts+=('no_brace_expand')
'builtin' 'setopt' 'no_aliases' 'no_sh_glob' 'brace_expand'

() {
  emulate -L zsh -o extended_glob

  # Unset all configuration options. This allows you to apply configuration changes without
  # restarting zsh. Edit ~/.p10k.zsh and type `source ~/.p10k.zsh`.
  unset -m '(POWERLEVEL9K_*|DEFAULT_USER)~POWERLEVEL9K_GITSTATUS_DIR'

  # Zsh >= 5.1 is required.
  [[ $ZSH_VERSION == (5.<1->*|<6->.*) ]] || return

  #===============================================================================
  # Prompt Structure - Define Left and Right Prompt Elements
  #===============================================================================

  # The list of segments shown on the left. Fill it with the most important segments.
  typeset -g POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(
    # =========================[ Line #1 ]=========================
    os_icon                 # OS identifier (useful for Termux)
    dir                     # Current directory
    vcs                     # Git status
    # =========================[ Line #2 ]=========================
    newline                 # \n
    prompt_char             # Prompt symbol (changes color on error)
  )

  # The list of segments shown on the right. Fill it with less important segments.
  # Right prompt segments are automatically hidden when the input line reaches them.
  # NOTE: Many less common segments are commented out below to save screen space
  #       in Termux. Uncomment any you frequently use.
  typeset -g POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(
    # =========================[ Line #1 ]=========================
    status                  # Exit code of the last command
    command_execution_time  # Duration of the last command
    background_jobs         # Presence of background jobs
    context                 # User@hostname (only shows if root or ssh)
    # --- Language/Environment Specific (Uncomment needed ones) ---
    # virtualenv              # Python virtual environment
    pyenv                   # Python environment (pyenv)
    # anaconda                # Conda environment
    # direnv                  # Direnv status (https://direnv.net/)
    # asdf                    # Asdf version manager (mult-language)
    # goenv                   # Go environment (goenv)
    # nodenv                  # Node.js version (nodenv)
    nvm                     # Node.js version (nvm)
    # nodeenv                 # Node.js environment (nodeenv)
    # node_version          # Node.js version (if not using a manager)
    # go_version            # Go version (if not using a manager)
    # rust_version          # Rustc version
    # dotnet_version        # .NET version
    # java_version          # Java version
    # php_version           # PHP version
    # ruby_version          # Ruby version (if not using a manager)
    # rbenv                   # Ruby version (rbenv)
    # rvm                     # Ruby version (rvm)
    # luaenv                  # Lua version (luaenv)
    # jenv                    # Java version (jenv)
    # plenv                   # Perl version (plenv)
    # perlbrew                # Perl version (perlbrew)
    # phpenv                  # PHP version (phpenv)
    # scalaenv                # Scala version (scalaenv)
    # haskell_stack           # Haskell stack version
    # fvm                     # Flutter version management
    # --- Cloud & Infra (Uncomment needed ones) ---
    # kubecontext             # Current kubernetes context
    # aws                     # AWS profile
    # azure                   # Azure account name
    # gcloud                  # Google Cloud CLI info
    # google_app_cred         # Google Application Credentials
    # terraform               # Terraform workspace
    # --- File Managers & Shells (Uncomment needed ones) ---
    # ranger                  # Ranger shell indicator
    # yazi                    # Yazi shell indicator
    # nnn                     # Nnn shell indicator
    # lf                      # Lf shell indicator
    # xplr                    # Xplr shell indicator
    vim_shell               # Vim shell indicator (:sh)
    midnight_commander      # Midnight Commander shell indicator
    # nix_shell               # Nix shell indicator
    # chezmoi_shell           # Chezmoi shell indicator
    # toolbox                 # Toolbox container indicator
    # --- Other System Info & Tools (Uncomment needed ones) ---
    # nordvpn                 # NordVPN status (Linux only)
    # vpn_ip                # VPN IP indicator (customize interface regex)
    # public_ip             # Public IP address
    # load                  # CPU load
    # disk_usage            # Disk usage
    # ram                   # Free RAM
    # swap                  # Used swap
    # todo                    # Todo items (todo.txt-cli)
    # timewarrior             # Timewarrior status
    # taskwarrior             # Taskwarrior task count
    # per_directory_history   # Oh My Zsh per-directory-history status
    # cpu_arch              # CPU architecture (can be useful on ARM)
    battery                 # Battery status (useful for Termux)
    time                    # Current time (24-hour format)
    # =========================[ Line #2 ]=========================
    newline
    # ip                    # IP address and bandwidth for specific interface
    # proxy                 # System-wide proxy indicator
    # wifi                  # WiFi speed
    # example               # Example user-defined segment
  )

  #===============================================================================
  # Core Visual Style Settings
  #===============================================================================

  # Defines character set used by powerlevel10k. Requires a Nerd Font.
  typeset -g POWERLEVEL9K_MODE=nerdfont-v3
  # Padding between icons and text. `moderate` adds space for non-monospace fonts.
  typeset -g POWERLEVEL9K_ICON_PADDING=none
  # Icon position relative to text. Empty means before on left, after on right.
  typeset -g POWERLEVEL9K_ICON_BEFORE_CONTENT=
  # Add an empty line before each prompt.
  typeset -g POWERLEVEL9K_PROMPT_ADD_NEWLINE=false

  # --- Multiline Prompt Connectors ---
  # Symbols connecting left prompt lines (when using ornaments).
  typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_PREFIX=
  typeset -g POWERLEVEL9K_MULTILINE_NEWLINE_PROMPT_PREFIX=
  typeset -g POWERLEVEL9K_MULTILINE_LAST_PROMPT_PREFIX=
  # Symbols connecting right prompt lines (right frame style).
  typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_SUFFIX='%240F─╮'
  typeset -g POWERLEVEL9K_MULTILINE_NEWLINE_PROMPT_SUFFIX='%240F─┤'
  typeset -g POWERLEVEL9K_MULTILINE_LAST_PROMPT_SUFFIX='%240F─╯'

  # --- Prompt Filler ---
  # Filler character between left and right prompts on the first line.
  typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_GAP_CHAR='·' # Alternatives: ' ', '─'
  typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_GAP_BACKGROUND=
  typeset -g POWERLEVEL9K_MULTILINE_NEWLINE_PROMPT_GAP_BACKGROUND=
  if [[ $POWERLEVEL9K_MULTILINE_FIRST_PROMPT_GAP_CHAR != ' ' ]]; then
    # Color of the filler character. Match the frame color (240).
    typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_GAP_FOREGROUND=240
    # Ensure filler extends to screen edges when prompts are empty.
    typeset -g POWERLEVEL9K_EMPTY_LINE_LEFT_PROMPT_FIRST_SEGMENT_END_SYMBOL='%{%}'
    typeset -g POWERLEVEL9K_EMPTY_LINE_RIGHT_PROMPT_FIRST_SEGMENT_START_SYMBOL='%{%}'
  fi

  # --- Segment Separators ---
  # Separator between same-color segments on the left.
  typeset -g POWERLEVEL9K_LEFT_SUBSEGMENT_SEPARATOR='\u2571' # '⧱' slant 1
  # Separator between same-color segments on the right.
  typeset -g POWERLEVEL9K_RIGHT_SUBSEGMENT_SEPARATOR='\u2571' # '⧱' slant 1
  # Separator between different-color segments on the left (Powerline slanted).
  typeset -g POWERLEVEL9K_LEFT_SEGMENT_SEPARATOR='\uE0BC' # '' slant right
  # Separator between different-color segments on the right (Powerline slanted).
  typeset -g POWERLEVEL9K_RIGHT_SEGMENT_SEPARATOR='\uE0BA' # '' slant left
  # To join segments without a separator, add "_joined" to the second segment name.

  # --- Prompt Boundaries (Blurred/Fade Effect) ---
  # The right end of left prompt.
  typeset -g POWERLEVEL9K_LEFT_PROMPT_LAST_SEGMENT_END_SYMBOL='▓▒░'
  # The left end of right prompt.
  typeset -g POWERLEVEL9K_RIGHT_PROMPT_FIRST_SEGMENT_START_SYMBOL='░▒▓'
  # The left end of left prompt.
  typeset -g POWERLEVEL9K_LEFT_PROMPT_FIRST_SEGMENT_START_SYMBOL='░▒▓'
  # The right end of right prompt.
  typeset -g POWERLEVEL9K_RIGHT_PROMPT_LAST_SEGMENT_END_SYMBOL='▓▒░'
  # Left prompt terminator for empty lines.
  typeset -g POWERLEVEL9K_EMPTY_LINE_LEFT_PROMPT_LAST_SEGMENT_END_SYMBOL=

  #===============================================================================
  # Segment Styling - Customize Appearance of Individual Segments
  #===============================================================================

  #################################[ os_icon: os identifier ]##################################
  # OS identifier colors.
  typeset -g POWERLEVEL9K_OS_ICON_FOREGROUND=232 # Black
  typeset -g POWERLEVEL9K_OS_ICON_BACKGROUND=7   # White
  # Custom icon for Termux/Android (using Nerd Font Android icon).
  # You might need to find the exact codepoint in your specific Nerd Font version.
  # Check https://www.nerdfonts.com/cheat-sheet
  typeset -g POWERLEVEL9K_OS_ICON_CONTENT_EXPANSION='\uF17B' # nf-fa-android

  ################################[ prompt_char: prompt symbol ]################################
  # Transparent background for the prompt character.
  typeset -g POWERLEVEL9K_PROMPT_CHAR_BACKGROUND=
  # Color on success (OK).
  typeset -g POWERLEVEL9K_PROMPT_CHAR_OK_{VIINS,VICMD,VIVIS,VIOWR}_FOREGROUND=76 # Green
  # Color on error (ERROR).
  typeset -g POWERLEVEL9K_PROMPT_CHAR_ERROR_{VIINS,VICMD,VIVIS,VIOWR}_FOREGROUND=196 # Red
  # Prompt symbols for different states (VI Insert, Command, Visual, Overwrite).
  typeset -g POWERLEVEL9K_PROMPT_CHAR_{OK,ERROR}_VIINS_CONTENT_EXPANSION='❯'
  typeset -g POWERLEVEL9K_PROMPT_CHAR_{OK,ERROR}_VICMD_CONTENT_EXPANSION='❮'
  typeset -g POWERLEVEL9K_PROMPT_CHAR_{OK,ERROR}_VIVIS_CONTENT_EXPANSION='V'
  typeset -g POWERLEVEL9K_PROMPT_CHAR_{OK,ERROR}_VIOWR_CONTENT_EXPANSION='▶'
  typeset -g POWERLEVEL9K_PROMPT_CHAR_OVERWRITE_STATE=true
  # Visual connection settings when prompt_char is at the edge.
  typeset -g POWERLEVEL9K_PROMPT_CHAR_LEFT_PROMPT_LAST_SEGMENT_END_SYMBOL=
  typeset -g POWERLEVEL9K_PROMPT_CHAR_LEFT_PROMPT_FIRST_SEGMENT_START_SYMBOL=
  # Remove surrounding whitespace for compactness.
  typeset -g POWERLEVEL9K_PROMPT_CHAR_LEFT_{LEFT,RIGHT}_WHITESPACE=

  ##################################[ dir: current directory ]##################################
  # Directory colors.
  typeset -g POWERLEVEL9K_DIR_BACKGROUND=4   # Blue
  typeset -g POWERLEVEL9K_DIR_FOREGROUND=254 # Light grey
  # Shortening strategy: Shorten deep paths, keeping unique prefixes.
  typeset -g POWERLEVEL9K_SHORTEN_STRATEGY=truncate_to_unique
  # Symbol indicating shortened path segments (empty = no symbol).
  typeset -g POWERLEVEL9K_SHORTEN_DELIMITER=
  # Colors for shortened/anchor parts of the path.
  typeset -g POWERLEVEL9K_DIR_SHORTENED_FOREGROUND=250 # Medium grey
  typeset -g POWERLEVEL9K_DIR_ANCHOR_FOREGROUND=255   # White
  # Make anchor directories bold.
  typeset -g POWERLEVEL9K_DIR_ANCHOR_BOLD=true
  # Anchor files: Directories containing these files won't be shortened as much.
  local anchor_files=(
    .bzr .citc .git .hg .node-version .python-version .go-version .ruby-version .lua-version
    .java-version .perl-version .php-version .tool-versions .envrc .env .venv Pipfile pyproject.toml
    .mise.toml .shorten_folder_marker .svn .terraform Cargo.toml composer.json go.mod package.json stack.yaml Makefile
  )
  typeset -g POWERLEVEL9K_SHORTEN_FOLDER_MARKER="(${(j:|:)anchor_files})"
  # Strategy for shortening based on markers (false = disabled).
  typeset -g POWERLEVEL9K_DIR_TRUNCATE_BEFORE_MARKER=false
  # Number of trailing directory segments to always keep visible.
  typeset -g POWERLEVEL9K_SHORTEN_DIR_LENGTH=1
  # Maximum directory length before shortening (tuned slightly for Termux).
  typeset -g POWERLEVEL9K_DIR_MAX_LENGTH=70 # Reduced from 80
  # Minimum columns/percentage to leave for command input.
  typeset -g POWERLEVEL9K_DIR_MIN_COMMAND_COLUMNS=40
  typeset -g POWERLEVEL9K_DIR_MIN_COMMAND_COLUMNS_PCT=50
  # Enable directory hyperlink (clickable link in supported terminals). Recommended for Termux.
  typeset -g POWERLEVEL9K_DIR_HYPERLINK=true
  # Show lock icon for non-writable directories.
  typeset -g POWERLEVEL9K_DIR_SHOW_WRITABLE=v3
  # Default lock icon (can be customized).
  # typeset -g POWERLEVEL9K_LOCK_ICON='🔒' # Example: nf-fa-lock
  # Define custom styles for specific directories (e.g., ~/work). Example commented out.
  # typeset -g POWERLEVEL9K_DIR_CLASSES=('~/work(|/*)' WORK '' '~(|/*)' HOME '' '*' DEFAULT '')
  # typeset -g POWERLEVEL9K_DIR_WORK_BACKGROUND=...

  #####################################[ vcs: git status ]######################################
  # VCS background colors based on repo state.
  typeset -g POWERLEVEL9K_VCS_CLEAN_BACKGROUND=2      # Green
  typeset -g POWERLEVEL9K_VCS_MODIFIED_BACKGROUND=3   # Yellow
  typeset -g POWERLEVEL9K_VCS_UNTRACKED_BACKGROUND=2  # Green (same as clean in original)
  typeset -g POWERLEVEL9K_VCS_CONFLICTED_BACKGROUND=3 # Yellow (same as modified in original)
  typeset -g POWERLEVEL9K_VCS_LOADING_BACKGROUND=8    # Dark Grey

  # Branch icon (Powerline symbol).
  typeset -g POWERLEVEL9K_VCS_BRANCH_ICON='\uE0A0 ' # nf-pl-branch ' '

  # Untracked files icon.
  typeset -g POWERLEVEL9K_VCS_UNTRACKED_ICON='?'

  # Custom Git status formatter function (from original config).
  function my_git_formatter() {
    emulate -L zsh

    if [[ -n $P9K_CONTENT ]]; then
      # Use P9K_CONTENT if gitstatus hasn't run yet (e.g., "loading" or from vcs_info).
      typeset -g my_git_format=$P9K_CONTENT
      return
    fi

    # Define colors for different parts of the status string.
    local       meta='%7F' # White foreground (for separators like ':', '@')
    local      clean='%0F' # Black foreground (on clean/untracked background)
    local   modified='%0F' # Black foreground (on modified/conflicted background)
    local  untracked='%0F' # Black foreground (on clean/untracked background)
    local conflicted='%1F' # Red foreground (for actions like 'merge')

    local res # String to build the result

    # Branch name
    if [[ -n $VCS_STATUS_LOCAL_BRANCH ]]; then
      local branch=${(V)VCS_STATUS_LOCAL_BRANCH}
      # Truncate long branch names (show first 12 ... last 12). Remove next line to disable.
      (( $#branch > 32 )) && branch[13,-13]="…"
      res+="${clean}${(g::)POWERLEVEL9K_VCS_BRANCH_ICON}${branch//\%/%%}"
    fi

    # Tag name (only shown if not on a branch, remove `&& -z ...` to always show)
    if [[ -n $VCS_STATUS_TAG && -z $VCS_STATUS_LOCAL_BRANCH ]]; then
      local tag=${(V)VCS_STATUS_TAG}
      # Truncate long tag names. Remove next line to disable.
      (( $#tag > 32 )) && tag[13,-13]="…"
      res+="${meta}#${clean}${tag//\%/%%}"
    fi

    # Commit SHA (only shown if no branch and no tag, remove `[[ -z ... ]] &&` to always show)
    [[ -z $VCS_STATUS_LOCAL_BRANCH && -z $VCS_STATUS_TAG ]] &&
      res+="${meta}@${clean}${VCS_STATUS_COMMIT[1,8]}" # Show first 8 chars

    # Remote tracking branch name (if different from local)
    if [[ -n ${VCS_STATUS_REMOTE_BRANCH:#$VCS_STATUS_LOCAL_BRANCH} ]]; then
      res+="${meta}:${clean}${(V)VCS_STATUS_REMOTE_BRANCH//\%/%%}"
    fi

    # "wip" marker if commit message contains "wip" or "WIP"
    if [[ $VCS_STATUS_COMMIT_SUMMARY == (|*[^[:alnum:]])(wip|WIP)(|[^[:alnum:]]*) ]]; then
      res+=" ${modified}wip"
    fi

    # Upstream remote status (ahead/behind)
    if (( VCS_STATUS_COMMITS_AHEAD || VCS_STATUS_COMMITS_BEHIND )); then
      (( VCS_STATUS_COMMITS_BEHIND )) && res+=" ${clean}⇣${VCS_STATUS_COMMITS_BEHIND}" # Behind: ⇣N
      (( VCS_STATUS_COMMITS_AHEAD && !VCS_STATUS_COMMITS_BEHIND )) && res+=" "
      (( VCS_STATUS_COMMITS_AHEAD  )) && res+="${clean}⇡${VCS_STATUS_COMMITS_AHEAD}" # Ahead: ⇡N
    elif [[ -n $VCS_STATUS_REMOTE_BRANCH ]]; then
      # Optional: Show '=' if up-to-date with remote. Uncomment next line to enable.
      # res+=" ${clean}="
    fi

    # Push remote status (ahead/behind)
    (( VCS_STATUS_PUSH_COMMITS_BEHIND )) && res+=" ${clean}⇠${VCS_STATUS_PUSH_COMMITS_BEHIND}" # Behind push: ⇠N
    (( VCS_STATUS_PUSH_COMMITS_AHEAD && !VCS_STATUS_PUSH_COMMITS_BEHIND )) && res+=" "
    (( VCS_STATUS_PUSH_COMMITS_AHEAD  )) && res+="${clean}⇢${VCS_STATUS_PUSH_COMMITS_AHEAD}" # Ahead push: ⇢N

    # Stash count
    (( VCS_STATUS_STASHES        )) && res+=" ${clean}*${VCS_STATUS_STASHES}" # Stashes: *N

    # Action status (e.g., 'merge', 'rebase')
    [[ -n $VCS_STATUS_ACTION     ]] && res+=" ${conflicted}${VCS_STATUS_ACTION}"

    # File status counts
    (( VCS_STATUS_NUM_CONFLICTED )) && res+=" ${conflicted}~${VCS_STATUS_NUM_CONFLICTED}" # Conflicted: ~N
    (( VCS_STATUS_NUM_STAGED     )) && res+=" ${modified}+${VCS_STATUS_NUM_STAGED}"     # Staged: +N
    (( VCS_STATUS_NUM_UNSTAGED   )) && res+=" ${modified}!${VCS_STATUS_NUM_UNSTAGED}"   # Unstaged: !N
    # Untracked files count. Remove next line to hide.
    (( VCS_STATUS_NUM_UNTRACKED  )) && res+=" ${untracked}${(g::)POWERLEVEL9K_VCS_UNTRACKED_ICON}${VCS_STATUS_NUM_UNTRACKED}" # Untracked: ?N

    # Indicator if unstaged file count is unknown (due to performance settings)
    (( VCS_STATUS_HAS_UNSTAGED == -1 )) && res+=" ${modified}─"

    typeset -g my_git_format=$res # Store the final formatted string
  }
  functions -M my_git_formatter 2>/dev/null # Make function available for expansion

  # Performance threshold for counting dirty files in large repos (-1 = infinity/no limit).
  typeset -g POWERLEVEL9K_VCS_MAX_INDEX_SIZE_DIRTY=-1

  # Disable VCS prompt within home directory for performance.
  typeset -g POWERLEVEL9K_VCS_DISABLED_WORKDIR_PATTERN='~'

  # Use the custom formatter defined above.
  typeset -g POWERLEVEL9K_VCS_DISABLE_GITSTATUS_FORMATTING=true
  typeset -g POWERLEVEL9K_VCS_CONTENT_EXPANSION='${$((my_git_formatter()))+${my_git_format}}'
  # Show counts for file statuses (-1 means show number, 0 means show symbol only, N means show if <= N).
  typeset -g POWERLEVEL9K_VCS_{STAGED,UNSTAGED,UNTRACKED,CONFLICTED,COMMITS_AHEAD,COMMITS_BEHIND}_MAX_NUM=-1

  # Disable the default icon for VCS segment (handled within the formatter).
  typeset -g POWERLEVEL9K_VCS_VISUAL_IDENTIFIER_EXPANSION=
  # Optional prefix (e.g., 'on ').
  # typeset -g POWERLEVEL9K_VCS_PREFIX='on '

  # Supported VCS backends. Add 'svn' or 'hg' if needed, but might impact performance.
  typeset -g POWERLEVEL9K_VCS_BACKENDS=(git)

  ##########################[ status: exit code of the last command ]###########################
  # Enable more granular status states (PIPE, SIGNAL).
  typeset -g POWERLEVEL9K_STATUS_EXTENDED_STATES=true

  # Hide status indicator on success (prompt_char color indicates this).
  typeset -g POWERLEVEL9K_STATUS_OK=false
  typeset -g POWERLEVEL9K_STATUS_OK_VISUAL_IDENTIFIER_EXPANSION='✔'
  typeset -g POWERLEVEL9K_STATUS_OK_FOREGROUND=2
  typeset -g POWERLEVEL9K_STATUS_OK_BACKGROUND=0

  # Show status indicator if a pipe command succeeded overall but had internal errors (e.g., `grep foo file | true`).
  typeset -g POWERLEVEL9K_STATUS_OK_PIPE=true
  typeset -g POWERLEVEL9K_STATUS_OK_PIPE_VISUAL_IDENTIFIER_EXPANSION='✔' # Consider '✔~' or similar?
  typeset -g POWERLEVEL9K_STATUS_OK_PIPE_FOREGROUND=2 # Green
  typeset -g POWERLEVEL9K_STATUS_OK_PIPE_BACKGROUND=0 # Black

  # Hide simple error status (prompt_char color indicates this).
  typeset -g POWERLEVEL9K_STATUS_ERROR=false
  typeset -g POWERLEVEL9K_STATUS_ERROR_VISUAL_IDENTIFIER_EXPANSION='✘'
  typeset -g POWERLEVEL9K_STATUS_ERROR_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_STATUS_ERROR_BACKGROUND=1 # Red

  # Show status indicator if the last command was killed by a signal.
  typeset -g POWERLEVEL9K_STATUS_ERROR_SIGNAL=true
  typeset -g POWERLEVEL9K_STATUS_VERBOSE_SIGNAME=false # Show short signal names (e.g., INT vs SIGINT(2))
  typeset -g POWERLEVEL9K_STATUS_ERROR_SIGNAL_VISUAL_IDENTIFIER_EXPANSION='✘' # Consider '⚡' or similar? nf-fa-bolt \uF0E7
  typeset -g POWERLEVEL9K_STATUS_ERROR_SIGNAL_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_STATUS_ERROR_SIGNAL_BACKGROUND=1 # Red

  # Show status indicator if a pipe command failed overall and had internal errors.
  typeset -g POWERLEVEL9K_STATUS_ERROR_PIPE=true
  typeset -g POWERLEVEL9K_STATUS_ERROR_PIPE_VISUAL_IDENTIFIER_EXPANSION='✘' # Consider '✘~' or similar?
  typeset -g POWERLEVEL9K_STATUS_ERROR_PIPE_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_STATUS_ERROR_PIPE_BACKGROUND=1 # Red

  ###################[ command_execution_time: duration of the last command ]###################
  # Execution time colors.
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_FOREGROUND=0 # Black
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_BACKGROUND=3 # Yellow
  # Show duration only if command takes at least 3 seconds.
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_THRESHOLD=3
  # Round to whole seconds.
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_PRECISION=0
  # Format: 1d 2h 3m 4s.
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_FORMAT='d h m s'
  # Icon disabled, using text only. Use '󰔟 ' (nf-mdi-timer_sand) or similar if desired.
  typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_VISUAL_IDENTIFIER_EXPANSION=
  # typeset -g POWERLEVEL9K_COMMAND_EXECUTION_TIME_PREFIX='took '

  #######################[ background_jobs: presence of background jobs ]#######################
  # Background jobs colors.
  typeset -g POWERLEVEL9K_BACKGROUND_JOBS_FOREGROUND=6 # Cyan
  typeset -g POWERLEVEL9K_BACKGROUND_JOBS_BACKGROUND=0 # Black
  # Show only an icon, not the count of jobs.
  typeset -g POWERLEVEL9K_BACKGROUND_JOBS_VERBOSE=false
  # Icon for background jobs. '󰌅 ' (nf-mdi-cog_counterclockwise)
  typeset -g POWERLEVEL9K_BACKGROUND_JOBS_VISUAL_IDENTIFIER_EXPANSION='󰌅 '

  ##################################[ context: user@hostname ]##################################
  # Context colors for different states (root, remote, local).
  typeset -g POWERLEVEL9K_CONTEXT_ROOT_FOREGROUND=1   # Red
  typeset -g POWERLEVEL9K_CONTEXT_ROOT_BACKGROUND=0   # Black
  typeset -g POWERLEVEL9K_CONTEXT_{REMOTE,REMOTE_SUDO}_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_CONTEXT_{REMOTE,REMOTE_SUDO}_BACKGROUND=0 # Black
  typeset -g POWERLEVEL9K_CONTEXT_FOREGROUND=3        # Yellow (used for remote)
  typeset -g POWERLEVEL9K_CONTEXT_BACKGROUND=0        # Black
  # Format template: user@hostname.
  typeset -g POWERLEVEL9K_CONTEXT_TEMPLATE='%n@%m'
  # Hide context segment unless root or SSH (saves space). Remove next line to always show.
  typeset -g POWERLEVEL9K_CONTEXT_{DEFAULT,SUDO}_{CONTENT,VISUAL_IDENTIFIER}_EXPANSION=
  # Icon. Consider '󰣘 ' (nf-mdi-monitor_shimmer) for remote, '󱄂 ' (nf-mdi-skull_crossbones) for root?
  # typeset -g POWERLEVEL9K_CONTEXT_ROOT_VISUAL_IDENTIFIER_EXPANSION='󱄂 '
  # typeset -g POWERLEVEL9K_CONTEXT_REMOTE_VISUAL_IDENTIFIER_EXPANSION='󰣘 '
  # typeset -g POWERLEVEL9K_CONTEXT_PREFIX='at '

  ####################################[ time: current time ]####################################
  # Time colors.
  typeset -g POWERLEVEL9K_TIME_FOREGROUND=0 # Black
  typeset -g POWERLEVEL9K_TIME_BACKGROUND=7 # White
  # Time format: 24-hour HH:MM:SS. Use '%I:%M:%S %p' for 12-hour AM/PM.
  typeset -g POWERLEVEL9K_TIME_FORMAT='%H:%M:%S'
  # Update time only when displaying prompt, not on command execution.
  typeset -g POWERLEVEL9K_TIME_UPDATE_ON_COMMAND=false
  # Icon. '󱑍 ' (nf-mdi-clock_time_eight_outline) or '󰥔 ' (nf-mdi-clock)
  typeset -g POWERLEVEL9K_TIME_VISUAL_IDENTIFIER_EXPANSION='󰥔 '
  # typeset -g POWERLEVEL9K_TIME_PREFIX='at '

  ################################[ battery: internal battery ]#################################
  # Battery colors and threshold for low level.
  typeset -g POWERLEVEL9K_BATTERY_LOW_THRESHOLD=20
  typeset -g POWERLEVEL9K_BATTERY_LOW_FOREGROUND=1         # Red
  typeset -g POWERLEVEL9K_BATTERY_{CHARGING,CHARGED}_FOREGROUND=2 # Green
  typeset -g POWERLEVEL9K_BATTERY_DISCONNECTED_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_BATTERY_BACKGROUND=0             # Black (match context/jobs)
  # Battery icons from low to high charge. Use Nerd Font battery icons.
  # Example: nf-mdi-battery_alert, _low, _medium, _high, _charging
  typeset -g POWERLEVEL9K_BATTERY_STAGES=('\uf244' '\uf243' '\uf242' '\uf241' '\uf240') # nf-fa-battery_empty to _full
  # Icons for charging/charged states.
  typeset -g POWERLEVEL9K_BATTERY_CHARGING_VISUAL_IDENTIFIER_EXPANSION='\uf0e7' # nf-fa-bolt (charging)
  typeset -g POWERLEVEL9K_BATTERY_CHARGED_VISUAL_IDENTIFIER_EXPANSION='\uf240' # nf-fa-battery_full (charged)
  # Don't show estimated time remaining/to charge.
  typeset -g POWERLEVEL9K_BATTERY_VERBOSE=false

  #===============================================================================
  # Language Version Managers & Environments Styling
  # (Keep defaults, mostly hidden if matching global/system version)
  #===============================================================================

  ###[ virtualenv: python virtual environment ]###
  typeset -g POWERLEVEL9K_VIRTUALENV_FOREGROUND=0
  typeset -g POWERLEVEL9K_VIRTUALENV_BACKGROUND=4 # Blue
  typeset -g POWERLEVEL9K_VIRTUALENV_SHOW_PYTHON_VERSION=false
  typeset -g POWERLEVEL9K_VIRTUALENV_SHOW_WITH_PYENV=false # Hide if pyenv is shown
  typeset -g POWERLEVEL9K_VIRTUALENV_{LEFT,RIGHT}_DELIMITER=
  # typeset -g POWERLEVEL9K_VIRTUALENV_VISUAL_IDENTIFIER_EXPANSION='󰌠 ' # nf-mdi-language_python

  ###[ pyenv: python environment (pyenv) ]###
  typeset -g POWERLEVEL9K_PYENV_FOREGROUND=0
  typeset -g POWERLEVEL9K_PYENV_BACKGROUND=4 # Blue
  typeset -g POWERLEVEL9K_PYENV_SOURCES=(shell local global)
  typeset -g POWERLEVEL9K_PYENV_PROMPT_ALWAYS_SHOW=false # Hide if same as global
  typeset -g POWERLEVEL9K_PYENV_SHOW_SYSTEM=true
  typeset -g POWERLEVEL9K_PYENV_CONTENT_EXPANSION='${P9K_CONTENT}${${P9K_CONTENT:#$P9K_PYENV_PYTHON_VERSION(|/*)}:+ $P9K_PYENV_PYTHON_VERSION}'
  # typeset -g POWERLEVEL9K_PYENV_VISUAL_IDENTIFIER_EXPANSION='󰌠 ' # nf-mdi-language_python

  ###[ nvm: node.js version (nvm) ]###
  typeset -g POWERLEVEL9K_NVM_FOREGROUND=0
  typeset -g POWERLEVEL9K_NVM_BACKGROUND=2 # Green
  typeset -g POWERLEVEL9K_NVM_PROMPT_ALWAYS_SHOW=false # Hide if same as default
  typeset -g POWERLEVEL9K_NVM_SHOW_SYSTEM=true
  # typeset -g POWERLEVEL9K_NVM_VISUAL_IDENTIFIER_EXPANSION='󰎙 ' # nf-mdi-nodejs

  # --- Add other language managers (asdf, rbenv, goenv, etc.) here if needed ---
  # --- Use the same pattern of setting FOREGROUND, BACKGROUND, PROMPT_ALWAYS_SHOW ---

  # Example: ASDF general settings (tool-specific overrides exist in original config)
  typeset -g POWERLEVEL9K_ASDF_FOREGROUND=0
  typeset -g POWERLEVEL9K_ASDF_BACKGROUND=7 # White
  typeset -g POWERLEVEL9K_ASDF_SOURCES=(shell local global)
  typeset -g POWERLEVEL9K_ASDF_PROMPT_ALWAYS_SHOW=false # Hide if same as global
  typeset -g POWERLEVEL9K_ASDF_SHOW_SYSTEM=true
  typeset -g POWERLEVEL9K_ASDF_SHOW_ON_UPGLOB=

  #===============================================================================
  # Tool Specific Styling (Keep essential ones, comment out others for Termux)
  #===============================================================================

  ###[ vim_shell: vim shell indicator (:sh) ]###
  typeset -g POWERLEVEL9K_VIM_SHELL_FOREGROUND=0
  typeset -g POWERLEVEL9K_VIM_SHELL_BACKGROUND=2 # Green
  # typeset -g POWERLEVEL9K_VIM_SHELL_VISUAL_IDENTIFIER_EXPANSION=' ' # nf-dev-vim

  ###[ midnight_commander: mc shell ]###
  typeset -g POWERLEVEL9K_MIDNIGHT_COMMANDER_FOREGROUND=3 # Yellow
  typeset -g POWERLEVEL9K_MIDNIGHT_COMMANDER_BACKGROUND=0 # Black
  # typeset -g POWERLEVEL9K_MIDNIGHT_COMMANDER_VISUAL_IDENTIFIER_EXPANSION='󰉋 ' # nf-mdi-folder_cog_outline

  ###[ kubecontext: kubernetes context ]###
  # Show only when running specific commands. Remove line to always show.
  typeset -g POWERLEVEL9K_KUBECONTEXT_SHOW_ON_COMMAND='kubectl|helm|kubens|kubectx|oc|istioctl|kogito|k9s|helmfile|flux|fluxctl|stern|kubeseal|skaffold|kubent|kubecolor|cmctl|sparkctl'
  typeset -g POWERLEVEL9K_KUBECONTEXT_CLASSES=('*' DEFAULT) # Default class
  typeset -g POWERLEVEL9K_KUBECONTEXT_DEFAULT_FOREGROUND=7 # White
  typeset -g POWERLEVEL9K_KUBECONTEXT_DEFAULT_BACKGROUND=5 # Magenta
  # typeset -g POWERLEVEL9K_KUBECONTEXT_DEFAULT_VISUAL_IDENTIFIER_EXPANSION='󱃾 ' # nf-mdi-kubernetes
  # Content: Cluster name (or full context name as fallback), then /namespace (if not default)
  typeset -g POWERLEVEL9K_KUBECONTEXT_DEFAULT_CONTENT_EXPANSION=
  POWERLEVEL9K_KUBECONTEXT_DEFAULT_CONTENT_EXPANSION+='${P9K_KUBECONTEXT_CLOUD_CLUSTER:-${P9K_KUBECONTEXT_NAME}}'
  POWERLEVEL9K_KUBECONTEXT_DEFAULT_CONTENT_EXPANSION+='${${:-/$P9K_KUBECONTEXT_NAMESPACE}:#/default}'
  # typeset -g POWERLEVEL9K_KUBECONTEXT_PREFIX='⛵ ' # nf-fa-ship

  ###[ aws: aws profile ]###
  # Show only when running specific commands. Remove line to always show.
  typeset -g POWERLEVEL9K_AWS_SHOW_ON_COMMAND='aws|awless|cdk|terraform|pulumi|terragrunt'
  typeset -g POWERLEVEL9K_AWS_CLASSES=('*' DEFAULT) # Default class
  typeset -g POWERLEVEL9K_AWS_DEFAULT_FOREGROUND=7 # White
  typeset -g POWERLEVEL9K_AWS_DEFAULT_BACKGROUND=208 # Orange
  # typeset -g POWERLEVEL9K_AWS_DEFAULT_VISUAL_IDENTIFIER_EXPANSION='󰸏 ' # nf-mdi-aws
  typeset -g POWERLEVEL9K_AWS_CONTENT_EXPANSION='${P9K_AWS_PROFILE//\%/%%}${P9K_AWS_REGION:+ ${P9K_AWS_REGION//\%/%%}}'

  # --- Add other tool segments (direnv, terraform, gcloud, etc.) here if needed ---
  # --- Refer to the original commented-out sections for their settings ---

  #===============================================================================
  # Advanced Features - Transient Prompt, Instant Prompt, Hot Reload
  #===============================================================================

  # Transient prompt: Trim prompt after command acceptance.
  #   - off:      Disabled.
  #   - always:   Always trim.
  #   - same-dir: Trim unless directory changed.
  typeset -g POWERLEVEL9K_TRANSIENT_PROMPT=always

  # Instant prompt: Show prompt instantly, even before zsh fully loads.
  # Requires careful management of .zshrc output.
  #   - off:     Disabled.
  #   - quiet:   Enabled, no warnings about init output (recommended if setup is clean).
  #   - verbose: Enabled, shows warnings.
  typeset -g POWERLEVEL9K_INSTANT_PROMPT=quiet

  # Hot reload: Allow changing POWERLEVEL9K variables in the shell to take effect immediately.
  # Slight performance cost (1-2ms). Disabled by default for speed.
  typeset -g POWERLEVEL9K_DISABLE_HOT_RELOAD=true

  #===============================================================================
  # User Defined Segments (Example)
  #===============================================================================

  # Example user-defined segment function.
  # Add 'example' to POWERLEVEL9K_LEFT/RIGHT_PROMPT_ELEMENTS to activate.
  # function prompt_example() {
  #   p10k segment -f yellow -b red -i '⭐' -t 'Hi, %n!'
  # }
  # function instant_prompt_example() {
  #   prompt_example
  # }
  # typeset -g POWERLEVEL9K_EXAMPLE_FOREGROUND=3
  # typeset -g POWERLEVEL9K_EXAMPLE_BACKGROUND=1

  #===============================================================================
  # Finalization
  #===============================================================================

  # Reload Powerlevel10k if it's already loaded.
  (( ! $+functions[p10k] )) || p10k reload
}

# Tell `p10k configure` which file it should overwrite.
typeset -g POWERLEVEL9K_CONFIG_FILE=${${(%):-%x}:a}

# Restore original zsh options.
(( ${#p10k_config_opts} )) && setopt ${p10k_config_opts[@]}
'builtin' 'unset' 'p10k_config_opts'

# vim: ft=zsh
