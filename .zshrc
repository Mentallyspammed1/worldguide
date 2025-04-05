# ~/.zshrc - Configuration for Termux - Enhanced and Optimized (Wizard Edition ✨)

# This configuration elevates your Termux Zsh experience to a new level of wizardry! 🧙‍♂️
# Key Features:
# - Lightning-fast shell startup with Powerlevel10k instant prompt. ⚡
# - Clean and organized configuration using XDG base directories. 📂
# - Enhanced shell options for maximum productivity and usability. 🚀
# - Robust and speedy plugin management with zinit. ⚙️
# - Comprehensive completion system for effortless command execution. ✍️
# - A plethora of useful aliases and functions for common and Termux-specific tasks. 🛠️
# - AI-powered command suggestions and text summarization via aichat. 🤖
# - Streamlined navigation and editing with custom keybindings. ⌨️
# - Optional, secure command history encryption using GPG. 🛡️
# - Modular and well-commented structure for easy customization. 🧩

# --- Powerlevel10k Instant Prompt ---
# Supercharge shell startup by caching the initial prompt. 🏎️
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# --- Environment Variables ---
export TERM="xterm-256color"       # Rock-solid 256-color support for Termux. Essential for vibrant themes! 🌈
export LANG="en_US.UTF-8"          # UTF-8 encoding and English locale for universal character support. 🌍
# export SHELL="/data/data/com.termux/files/usr/bin/zsh" # Shell path (usually auto-detected, only needed in rare cases).
export EDITOR="nvim"              # Neovim is the preferred editor if available, otherwise falls back to 'nano' in functions/aliases. 📝
command -v nvim >/dev/null 2>&1 || export EDITOR="nano" # Fallback to nano if nvim is not found.
export VISUAL="$EDITOR"            # Visual editor, often defaults to the same as EDITOR.

# --- XDG Base Directory Support for Zsh Configuration ---
# Keep your home directory pristine by centralizing Zsh config in XDG directories. ✨
# IMPORTANT: Move your `.zshrc` to `~/.config/zsh/.zshrc` (or `$XDG_CONFIG_HOME/zsh/.zshrc` if set).
export ZDOTDIR="${XDG_CONFIG_HOME:-$HOME/.config}/zsh" # Main Zsh config directory, defaults to ~/.config/zsh.
export ZSHRC="$ZDOTDIR/.zshrc"     # Path to your Zsh configuration file.

# --- Zsh Cache and State Directories (XDG Compliant) ---
# Store Zsh cache and state data in XDG directories for a cleaner system. 🧹
export ZSH_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/zsh" # Cache directory for zcompdump, zinit, etc.
export ZSH_COMPDUMP="${ZSH_CACHE_DIR}/zcompdump"         # Path to the Zsh completion dump file (performance boost!).
export HISTFILE="${XDG_STATE_HOME:-$HOME/.local/state}/zsh/history_encrypted" # Encrypted Zsh history path in XDG state directory. 🔒

# Create essential directories if they don't exist (silent and robust). 🛠️
mkdir -p "$ZSH_CACHE_DIR" "$(dirname "$HISTFILE")"

# --- Zsh Options (setopt) ---
# Fine-tune Zsh behavior for a smoother, more powerful shell experience. ⚙️
# Explore `man zshoptions` for the full spellbook of options! 📖

setopt auto_cd                # Magically `cd` into directories without typing 'cd'. ✨
setopt auto_pushd             # Automatically remember previous directories with `cd` (use `dirs`, `popd` to navigate). 🚶
setopt pushd_ignore_dups      # Keep your directory stack clean, no duplicate entries. 🧹
setopt correct                # Zsh will suggest corrections for mistyped commands. Did you mean...? 🤔
setopt numeric_glob_sort      # Sort numbered files naturally (file1, file2, file10). 🔢
setopt no_flow_control        # Disable Ctrl+S/Ctrl+Q (often accidentally freezes terminal). 🧊
setopt extended_glob          # Unleash powerful globbing patterns (`^`, `#`, `~`, etc.). 🌟
setopt glob_dots              # Include hidden files (dotfiles) in directory listings by default. 🕵️‍♂️
setopt interactive_comments   # Add comments to your interactive commands (for complex one-liners). #📝
setopt promptsubst            # Enable expansions in prompts (variables, commands, substitutions - P10k needs this!). 🔮
setopt share_history          # Instantly share history across all Zsh sessions. 🤝
setopt inc_append_history     # History is updated immediately after each command. ✍️
setopt hist_expire_dups_first # Remove duplicate history entries when trimming history. ✂️
setopt hist_ignore_dups       # Don't record consecutive duplicate commands in history. 🤫
setopt hist_ignore_space      # Commands starting with space are not saved (for secrets or throwaways). 🙈
setopt hist_find_no_dups      # History search is cleaner, no adjacent duplicates. 🔍
setopt hist_no_functions      # Keep function definitions out of history (cleaner history). 🧹
# setopt hist_verify            # Show history command before execution (can be verbose).
setopt rc_quotes              # Embrace 'foo'bar' style quoting. 🎸

# --- History Settings ---
# Control your command history - remember the past, but not *everything*. 🕰️
export HISTSIZE=50000         # Keep a generous 50,000 lines in memory history. 🧠
export SAVEHIST=50000         # Save up to 50,000 lines to the history file. 💾

# Commands to banish from history (separated by colons). Add your own secret incantations! 🤫
export HIST_IGNORE="ls:cd:pwd:exit:history:bg:fg:jobs:clear:htop:ncdu:lazygit:lazydocker:man:df:free:top"

# --- PATH Configuration ---
# Forge your command search path, ensuring your tools are always within reach. 🧭
# Order matters: first paths have priority. 🥇

# Standard user binary directories (check if they exist before adding). 🛡️
[[ -d "$HOME/bin" ]] && PATH="$HOME/bin:$PATH"
[[ -d "$HOME/.local/bin" ]] && PATH="$HOME/.local/bin:$PATH"

# Termux default binaries (always essential). 🧰
[[ -d "/data/data/com.termux/files/usr/bin" ]] && PATH="/data/data/com.termux/files/usr/bin:$PATH"

# Rust/Cargo binaries (for Rustaceans 🦀).
[[ -d "$HOME/.cargo/bin" ]] && PATH="$PATH:$HOME/.cargo/bin"

# Android platform-tools (ADB, fastboot - for Android wizards 📱).
[[ -d "$HOME/platform-tools" ]] && PATH="$PATH:$HOME/platform-tools"

# Ensure PATH is unique (remove duplicates, for extra cleanliness - requires modern Zsh). 🧼
# typeset -U path # Uncomment if you encounter duplicate paths.

export PATH # Export the carefully crafted PATH. 📤

# --- Function Path (fpath) Configuration ---
# Expand Zsh's function autoloading paths. Create `~/.config/zsh/functions` for your custom spells. 📜
# fpath=($ZDOTDIR/functions $fpath) # Uncomment to activate custom function autoloading.

# --- Oh My Zsh Theme (Powerlevel10k) ---
# Unleash the power of Powerlevel10k for a stunning and informative prompt. ✨
# Falls gracefully back to Agnoster if P10k is not found (but P10k is highly recommended!).

THEME_DIR="${ZSH_CUSTOM:-${XDG_DATA_HOME:-$HOME/.local/share}/oh-my-zsh/custom}/themes/powerlevel10k" # XDG-compliant theme directory.
THEME_FILE="$THEME_DIR/powerlevel10k.zsh-theme"

if [[ -f "$THEME_FILE" ]]; then
  ZSH_THEME="powerlevel10k/powerlevel10k" # Engage Powerlevel10k! 🚀
else
  ZSH_THEME="agnoster" # Fallback to Agnoster, still decent, but P10k is better. 🥲
  echo $'\e[33mNotice: Powerlevel10k theme files not found at '"$THEME_FILE"$'. Using fallback theme '\'"$ZSH_THEME"$'\'.\e[0m'
  echo $'\e[36mTo unlock the full prompt potential, install Powerlevel10k:\e[0m'
  echo $'\e[36mgit clone --depth=1 https://github.com/romkatv/powerlevel10k.git "'"$THEME_DIR"'"\e[0m'
  command -v termux-toast >/dev/null && termux-toast -g middle "Install Powerlevel10k for enhanced prompt" # Gentle reminder via toast.
fi

# --- Powerlevel10k Configuration ---
# Load your personalized Powerlevel10k settings. ⚙️
# Customize your prompt by running `p10k configure` (highly recommended!).
# Configuration file is typically at `~/.config/zsh/.p10k.zsh`.
[[ -f "$ZDOTDIR/.p10k.zsh" ]] && source "$ZDOTDIR/.p10k.zsh"

# --- Plugin Manager (zinit) ---
# Zinit: Your trusty spellbook for managing Zsh plugins with speed and grace. 🧙‍♂️✨
# Replaces manual plugin loading, making your shell configuration cleaner and faster.

ZINIT_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/zinit/zinit.git" # Zinit's home in XDG data directory.

# Install zinit if it's not already summoned. 🪄
if [[ ! -d "$ZINIT_HOME" ]]; then
  echo "\e[36mSummoning zinit plugin manager...\e[0m"
  mkdir -p "$(dirname $ZINIT_HOME)"
  git clone https://github.com/zdharma-continuum/zinit.git "$ZINIT_HOME" && \
    echo "\e[32mzinit successfully summoned.\e[0m" || \
    echo "\e[31mWizard's Warning: Failed to summon zinit! Check internet connection and try again.\e[0m"
fi

# Source zinit's magic if the summoning ritual was successful (or it already exists). ✨
if [[ -f "$ZINIT_HOME/zinit.zsh" ]]; then
  source "$ZINIT_HOME/zinit.zsh"

  # --- Load Plugins with Zinit ---
  # Cast spells to load essential plugins. 🧙‍♂️

  # Oh My Zsh library core - the foundation, but skip theme/completion loading (we handle those). 🏛️
  zinit light-mode for \
      ohmyzsh/ohmyzsh

  # Essential Oh My Zsh plugins - carefully selected for maximum utility. 💎
  zinit ice from:ohmyzsh light-mode for \
      plugins/command-not-found # Smart suggestions for misspelled commands. 🤔
      plugins/sudo            # Enhanced `sudo` handling in Termux (crucial for `tsudo`). 🛡️
      plugins/extract         # Handy `extract` and `sextract` aliases for archive management. 📦
      plugins/web-search      # Quick web searches right from your terminal (`google`, `wiki`, etc.). 🌐

  # External plugins - expand your shell's capabilities. 🌟
  # Syntax Highlighting - make your commands visually stunning and error-prone. 🎨
  zinit light-mode for \
      zdharma-continuum/fast-syntax-highlighting

  # Auto Suggestions - anticipate your commands, saving you keystrokes. 🔮
  zinit light-mode for \
      zsh-users/zsh-autosuggestions

  # Enhanced Completions - supercharge Zsh's completion system with more definitions. ✍️
  zinit light-mode for zsh-users/zsh-completions

  # Directory Jumping (`z`) - teleport to frequently used directories in an instant. 💨
  # `nice'10` reduces background priority, `atload` for post-load tweaks.
  zinit ice nice"10" depth"1" # Shallow clone for speed. 🚀
  zinit light-mode for agkozak/zsh-z

  # Optional: Load custom plugins from your magic workshop at `$ZDOTDIR/plugins`. 🛠️
  # zinit light-mode wait'1' lucid for $ZDOTDIR/plugins/my-custom-plugin

  # Initialize the completion system and replay plugin-provided completions. 🪄
  zicompinit
  zicdreplay

else
  echo "\e[31mWizard's Warning: zinit.zsh not found. Plugin loading skipped. Ensure zinit installation.\e[0m"
fi

# --- Zsh Completion System Configuration ---
# Master the art of completion for faster and more accurate command invocation. 🎯
# These settings refine Zsh's completion engine for a superior experience. ⚙️

zstyle ':completion:*' menu select=1                            # Enable interactive completion menu (cycle through options). 🎡
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}' 'r:|=*' 'l:|=* r:|=*' # Case-insensitive, partial-word matching for flexible completion. 🤹
zstyle ':completion:*' use-cache yes                           # Cache completions for blazing speed. ⚡
zstyle ':completion:*' cache-path "$ZSH_CACHE_DIR"             # Where to store the completion cache (XDG compliant!). 📂
zstyle ':completion:*:*:*:*:corrections' format $'\e[31m%d (errors: %e)\e[0m' # Highlight correction suggestions in red. 🚨
zstyle ':completion:*:*:*:*:descriptions' format $'\e[35m%d\e[0m'         # Descriptions in magenta for clarity. 🌸
zstyle ':completion:*:*:*:*:messages' format $'\e[33m%d\e[0m'            # Messages in yellow for attention. 🌟
zstyle ':completion:*:*:*:*:warnings' format $'\e[31mNo matches for: %d\e[0m' # No matches warning in red. ⛔
zstyle ':completion:*:default' list-colors ${(s.:.)LS_COLORS}    # Colorize completions using your `LS_COLORS` (consistent look). 🌈

# --- Initialize Completion System (Handled by Zinit) ---
# Zinit's `zicompinit` and `zicdreplay` handle completion initialization automatically. 🪄
# Manual fallback (if NOT using zinit - uncomment ONLY if you are *not* using zinit).
# if ! command -v zinit > /dev/null 2>&1; then
#   autoload -Uz compinit
#   if [[ ! -f "$ZSH_COMPDUMP" || -z "$ZSH_COMPDUMP"(N.mh+168) ]]; then
#     compinit -i -d "$ZSH_COMPDUMP"
#   else
#     compinit -i -C -d "$ZSH_COMPDUMP"
#   fi
# fi

# --- Aliases ---
# Craft aliases to shorten frequently used commands and boost your workflow. 🚀
# Organized into categories for easy navigation.

# --- Basic Navigation and File Operations --- 🧭
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..' # Deeper navigation shortcuts. 🚶‍♂️
alias -- -='cd -'             # Quickly jump back to the previous directory. ↩️
alias ls='ls --color=auto'    # `ls` with automatic color output. 🌈
alias l='ls -lhF'             # List human-readable, long listing, classified files. 📜
alias la='ls -lhaF'           # List all (including hidden), human-readable, long, classified. 🕵️‍♂️
alias ll='ls -lAhF'           # List almost all (no . or ..), human-readable, long, classified. 📃
alias lt='ls --tree'          # Tree view of directories (if `ls --tree` is supported, otherwise fallback). 🌳
if ! ls --tree > /dev/null 2>&1; then alias lt='tree -C'; fi # Fallback to 'tree' if 'ls --tree' is not available (requires 'tree' package).
alias grep='grep --color=auto' # `grep` with color highlighting for matches. 🔍
alias df='df -h'              # Disk free space in human-readable format. 💾
alias du='du -h'              # Disk usage in human-readable format. 📊
alias mkdir='mkdir -p'        # Create directories recursively (including parents). 📂
alias rmdir='rmdir -p'        # Remove empty directories recursively. 🗑️
alias path='echo -e ${PATH//:/\\n}' # Display PATH variable, one entry per line. 🛣️
alias cp='cp -i'              # `cp` with interactive prompt before overwriting. ✍️
alias mv='mv -i'              # `mv` with interactive prompt before overwriting. ✍️
alias rm='rm -i'              # `rm` with interactive prompt before deleting (safety first!). ⚠️
alias cat='bat --paging=auto --theme=Dracula' # `cat` replacement with syntax highlighting and Dracula theme if bat is installed. 🦇
command -v bat >/dev/null 2>&1 || alias cat='cat' # Fallback to standard `cat` if bat is not found.
alias find='fd'              # `find` replacement with `fd` if installed (faster and more user-friendly). 🔎
command -v fd >/dev/null 2>&1 || alias find='find' # Fallback to standard `find` if fd is not found.

# --- Enhanced Tools (Conditional Aliases - use if available) --- 🛠️
# Modern replacements for core utilities, used if present.

if command -v eza >/dev/null 2>&1; then
  alias ls='eza --group-directories-first --icons' # `eza` as `ls`, group dirs, icons. ✨
  alias l='eza -lhF --git --icons'  # `eza` long listing, git info, icons. 📜
  alias la='eza -lhaF --git --icons' # `eza` all files, git, icons. 🕵️‍♂️
  alias ll='eza -lAhF --git --icons' # `eza` almost all, git, icons. 📃
  alias lt='eza --tree --level=2 --icons' # `eza` tree view, icons. 🌳
fi

# --- Termux Specific Aliases --- 📱
alias pkg='pkg'                             # Termux package manager. 📦
alias sudo='tsudo'                          # Termux `sudo` replacement (OMZ sudo plugin helps manage this). 🛡️
alias update-termux='pkg update && pkg upgrade -y' # Full Termux system update. 🔄
alias open='termux-open'                    # Open files/URLs with Termux opener. 📤
alias share='termux-share'                  # Share files from Termux via Android share sheet. 📲
alias toast='termux-toast'                  # Show Termux toast notifications. 🍞
alias getclip='termux-clipboard-get'        # Get clipboard content. 📋
alias setclip='termux-clipboard-set'        # Set clipboard content. 📝
alias termux-setup-storage='termux-setup-storage' # Grant storage access to Termux. 🗄️
alias termux-reload-settings='termux-reload-settings' # Reload Termux settings. ⚙️

# --- Git Aliases --- 🌿
alias g='git'
alias gs='git status -sb' # Short branch status. 🌿
alias ga='git add'
alias gaa='git add --all'
alias gc='git commit -v'
alias gca='git commit -v -a' # Commit all changes. ✍️
alias gco='git checkout'
alias gb='git branch'
alias gm='git merge'
alias gp='git push'
alias gpf='git push --force-with-lease' # Safer force push. 🚀
alias gpu='git pull --rebase --autostash' # Pull with rebase and autostash. 🔄
alias glog="git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit" # Pretty git log graph. 📈
alias gd='git diff'
alias gdc='git diff --cached' # Diff staged changes. 📜
alias gst='git stash'         # Stash changes. 📦
alias gsta='git stash apply'   # Apply stashed changes. 📤
alias gstp='git stash pop'     # Pop stashed changes. 💥
alias gcl='git clone --depth=1' # Shallow clone for faster cloning. 🚀

# --- Navigation Shortcuts --- 🗺️
alias home='cd ~'            # Go home. 🏠
alias dl='cd ~/Download'     # Go to Downloads. 📥
alias dotfiles='cd $ZDOTDIR' # Go to dotfiles directory. ⚙️

# --- Miscellaneous Aliases --- 🧰
alias vim="$EDITOR"          # Use preferred editor as 'vim'. 📝
alias vi="$EDITOR"           # Use preferred editor as 'vi'. 📝
alias c='clear'
alias h='history'
alias j='jobs -l'
alias reload='source $ZSHRC' # Reload Zsh config. 🔄
alias ip='myip'              # Shorter alias for public IP function. 🌐
alias myip='curl -s ifconfig.me || curl -s ipinfo.io/ip' # Get public IP address. 🌐
alias ports='netstat -tulnp' # List listening ports (if net-tools is installed). 👂
alias now='date "+%Y-%m-%d %H:%M:%S"' # Current date and time. ⏱️
alias today='date "+%Y-%m-%d"'      # Current date. 📅
alias tomorrow='date -d tomorrow "+%Y-%m-%d"' # Tomorrow's date. 📅
alias yesterday='date -d yesterday "+%Y-%m-%d"' # Yesterday's date. 📅
alias fixperms='chmod -R go-w ~' # Secure home directory permissions. 🔒
alias flushdns='rm -f /data/data/com.termux/files/usr/etc/resolv.conf && rm -f /etc/resolv.conf && pkill -HUP -x dnsmasq' # Flush DNS cache (Termux specific, may require root/tsudo). 🌐

# Calculate directory sizes, sorted by size (human-readable). 📊
alias dus='du -hs * .[!.]* | sort -rh' # Summary of current level, sorted by size. 📊
alias duu='du -shc * .[!.]* | sort -hr' # Total size of dirs/files, sorted by size. 📈

# --- Custom Functions ---
# Define functions for complex or repetitive tasks. 🧙‍♂️
# Functions are designed for robustness with error handling and informative messages.

# --- Helper Function: Check if a command exists ---
# Internal function to verify command availability. 🛡️
_command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Function: Search man pages by keyword ---
# Quickly find relevant man pages using `man -k`. 📖
man-search() {
  if ! _command_exists man; then echo "\e[31mError: 'man' command not found. Install 'man-db' package?\e[0m"; return 1; fi
  man -k "$@" | less || echo "\e[33mNo man pages found for '$@'.\e[0m"
}

# --- Function: List processes sorted by memory usage ---
# Show top memory-consuming processes. 🧠
list_processes_by_memory() {
  if _command_exists ps; then
    ps -eo pid,user,%mem,%cpu,rss,start_time,cmd --sort=-%mem | head -n 20 || ps aux # Detailed `ps` or fallback to `ps aux`.
  else
    echo "\e[31mError: 'ps' command not found. Install 'procps' package?\e[0m"; return 1;
  fi
}

# --- Function: Update Termux packages (update and upgrade) ---
# Streamline Termux system updates. 🔄
update_all() {
  if ! _command_exists pkg; then echo "\e[31mError: 'pkg' command not found. Cannot update.\e[0m"; return 1; fi
  echo "\e[36mRunning Termux package update and upgrade...\e[0m"
  pkg update && pkg upgrade -y && echo "\e[32mTermux packages updated successfully. ✨\e[0m" || echo "\e[31mTermux package update failed. 💔\e[0m"
}

# --- Function: Synchronize dotfiles using Git ---
# Keep your dotfiles in sync with a Git repository. 🔄
sync_dotfiles() {
  local repo_dir="${1:-$ZDOTDIR}" # Default to ZDOTDIR if no argument.
  if ! _command_exists git; then echo "\e[31mError: 'git' command not found. Install 'git' package?\e[0m"; return 1; fi
  if [[ ! -d "$repo_dir" ]]; then echo "\e[31mError: Directory '$repo_dir' not found: '$repo_dir'.\e[0m"; return 1; fi
  (
    cd "$repo_dir" && \
    if [[ ! -d ".git" ]]; then echo "\e[31mError: '$repo_dir' is not a git repository. Ensure '.git' directory exists.\e[0m"; return 1; fi && \
    echo "\e[36mSyncing dotfiles in '$repo_dir' with git...\e[0m" && \
    git pull --ff-only && git submodule update --init --recursive && \
    echo "\e[32mDotfiles synchronized. Reloading shell... ✨\e[0m" && \
    exec zsh # Replace shell process for clean reload. 🔄
  ) || echo "\e[31mFailed to sync dotfiles. 💔\e[0m"
}

# --- Function: Generate SSH key pair (ed25519) ---
# Securely generate a new ed25519 SSH key pair. 🔑
manage_ssh_keys() {
  if ! _command_exists ssh-keygen; then echo "\e[31mError: 'ssh-keygen' command not found. Install 'openssh' package?\e[0m"; return 1; fi
  local comment="${1:-$(whoami)@$(hostname)-termux}" # Default comment: user@hostname-termux.
  local key_path="$HOME/.ssh/id_ed25519"
  echo "\e[36mGenerating new ed25519 SSH keypair...\e[0m"
  ssh-keygen -t ed25519 -C "$comment" -f "$key_path"
  echo "\e[32mKeypair generated. Public key content (add this to your server's authorized_keys):\e[0m"
  cat "${key_path}.pub"
}

# --- Placeholder Function: SSH to work server (customize!) ---
# Example function for SSHing to a work server - **EDIT ME!** ⚠️
ssh_work() {
  echo "\e[33mReminder: Edit the 'ssh_work' function in your Zsh configuration with your SSH connection details!\e[0m"
  # Example: ssh -p 2222 user@work.example.com # Customize port, user, and hostname.
}

# --- Function: Generate a random password ---
# Create strong, random passwords of specified length (default 32 chars). 🎲
gen_password() {
  local length="${1:-32}" # Default password length: 32 characters.
  if [[ ! "$length" =~ ^[0-9]+$ || "$length" -eq 0 ]]; then
    echo "\e[31mUsage: gen_password [length > 0] (length must be a positive number).\e[0m"
    return 1
  fi
  # Use /dev/urandom for high-quality randomness. 🎲
  LC_ALL=C tr -dc 'A-Za-z0-9_!@#$%^&*()-' < /dev/urandom | head -c "$length" ; echo
}

# --- Function: Open current directory in Termux file picker ---
# Launch Termux's file picker in the current directory. 📂
open_dir() {
  if ! _command_exists termux-open; then echo "\e[31mError: Termux file opener ('termux-open') unavailable. Install 'termux-api' package?\e[0m"; return 1; fi
  termux-open . || echo "\e[31mFailed to open directory in Termux file picker. 💔\e[0m"
}

# --- Function: Run a command in all immediate subdirectories ---
# Execute a command in each subdirectory (with confirmation for safety!). ⚠️
run_in_dirs() {
  local cmd_to_run=("$@") # Store command and arguments.
  if [[ ${#cmd_to_run[@]} -eq 0 ]]; then
    echo "\e[31mUsage: run_in_dirs <command> [args...] (command to run in subdirectories).\e[0m"
    return 1
  fi
  echo $'\e[33mWARNING: About to run "'"${cmd_to_run[*]}"$'" in all subdirectories here. Proceed? (y/N - default N):\e[0m'
  read -k 1 -r reply # Single-key confirmation.
  echo # Newline after input.
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "\e[31mAborted. Command not run in subdirectories.\e[0m"; return 1; }

  for dir in */; do
    if [[ -d "$dir" ]]; then
      ( # Subshell to isolate `cd` and prevent main shell directory change. 🛡️
        echo $'\e[36m--> Entering '"$dir"$'\e[0m'
        cd "$dir" && "${cmd_to_run[@]}" && \
        echo $'\e[32m<-- Exited '"$dir"$'\e[0m'
      ) || echo $'\e[31m### Failed in '"$dir"$'\e[0m'
    fi
  done
  echo "\e[32mrun_in_dirs completed.\e[0m"
}

# --- Function: Find files larger than a specified size ---
# Locate large files (default: > 100MB). 🔍
find_large_files() {
  local size="${1:-+100M}" # Default size: > 100MB.
  if ! _command_exists find; then echo "\e[31mError: 'find' command not found. Install 'findutils' package?\e[0m"; return 1; fi
  echo "\e[36mSearching for files larger than ${size}...\e[0m"
  find . -type f -size "$size" -exec ls -lh {} \; 2>/dev/null || echo "\e[33mNo files found larger than ${size} in current directory or search failed.\e[0m"
}

# --- Function: Shorten a URL using TinyURL service ---
# Condense long URLs into tiny links using TinyURL. 🔗
shorten_url() {
  if ! _command_exists curl; then echo "\e[31mError: 'curl' command not found. Install 'curl' package?\e[0m"; return 1; fi
  if [[ -z "$1" ]]; then echo "\e[31mUsage: shorten_url <URL> (URL to shorten).\e[0m"; return 1; fi
  curl -s "https://tinyurl.com/api-create.php?url=$1"; echo # TinyURL API magic. 🪄
}

# --- Function: Clean temporary files in home directory ---
# Remove common temporary files (patterns configurable). 🧹
clean_temp_files() {
  local patterns=("*.tmp" "*.bak" "*~" "._*") # Common temp file patterns. Add more if needed.
  local args=()
  local first=1
  for p in "${patterns[@]}"; do
      [[ $first -ne 1 ]] && args+=("-o") # OR condition for 'find'.
      args+=("-name" "$p")
      first=0
  done
  if ! _command_exists find; then echo "\e[31mError: 'find' command not found. Install 'findutils' package?\e[0m"; return 1; fi
  echo "\e[33mCleaning temporary files (${patterns[*]}) in home directory (max depth 3)...\e[0m"
  find "$HOME" -maxdepth 3 -type f \( "${args[@]}" \) -print -delete
  echo "\e[32mTemporary file cleanup complete. ✨\e[0m"
}

# --- Function: Display Termux system information ---
# Show detailed Termux system info using `termux-info`. ℹ️
system_info() {
  if ! _command_exists termux-info; then echo "\e[31mError: Termux diagnostics ('termux-info') not installed. Install 'termux-api' package?\e[0m"; return 1; fi
  termux-info
}

# --- Function: Reload Zsh configuration (using exec for cleanliness) ---
# Refresh your shell configuration without lingering processes. 🔄
update_shell() {
  echo "\e[36mReloading Zsh configuration... ✨\e[0m"
  exec zsh
}

# --- Function: Display welcome message ---
# Greet the wizard upon shell startup. 🧙‍♂️
welcome_message() {
  local username=$(whoami)
  local hostname=$(hostname)
  local uptime=$(uptime -p | sed 's/up //') # Human-readable uptime.
  local ip_addr=$(myip) # Reuse myip function.
  local disk_usage=$(df -h . | awk 'NR==2{print $5}') # Disk usage of current dir.

  echo -e "\e[1;35m🧙‍♂️ Welcome, $username@$hostname, to the Termux Zsh Nexus! ✨\e[0m [\e[32m$(date '+%Y-%m-%d %H:%M:%S')\e[0m]"
  echo -e "\e[2mUptime: $uptime | Public IP: $ip_addr | Disk Usage (current dir): $disk_usage\e[0m"
  echo -e "\e[36mTip: Use 'helpme' or 'functions' to explore available commands and functions. Type 'reload' to refresh config.\e[0m"
}

# --- Function: List currently defined aliases ---
# Show your custom aliases. 🏷️
list_aliases() { alias | sort; } # Sorted alias list.

# --- Function: List currently defined custom functions ---
# Reveal your custom function spells. 📜
list_functions() { typeset -f | grep -v '^_' | sed -n '/^[a-zA-Z]/s/^\([a-zA-Z0-9_]*\).*/\1/p' | sort; } # Filter out internal functions, sorted.

# --- Function: Search command history ---
# Dig through your command history with ease. 🔍
history_search() { history 0 | grep --color=auto "$@"; } # `history 0` shows full history.

# --- Function: Backup important dotfiles ---
# Create a backup archive of your precious configuration files. 📦
backup_dotfiles() {
    local backup_file="$HOME/dotfiles_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    local files_to_backup=(
        "$ZSHRC"                             # Main Zsh config.
        "$ZDOTDIR/.p10k.zsh"                 # Powerlevel10k config.
        "$ZDOTDIR/.zshenv"                   # Environment vars (if used).
        "$ZDOTDIR"                           # Entire config dir (optional, all-inclusive).
        # Add more critical dotfiles here, e.g.:
        # "$HOME/.gitconfig"
        # "$HOME/.config/nvim/init.vim" # Neovim config (or init.lua).
        # "$HOME/.ssh/config"
    )
    local existing_files=() # Filter out non-existent files.
    for f in "${files_to_backup[@]}"; do
        [[ -e "$f" ]] && existing_files+=("$f")
    done

    if [[ ${#existing_files[@]} -eq 0 ]]; then
        echo "\e[31mWarning: No configuration files found to backup. 💔\e[0m"
        return 1
    fi

    echo "\e[36mCreating dotfiles backup archive (${#existing_files[@]} files) to $backup_file...\e[0m"
    tar -czf "$backup_file" "${existing_files[@]}" && \
      echo "\e[32mDotfiles backup created successfully at $backup_file. 📦✨\e[0m" || \
      echo "\e[31mDotfiles backup failed. 💔\e[0m"
}

# --- Function: Execute a Python script ---
# Run Python scripts, auto-detecting `python3` or `python`. 🐍
run_python() {
    local python_exe script_path="$1"
    shift # Remove script path from arguments.

    if _command_exists python3; then
        python_exe="python3"
    elif _command_exists python; then
        python_exe="python"
    else
        echo "\e[31mError: Python interpreter not found ('python3' or 'python'). Install 'pkg install python'.\e[0m"
        return 1
    fi

    if [[ -f "$script_path" ]]; then
        "$python_exe" "$script_path" "$@" || echo "\e[31mError executing Python script: $script_path. 💔\e[0m"
    elif [[ -z "$script_path" ]]; then
        echo "\e[31mUsage: run_python <script.py> [arguments...] (path to Python script).\e[0m"
        return 1
    else
        echo "\e[31mError: Python script '$script_path' not found or not a file. 💔\e[0m"
        return 1
    fi
}

# --- Function: Get system uptime in human-readable format ---
# Display system uptime in a user-friendly way. ⏱️
get_uptime() { uptime -p | sed 's/up //'; } # Simple wrapper for uptime -p.

# --- Function: Display help information (aliases and functions) ---
# Quick guide to your custom commands and functions. 📖
helpme() {
  echo -e "\e[1;36mWizard's Help Scroll 📜\e[0m"
  echo -e "\e[36mAvailable Aliases:\e[0m"
  list_aliases | head -n 15 # Show first 15 aliases.
  echo -e "\n\e[36m... (use 'aliases' command to see all)\e[0m"
  echo -e "\n\e[36mAvailable Functions:\e[0m"
  list_functions | head -n 15 # Show first 15 functions.
  echo -e "\n\e[36m... (use 'functions' command to see all)\e[0m"
  echo -e "\n\e[36mFor function-specific help, examine the '.zshrc' file directly.\e[0m"
}
alias help='helpme' # Alias 'help' to 'helpme'.
alias functions='list_functions' # Alias 'functions' to list functions.
alias aliases='list_aliases'     # Alias 'aliases' to list aliases.

# --- Network Utility Functions (if network tools are installed) --- 🌐
# ping function
ping_host() {
  if ! _command_exists ping; then echo "\e[31mError: 'ping' command not found. Install 'iputils' package?\e[0m"; return 1; fi
  if [[ -z "$1" ]]; then echo "\e[31mUsage: ping_host <hostname or IP> (host to ping).\e[0m"; return 1; fi
  ping -c 5 "$1" # Ping 5 times.
}

# traceroute function
traceroute_host() {
  if ! _command_exists traceroute; then echo "\e[31mError: 'traceroute' command not found. Install 'traceroute' package?\e[0m"; return 1; fi
  if [[ -z "$1" ]]; then echo "\e[31mUsage: traceroute_host <hostname or IP> (host to traceroute).\e[0m"; return 1; fi
  traceroute "$1"
}

# whois function
whois_domain() {
  if ! _command_exists whois; then echo "\e[31mError: 'whois' command not found. Install 'whois' package?\e[0m"; return 1; fi
  if [[ -z "$1" ]]; then echo "\e[31mUsage: whois_domain <domain name> (domain to whois).\e[0m"; return 1; fi
  whois "$1"
}

# dig function (DNS lookup)
dig_domain() {
  if ! _command_exists dig; then echo "\e[31mError: 'dig' command not found. Install 'bind-tools' package?\e[0m"; return 1; fi
  if [[ -z "$1" ]]; then echo "\e[31mUsage: dig_domain <domain name> (domain to dig).\e[0m"; return 1; fi
  dig "$1"
}

# --- AI Integration (aichat) ---
# Unleash AI power with `aichat` for command suggestions and text wizardry. 🤖✨
# Requires `aichat` installation (`cargo install aichat` - Rust dependency: `pkg install rust`).
# Ensure AIChat API key is configured (usually in `$AICHA_CONFIG/aichat.toml`).
export AICHA_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/aichat" # XDG-compliant aichat config dir.

# --- Helper Function: Check if aichat is installed ---
# Internal check for `aichat` command availability. 🛡️
_check_aichat_installed() {
    if ! _command_exists aichat; then
        echo $'\e[33mNotice: AIChat tool not detected. AI features are limited. Install Rust & aichat for full power. 🤖\e[0m'
        echo $'\e[36mTo enable AI, install Rust (\'pkg install rust\'), then \'cargo install aichat\'.\e[0m'
        return 1
    fi
    return 0
}

# --- ZLE Widget: AI Command Suggestion (bound to Alt+E) ---
# Get AI command suggestions using aichat (bound to Alt+E). 🔮
_aichat_zsh_suggest() {
    local current_buffer="$BUFFER" # Current command line buffer.
    if ! _check_aichat_installed; then zle redisplay; return 1; fi # Check aichat availability.

    if [[ -n "$current_buffer" ]]; then
        echo -n $'\n\e[34mAI Suggestion: \e[0m' # AI suggestion prefix.
        aichat --no-stream -r cmd "$current_buffer -- Refine or suggest an alternative zsh command for the line above." 2>/dev/null || echo $'\e[31mAI suggestion failed. 💔\e[0m'
    else
        echo $'\n\e[34mAIChat is ready... (Type a command to get AI-powered suggestions) 🤖\e[0m'
    fi
    zle redisplay # Redraw prompt to show suggestion/message.
}
zle -N _aichat_zsh_suggest # Register widget: `_aichat_zsh_suggest`.

# --- Function: Summarize text output using AI (aichat) ---
# Summarize text (stdin or arguments) using aichat. 📝
summarize_output() {
    local input_text # Input text for summarization.
    if [[ ! -t 0 ]]; then # Read from stdin if not a terminal.
        input_text="$(cat)"
    elif [[ $# -gt 0 ]]; then # Or use arguments.
        input_text="$*"
    else
        echo $'\e[31mUsage: <command> | summarize_output  or  summarize_output "text to summarize" (text for AI summarization).\e[0m'
        return 1
    fi

    if ! _check_aichat_installed; then # Fallback if aichat is missing.
        echo $'\e[33m(AIChat not found, showing first few lines as a basic summary)\e[0m'
        head -n 7 <<< "$input_text" # Basic summary using `head`.
        return 1
    fi

    echo -e "\e[34mSummarizing with AI... 🤖\e[0m"
    aichat -r summary --no-stream <<< "$input_text" 2>/dev/null || echo $'\e[31mAI summarization failed. 💔\e[0m'
}

# --- Keybindings (bindkey) ---
# Customize keybindings for efficient shell interaction. ⌨️
# Explore `man zshzle` for the full keybinding spellbook. 📖

# --- AI Suggestion Keybinding ---
bindkey '^[e' _aichat_zsh_suggest           # Alt+E: Trigger AI command suggestion. 🔮

# --- Essential Navigation and Editing Keybindings --- 🧭
bindkey '^[[H' beginning-of-line         # Home key: Go to line start. 🏠
bindkey '^[[F' end-of-line               # End key: Go to line end. 🏁
bindkey '^[[3~' delete-char              # Delete key: Delete char at cursor. ❌
bindkey '^?'   backward-delete-char       # Backspace: Delete char before cursor. 🔙
bindkey '^W'   backward-kill-word         # Ctrl+W: Delete word backward. ✂️
bindkey '^U'   backward-kill-line         # Ctrl+U: Delete line backward. ✂️
bindkey '^A'   beginning-of-line         # Ctrl+A: Go to line start. 🏠
bindkey '^E'   end-of-line               # Ctrl+E: Go to line end. 🏁
bindkey '^L'   clear-screen              # Ctrl+L: Clear screen. 🧹
bindkey '^R'   history-incremental-search-backward # Ctrl+R: Incremental history search backward. 🔍

# --- Autosuggestion Bindings (if zsh-autosuggestions plugin is loaded) --- 🔮
# Keybindings for accepting suggestions from zsh-autosuggestions.
bindkey '^[[C' forward-char    # Right Arrow: Accept suggestion (default). →
bindkey '^[[F' end-of-line     # End key: Accept suggestion (default). 🏁
# Alternative accept keys (uncomment to use):
# bindkey '^ ' autosuggest-accept # Ctrl+Space: Accept suggestion.
# bindkey '^E' autosuggest-accept # Ctrl+E: Accept suggestion (overrides default Ctrl+E).

# --- History Substring Search Bindings (optional - plugin required) --- 🔍
# Uncomment if you use history-substring-search plugin.
# bindkey '^[[A' history-substring-search-up   # Up arrow: History substring search up. ↑
# bindkey '^[[B' history-substring-search-down # Down arrow: History substring search down. ↓

# --- History Encryption/Decryption (Optional - using GPG) --- 🛡️
# Encrypt command history using GPG for enhanced security. 🔒
# Requires GPG: `pkg install gnupg`.
# Securely manage GPG passphrase (e.g., in `$ZDOTDIR/.zshenv`, chmod 600, or gpg-agent).

# --- Function: Encrypt history file ---
# Encrypt history file using GPG. 🛡️
encrypt_history() {
    if _command_exists gpg && [[ -f "$HISTFILE" && -n "$GPG_PASSPHRASE" ]]; then # Auto encryption (passphrase set).
        gpg -c --batch --yes --passphrase-fd 0 -o "$HISTFILE.gpg" "$HISTFILE" <<< "$GPG_PASSPHRASE" && rm "$HISTFILE"
    elif _command_exists gpg && [[ -f "$HISTFILE" ]]; then # Manual encryption (prompts for passphrase).
        echo $'\n\e[36mEncrypting history (passphrase required)...\e[0m'
        gpg -c -o "$HISTFILE.gpg" "$HISTFILE" && rm "$HISTFILE"
    fi
}

# --- Function: Decrypt history file ---
# Decrypt history file using GPG. 🔓
decrypt_history() {
    if _command_exists gpg && [[ -f "$HISTFILE.gpg" && -n "$GPG_PASSPHRASE" ]]; then # Auto decryption (passphrase set).
        gpg -d --batch --yes --passphrase-fd 0 "$HISTFILE.gpg" > "$HISTFILE" 2>/dev/null <<< "$GPG_PASSPHRASE"
    elif _command_exists gpg && [[ -f "$HISTFILE.gpg" ]]; then # Manual decryption (may prompt).
        echo "\e[36mAttempting history decryption (passphrase may be required)...\e[0m"
        gpg -d "$HISTFILE.gpg" > "$HISTFILE" 2>/dev/null
    fi
}

# --- History Encryption Setup ---
# Configure history encryption hooks based on GPG and passphrase. ⚙️
if _command_exists gpg; then
    if [[ -n "$GPG_PASSPHRASE" ]]; then # Automatic encryption mode (passphrase set).
        decrypt_history # Decrypt on startup. 🔓
        trap 'encrypt_history' EXIT # Encrypt on exit. 🛡️
    else # Manual encryption mode (no passphrase set).
        echo "\e[33mNotice: Set GPG_PASSPHRASE env variable (in .zshenv or gpg-agent) for automatic history encryption/decryption.\e[0m"
        decrypt_history # Attempt decrypt on startup (may prompt). 🔓
        trap 'encrypt_history' EXIT # Encrypt on exit (will prompt if needed). 🛡️
    fi
else # GPG not found - history encryption disabled. ⛔
    echo "\e[31mWarning: 'gpg' command not found. History encryption is disabled. Install 'gnupg' for encryption.\e[0m"
fi

# --- Load Final Custom User Configurations ---
# Load extra custom configurations from files in `$ZDOTDIR` for modularity. 🧩
# These can override previous settings or add specific tweaks.
[[ -f "$ZDOTDIR/aliases.zsh" ]] && source "$ZDOTDIR/aliases.zsh"    # Custom aliases. 🏷️
[[ -f "$ZDOTDIR/functions.zsh" ]] && source "$ZDOTDIR/functions.zsh"  # Custom functions. 📜
[[ -f "$ZDOTDIR/local.zsh" ]] && source "$ZDOTDIR/local.zsh"      # Machine-specific or local tweaks. 🛠️

# --- Final Touches ---
# Apply any final settings, especially for Powerlevel10k theme. ✨
# Customize P10k settings here or in `~/.config/zsh/.p10k.zsh` (via `p10k configure`).
# Examples (uncomment and adjust):
# POWERLEVEL10K_MODE='nerdfont-complete' # Nerd Font icons.
# POWERLEVEL10K_LEFT_PROMPT_ELEMENTS=(context dir vcs time) # Left prompt elements.
# POWERLEVEL10K_RIGHT_PROMPT_ELEMENTS=(status command_execution_time ram) # Right prompt elements.
# POWERLEVEL10K_PROMPT_ON_NEWLINE=true # Prompt on a new line.
# POWERLEVEL10K_MULTILINE_NEWLINE=true # Multiline prompt newline.

# --- Display Welcome Message ---
# Show the welcome message upon shell startup. 🎉
welcome_message

# --- End of ~/.zshrc - Wizard Edition ✨ ---
