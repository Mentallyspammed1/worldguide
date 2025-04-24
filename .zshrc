# Enable Powerlevel10k instant prompt (must be near the top)
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Last Updated: 2025-04-03
# Enhanced ZSH Configuration 2.4.1 (Termux Compatible with aichat)

# Core Environment Variables
export TERM="xterm-256color"
export LANG="en_US.UTF-8"
export SHELL="$(which zsh)"
export ZSH="$HOME/.oh-my-zsh"
export ZSHRC="$HOME/.zshrc"
export PATH="$HOME/bin:$HOME/.local/bin:/data/data/com.termux/files/usr/bin:$PATH"
export PATH="$HOME/.cargo/bin:$PATH"  # Rust support
export EDITOR="nano"
export VISUAL="$EDITOR"
export PYTHONPATH="/data/data/com.termux/files/usr/bin/python"
export AICHA_CONFIG="$HOME/.config/aichat"  # aichat configuration directory

# Shell Options
setopt auto_cd correct numeric_glob_sort extended_glob interactive_comments glob_dots
setopt hist_verify share_history inc_append_history hist_ignore_dups hist_ignore_space

# History Configuration
export HISTFILE="$HOME/.zsh_history"
export HISTSIZE=50000
export SAVEHIST=50000
export HIST_IGNORE_SPACE="true"
export HIST_IGNORE_DUPS="true"
export HIST_NO_STORE="ls:cd:pwd:exit:history:bg:fg:jobs"

# Theme Configuration (Powerlevel10k)
THEME_DIR="$ZSH/custom/themes/powerlevel10k"
THEME_FILE="$THEME_DIR/powerlevel10k.zsh-theme"
if [[ -f "$THEME_FILE" ]]; then
  ZSH_THEME="powerlevel10k/powerlevel10k"
  POWERLEVEL10K_MODE='nerdfont-complete'
  POWERLEVEL10K_LEFT_PROMPT_ELEMENTS=(context dir vcs time)
  POWERLEVEL10K_RIGHT_PROMPT_ELEMENTS=(status command_execution_time ram)
  POWERLEVEL10K_PROMPT_ON_NEWLINE=true
  POWERLEVEL10K_MULTILINE_NEWLINE=true
  source "$THEME_FILE"
else
  ZSH_THEME="agnoster"
  echo "Powerlevel10k not found. Install with: git clone https://github.com/romkatv/powerlevel10k.git $THEME_DIR"
  [[ -x "$(command -v termux-toast)" ]] && termux-toast "Install Powerlevel10k for enhanced prompt"
fi

# Plugins (including aichat)
plugins=(
  git
  zsh-autosuggestions
  zsh-completions
  fzf-tab
  zsh-autopair
  alias-tips
  fast-syntax-highlighting
  aichat  # Added aichat plugin
)
fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src
source "$ZSH/oh-my-zsh.sh" 2>/dev/null || {
  echo "Oh My Zsh not found. Install with: sh -c \"\$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)\""
  [[ -x "$(command -v termux-toast)" ]] && termux-toast "Install Oh My Zsh"
}

# aichat Integration
if [[ ! -x "$(command -v aichat)" ]]; then
  echo "aichat not installed. Install with: cargo install aichat (requires Rust)"
fi
_aichat_zsh() {
  local current_buffer="$BUFFER"
  if [[ -n "$current_buffer" && -x "$(command -v aichat)" ]]; then
    echo -n "AI Suggestion: "
    aichat suggest "$current_buffer" --max-length 20000 2>/dev/null || echo "Error with aichat"
  else
    echo "AIChat: Type a command first or install aichat (Alt+E)"
  fi
  zle accept-line
}
zle -N _aichat_zsh
bindkey '^[e' _aichat_zsh  # Alt+E for AI suggestions

# Enhanced Aliases
[[ -x "$(command -v bat)" ]] && alias cat="bat --theme=Dracula" || alias cat="cat"
[[ -x "$(command -v eza)" ]] && {
  alias ls="eza --group-directories-first --icons"
  alias ll="eza -lh --git"
  alias la="eza -lah --icons"
  alias lt="eza -T --level=2"
} || alias ls="ls --color=auto"
[[ -x "$(command -v fd)" ]] && alias find="fd" || alias find="find"
alias grep="grep --color=auto"
alias df="df -h"
alias du="du -h --max-depth=1"
alias mkdir="mkdir -p"
alias gs="git status"
alias ga="git add"
alias gc="git commit -m"
alias gp="git push"

# Custom Functions
gen_password() { tr -dc 'A-Za-z0-9!@#$%^&*' </dev/urandom | head -c 16; echo; }
find_large_files() { find . -type f -size +100M -exec ls -lh {} \; 2>/dev/null; }
update_all() { pkg update && pkg upgrade -y; }
system_info() { termux-info 2>/dev/null || uname -a; }
history_search() { fc -l 1 | grep "$@"; }
ai_summarize() {  # Uses aichat to summarize command output
  local output="$1"
  if [[ -x "$(command -v aichat)" ]]; then
    echo "$output" | aichat summarize --max-length 5000 2>/dev/null || echo "Error summarizing"
  else
    echo "$output" | head -n 1
  fi
}

# Completion Optimizations
autoload -Uz compinit && compinit -i
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}' 'r:|=*' 'l:|=* r:|=*'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$HOME/.zsh_cache"

# Keybindings
bindkey '^[[H' beginning-of-line
bindkey '^[[F' end-of-line
bindkey '^R' history-incremental-search-backward
bindkey '^L' clear-screen

# Load Custom Configurations
ZSH_CUSTOM_DIR="$HOME/.config/zsh"
[[ -d "$ZSH_CUSTOM_DIR" ]] && for f in "$ZSH_CUSTOM_DIR"/*.zsh; do source "$f"; done

# Final Setup
[[ -f ~/.p10k.zsh ]] && source ~/.p10k.zsh
echo "ZSH loaded successfully - $(date)"
