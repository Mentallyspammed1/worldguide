[[ -f ${ZDOTDIR}/conf.d/*.zsh ]] && source ${ZDOTDIR}/conf.d/*.zsh(.)                                                                         # -*- Status Report -*-                                                print -P "%F{082}Zsh ${ZSH_VERSION} loaded in ${(%.3)$(( SECONDS * 1000 ))} ms%f"                                                             EOF && source ~/.config/zsh/.zshrc
# ~/.zshrc - Configuration for Termux - Enhanced and Optimized (Wizard Edition ✨)
# This configuration elevates your Termux Zsh experience to a new level of wizardry! 🧙‍♂️

# --- Startup Optimization ---
# Source instant prompt *after* setting ZDOTDIR for correct cache path.
export ZDOTDIR="${XDG_CONFIG_HOME:-$HOME/.config}/zsh"
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# --- Environment Variables ---
export TERM="xterm-256color"
export LANG="en_US.UTF-8"
# Dynamically set EDITOR based on availability (nvim, vim, then nano).
if command -v nvim >/dev/null 2>&1; then
  export EDITOR="nvim"
elif command -v vim >/dev/null 2>&1; then
  export EDITOR="vim"
else
  export EDITOR="nano"
fi
export VISUAL="$EDITOR"
alias gimg='bash "$HOME/gimg"'
# --- Zsh Configuration Directories (XDG Compliant) ---
export ZSHRC="$ZDOTDIR/.zshrc"
export ZSH_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/zsh"
export ZSH_COMPDUMP="${ZSH_CACHE_DIR}/zcompdump"
# Use a more standard history file name
export HISTFILE="${XDG_STATE_HOME:-$HOME/.local/state}/zsh/history"
mkdir -p "$ZSH_CACHE_DIR" "$(dirname "$HISTFILE")"

# --- Zsh Options (setopt) ---
setopt auto_cd auto_pushd pushd_ignore_dups correct numeric_glob_sort \
      no_flow_control extended_glob glob_dots interactive_comments \
      promptsubst share_history inc_append_history hist_expire_dups_first \
      hist_ignore_dups hist_ignore_space hist_find_no_dups hist_no_functions \
      rc_quotes

# --- History Settings ---
export HISTSIZE=50000
export SAVEHIST=50000
export HIST_IGNORE="ls:cd:pwd:exit:history:bg:fg:jobs:clear:htop:ncdu:lazygit:lazydocker:man:df:free:top"


# --- PATH Configuration ---
# Use a function for more robust and flexible PATH management
add_to_path() {
  if [[ -d "$1" ]]; then
    export PATH="$1:$PATH"
    typeset -U path # Ensure PATH is unique
  fi
}

add_to_path "$HOME/bin"
add_to_path "$HOME/.local/bin"
add_to_path "/data/data/com.termux/files/usr/bin"
add_to_path "$HOME/.cargo/bin"
add_to_path "$HOME/platform-tools"



# --- Function Path (fpath) ---
fpath=($ZDOTDIR/functions $fpath) # Enable custom functions


# --- Theme (Powerlevel10k with improved fallback) ---
THEME_DIR="${ZSH_CUSTOM:-${XDG_DATA_HOME:-$HOME/.local/share}/oh-my-zsh/custom}/themes/powerlevel10k"
if [[ -f "$THEME_DIR/powerlevel10k.zsh-theme" ]]; then
  ZSH_THEME="powerlevel10k/powerlevel10k"
else
  # Check for more common fallback themes before resorting to agnoster
  if [[ -f "${ZSH_CUSTOM:-${ZDOTDIR:-$HOME/.oh-my-zsh}/custom}/themes/robbyrussell.zsh-theme" ]]; then
    ZSH_THEME="robbyrussell"
  elif [[ -f "${ZSH_CUSTOM:-${ZDOTDIR:-$HOME/.oh-my-zsh}/custom}/themes/bira.zsh-theme" ]]; then
    ZSH_THEME="bira"
  else
    ZSH_THEME="agnoster"
  fi
  print -P "%F{yellow}Notice:%f Powerlevel10k not found. Using fallback theme: %F{cyan}$ZSH_THEME%f"
  print -P "%F{blue}Install Powerlevel10k for a better experience:%f %F{green}git clone --depth=1 https://github.com/romkatv/powerlevel10k.git $THEME_DIR%f"
fi


# --- Powerlevel10k Configuration ---
if [[ -f "$ZDOTDIR/.p10k.zsh" ]]; then source "$ZDOTDIR/.p10k.zsh"; fi

# --- Plugin Manager (zinit) with improved error handling ---
ZINIT_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/zinit/zinit.git"
if [[ ! -d "$ZINIT_HOME" ]]; then
  print -P "%F{blue}Installing zinit...%f"
  mkdir -p "$(dirname $ZINIT_HOME)"
  if ! git clone --depth=1 https://github.com/zdharma-continuum/zinit.git "$ZINIT_HOME"; then
    print -P "%F{red}Error: zinit installation failed! Check internet connection.%f"
    return 1 # Stop .zshrc execution on zinit failure
  fi
fi

source "$ZINIT_HOME/zinit.zsh"

# --- Load Plugins with Zinit (with Turbo mode and improved snippets) ---
zinit ice wait lucid turbo-mode for \
    ohmyzsh/ohmyzsh \
    zdharma-continuum/fast-syntax-highlighting \
    zsh-users/zsh-autosuggestions \
    zsh-users/zsh-completions \
    agkozak/zsh-z

# Load Oh My Zsh plugins with ice and light-mode
zinit ice from:ohmyzsh light-mode for \
    plugins/command-not-found \
    plugins/sudo \
    plugins/extract \
    plugins/web-search

# Initialize completion system after plugins are loaded
zicompinit
zicdreplay

# ... (Rest of the .zshrc file - aliases, functions, etc.)

# --- Improved Aliases ---
# ... (Existing aliases)
alias l='exa -l --git' # Use exa if available
alias la='exa -la --git'
alias ll='exa -lAh --git'
alias lt='exa --tree --level=2'


# --- Improved Functions ---
# ... (Existing functions)


# --- AI Integration (aichat) with API Key check and improved messages ---
# ... (Existing aichat code)
_check_aichat_installed() { # Enhanced AIChat check with API key verification
    if ! command -v aichat &>/dev/null ; then
        print -P "%F{yellow}AIChat not found. Install with: %f%F{green}cargo install aichat%f"
        return 1
    fi

    if [[ -z "$(grep '^api_key =' "${AICHA_CONFIG}/aichat.toml")" ]]; then # Check for API key
        print -P "%F{yellow}AIChat API key not configured. AI features will be limited.%f Edit %F{cyan}\$AICHA_CONFIG/aichat.toml%f"
        return 1 # Or optionally continue without AI features but notify the user
    fi
}

# ... (Rest of AIChat functions)

# ---  Keybindings ---
# ... (Existing Keybindings)

# --- History Encryption (Optional - using GPG) with improved passphrase handling---
# ... (existing encryption/decryption functions)

# --- History Encryption Setup (improved) ---
if command -v gpg &>/dev/null; then
    # Try to get passphrase from gpg-agent
    GPG_PASSPHRASE=$(gpg-connect-agent "GET_PASSPHRASE --data histfile" /bye 2>/dev/null)


    if [[ -n "$GPG_PASSPHRASE" ]]; then
      decrypt_history
      trap 'encrypt_history' EXIT
    else
      if [[ -f "$HISTFILE.gpg" ]]; then # Only ask if an encrypted file exists
          read -s -k "?"$'\n\e[36mEnter passphrase to decrypt history (or press Enter to skip):\e[0m ' GPG_PASSPHRASE
          echo  # Newline for cleaner output
          decrypt_history
      fi
       trap 'encrypt_history' EXIT # Encrypt on exit, even if decryption failed/skipped

    fi
else
    print -P "%F{yellow}gpg not found. History encryption disabled.%f Install with: %F{green}pkg install gnupg%f"
fi


# --- Welcome Message (Show only in interactive shells) ---
if [[ $- == *i* ]]; then
  welcome_message
fi

# --- End of ~/.zshrc - Wizard Edition ✨ ---
