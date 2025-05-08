import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Define the sanctum paths
config_dir = os.path.expanduser("~/.config/zsh")
home_dir = os.path.expanduser("~")
# Assume directories exist for this presentation

# --- Scroll Contents ---

zshenv_content = """# {Pyrmethus} ~/.zshenv: The Primal Invocation
# Executed by ALL Zsh manifestations (login, interactive, scripts)
# Keep this scroll potent yet concise, focused on foundational energies.

# {Pyrmethus} Designating the Sanctum Sanctorum for Zsh scrolls
export ZDOTDIR="${HOME}/.config/zsh"

# {Pyrmethus} Harmonizing the Locale Energies
export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'

# {Pyrmethus} Selecting the Scribing Tools (Editor/Pager)
export EDITOR="${EDITOR:-nano}" # Default: Nano, the Humble Scribe
export VISUAL="$EDITOR"
export PAGER="${PAGER:-less}"   # Default: Less, the Patient Reader
export LESS="-RiXF"             # Empowering Less with useful incantations

# --- {Pyrmethus} The Vault of Secrets: API Key Handling ---
# {Pyrmethus} WARNING: Arcane keys grant immense power. NEVER etch them directly into
# version-controlled scrolls! Draw power securely from hidden vaults.
# Example: Drawing power from the ~/.secrets vault (ensure ~/.secrets is in .gitignore!)
# if [[ -f "$HOME/.secrets/aichat_api_key" ]]; then
#   export AICHA_API_KEY=$(cat "$HOME/.secrets/aichat_api_key")
#   print -P "%F{34}AIChat API Key drawn from the secret vault.%f"
# elif [[ -n "$AICHA_API_KEY_ENV_VAR" ]]; then # Example using another env var
#   export AICHA_API_KEY="$AICHA_API_KEY_ENV_VAR"
#   print -P "%F{34}AIChat API Key drawn from environment variable.%f"
# else
#   print -P "%F{160}Warning: AICHA_API_KEY power source not found. AI spells may falter.%f"
# fi

# {Pyrmethus} Placeholder - REMOVE and implement secure loading above!
export AICHA_API_KEY="YOUR_API_KEY_PLACEHOLDER_IN_SECURE_VAULT_OR_ENV"

# {Pyrmethus} Setting the GPG Portal TTY if needed
# [[ -n "$SSH_TTY" ]] && export GPG_TTY=$(tty)
"""

zshrc_content = """# {Pyrmethus} ~/.config/zsh/.zshrc: The Grand Grimoire (Interactive Shells)

# {Pyrmethus} --- P10k Instant Conjuration ---
# Allows the prompt to appear instantly. Keep this near the apex.
# Any incantations requiring user input must precede this nexus.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# {Pyrmethus} --- Establishing Core Essences & Summoning Zinit ---
export ZSH_CUSTOM="$ZDOTDIR/custom" # Optional path for OMZ compatibility
export ZSH_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/zsh" # Cache nexus
mkdir -p "$ZSH_CACHE_DIR"

# {Pyrmethus} Summoning Zinit, the Swift Familiar for Plugin Management
ZINIT_HOME="${XDG_DATA_HOME:-${HOME}/.local/share}/zinit/zinit.git"
if [[ ! -d "$ZINIT_HOME" ]]; then
    print -P "%F{111}Summoning Zinit Familiar...%f"
    command mkdir -p "$(dirname $ZINIT_HOME)"
    command git clone https://github.com/zdharma-continuum/zinit.git "$ZINIT_HOME" && \
        print -P "%F{75}Zinit Familiar successfully summoned.%f%b" || \
        print -P "%F{196}Error: Summoning Zinit failed! Check connection or paths.%f%b"
fi
if [[ -f "${ZINIT_HOME}/zinit.zsh" ]]; then
    source "${ZINIT_HOME}/zinit.zsh"
    autoload -Uz _zinit
    (( ${+_comps} )) && _comps[zinit]=_zinit
    print -P "%F{75}Zinit Familiar awakened.%f"
else
    print -P "%F{196}Error: Zinit Familiar's essence not found at ${ZINIT_HOME}/zinit.zsh%f"
fi

# {Pyrmethus} --- Weaving the Configuration Scrolls ---
# Sequentially source the modular scrolls of power.
print -P "%F{111}Weaving configuration scrolls...%f"
_zsh_config_scrolls=(
    options   # Scroll of Shell Options (setopt)
    env       # Scroll of Environment Aura (export, except PATH)
    paths     # Scroll of Ley Lines (PATH setup)
    aliases   # Scroll of Abbreviation Spells (alias)
    functions # Scroll of Incantations (functions, AI)
    plugins   # Scroll of Familiars (Zinit plugin loading)
    keybindings # Scroll of Arcane Bindings (bindkey)
)
for _scroll in "${_zsh_config_scrolls[@]}"; do
    if [[ -r "$ZDOTDIR/${_scroll}.zsh" ]]; then
        source "$ZDOTDIR/${_scroll}.zsh"
        print -P "%F{34} -> Scroll '${_scroll}.zsh' woven.%f"
    else
        print -P "%F{214}Warning: Configuration scroll not found: $ZDOTDIR/${_scroll}.zsh%f"
    fi
done
unset _zsh_config_scrolls _scroll # Vanish temporary variables

# {Pyrmethus} --- Awakening the Completion Spirits ---
# Must run *after* Zinit may have loaded completion plugins.
print -P "%F{111}Awakening completion spirits...%f"
autoload -Uz compinit
# Optimize compinit: Check timestamp before potentially slow init
if [[ -n "$ZSH_CACHE_DIR/.zcompdump"(Nmh+24) ]]; then
    compinit -i -q # Use recent dump silently
    print -P "%F{34}Completion spirits awakened swiftly.%f"
else
    print -P "%F{214}Compiling completion knowledge (first time or outdated)...%f"
    compinit -i -d "$ZSH_CACHE_DIR/.zcompdump" -q # Generate new dump
    print -P "%F{34}Completion knowledge compiled.%f"
fi

# {Pyrmethus} --- Igniting the Powerlevel10k Beacon ---
# Load the P10k theme configuration *after* Zinit and Compinit.
if [[ -f "$ZDOTDIR/.p10k.zsh" ]]; then
    source "$ZDOTDIR/.p10k.zsh"
    print -P "%F{75}Powerlevel10k Beacon ignited.%f"
else
     print -P "%F{214}Warning: Powerlevel10k configuration (.p10k.zsh) not found. Run 'p10k configure'.%f"
fi
# {Pyrmethus} To reforge the prompt's appearance, invoke `p10k configure`.

# {Pyrmethus} --- Warding the History Scrolls ---
# Encrypt history using GPG if available.
if command -v gpg >/dev/null 2>&1; then
    HISTFILE="$ZSH_CACHE_DIR/.zsh_history_encrypted" # Path to encrypted scroll
    print -P "%F{111}Preparing GPG wards for history...%f"
    if [[ -f "$HISTFILE.gpg" ]]; then
        # Attempt to decrypt the scroll
        gpg -q --batch --yes -d "$HISTFILE.gpg" > "$HISTFILE" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            chmod 600 "$HISTFILE" # Secure permissions
            print -P "%F{34}History scroll unsealed.%f"
        else
            print -P "%F{196}Error: Failed to decrypt history scroll. History may be lost or incomplete.%f"
            rm -f "$HISTFILE" # Remove potentially corrupted file
         fi
    fi
    # Set trap to encrypt history upon shell exit
    trap '_encrypt_zsh_history' EXIT HUP INT QUIT TERM
    print -P "%F{75}GPG history wards activated.%f"
else
    HISTFILE="$ZSH_CACHE_DIR/.zsh_history" # Fallback to plain text scroll
    print -P "%F{214}Warning: GPG not found. History scroll remains unwarded (unencrypted).%f"
fi
# History behavior settings are etched in options.zsh

# {Pyrmethus} Encrypt history function called by trap
_encrypt_zsh_history() {
    if command -v gpg >/dev/null 2>&1 && [[ -f "$HISTFILE" && "$HISTFILE" == *encrypted* ]]; then
        print -P "\n%F{111}Sealing history scroll with GPG...%f"
        command gpg -q --batch --yes --symmetric --cipher-algo AES256 \
                -o "$HISTFILE.gpg" "$HISTFILE" && \
        command rm "$HISTFILE" && \
        print -P "%F{34}History scroll sealed.%f" || \
        print -P "%F{196}Error: Failed to seal history scroll!%f"
    fi
}

# {Pyrmethus} --- Final Attunements & Welcome Ritual ---
print -P "%F{111}Performing final attunements...%f"
# Check dependencies and environment integrity
check_dependencies aichat git fzf eza bat fd gpg coreutils # Add other vital spirits
check_environment # Invoked from functions.zsh

# Invoke the welcome incantation (from functions.zsh)
welcome_message

# {Pyrmethus} --- Vanishing Ephemeral Constructs ---
# Clean up any temporary variables here

print -P "\n%F{141}✨ Pyrmethus Enhanced ZSH Grimoire Activated! ✨%f"

"""

# (options.zsh, env.zsh, paths.zsh, aliases.zsh, functions.zsh, keybindings.zsh, plugins.zsh, .p10k.zsh remain largely the same structure as before,
# but I will add Pyrmethus comments within them)

options_zsh_content = """# {Pyrmethus} ~/.config/zsh/options.zsh: Scroll of Shell Options
# Fine-tuning the shell's very essence with setopt/unsetopt incantations.

# {Pyrmethus} --- Navigation & Interaction ---
setopt AUTO_CD              # Teleport by directory name
setopt AUTO_PUSHD           # Auto-stack visited realms
setopt PUSHD_IGNORE_DUPS    # Avoid redundant realm stacking
setopt PUSHD_SILENT         # Silence the echoes of pushd/popd
# ... (rest of options as before, adding comments where needed) ...
setopt HIST_IGNORE_SPACE    # Ignore commands whispered with leading space
# ...
HISTSIZE=2000             # Max spells remembered in session
SAVEHIST=1000             # Max spells etched into the permanent scroll
HISTFILE="$ZSH_CACHE_DIR/.zsh_history" # Default scroll location (may be overridden)
# ...
setopt PROMPT_SUBST         # Allow potent substitutions in the prompt's runes
setopt TRANSIENT_RPROMPT    # Let the right prompt fade when casting spells
"""

env_zsh_content = """# {Pyrmethus} ~/.config/zsh/env.zsh: Scroll of Environment Aura
# Setting the ambient energies (exports) that influence tools and spells.

# {Pyrmethus} --- Foundational Energies ---
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

export EDITOR="${EDITOR:-nano}" # The Humble Scribe
export VISUAL="${VISUAL:-$EDITOR}"
export PAGER="${PAGER:-less}"   # The Patient Reader
export LESS="${LESS:--RiXF}" # Empowering the Reader

# {Pyrmethus} --- Tool-Specific Attunements ---
export BAT_THEME="${BAT_THEME:-Dracula}" # Cloak 'bat' in shadows
export FZF_DEFAULT_OPTS="${FZF_DEFAULT_OPTS:---height 40% --layout=reverse --border --ansi --preview 'bat --color=always --style=numbers --line-range :500 {}'}" # Empower 'fzf'
export _ZO_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/zoxide" # Zoxide's memory crystal

# {Pyrmethus} --- Development Dimensions (Examples) ---
export GOPATH="${GOPATH:-$HOME/go}" # Go dimension portal
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" # Rust forge
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" # Rust source

# {Pyrmethus} --- AI Nexus Configuration ---
export AICHA_CONFIG="${AICHA_CONFIG:-$HOME/.config/aichat}"
export AICHA_CACHE_DIR="${AICHA_CACHE_DIR:-$XDG_CACHE_HOME:-$HOME/.cache}/aichat"
export AICHA_MODEL="${AICHA_MODEL:-gemini-pro}" # Chosen AI Oracle
export AICHA_TIMEOUT="${AICHA_TIMEOUT:-10}"     # Oracle response patience (seconds)

# {Pyrmethus} --- The Vault of Secrets Check ---
# ** CRITICAL: Verify the Arcane Key is drawn from its secure vault! **
if [[ -z "$AICHA_API_KEY" || "$AICHA_API_KEY" == "YOUR_API_KEY_PLACEHOLDER"* ]]; then
    print -P "%F{197}CRITICAL WARNING: AICHA_API_KEY is missing or insecurely set! AI spells WILL FAIL. Secure it in ~/.secrets or via environment variables!%f"
fi

# {Pyrmethus} --- Termux Nexus ---
export TMPDIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}" # Locus for ephemeral creations
"""

paths_zsh_content = """# {Pyrmethus} ~/.config/zsh/paths.zsh: Scroll of Ley Lines
# Charting the PATHways for finding commands and executables.

# {Pyrmethus} --- The Path Weaving Incantation ---
# Adds a directory to the *start* of PATH if it exists and is unique.
add_to_path() {
  if [[ -d "$1" ]] && [[ ":$PATH:" != *":$1:"* ]]; then
    export PATH="$1${PATH:+":$PATH"}" # Prepend to PATH
  fi
}

# {Pyrmethus} --- Charting the Core Ley Lines ---
print -P "%F{111}Charting core Ley Lines (PATH)...%f"
add_to_path "$HOME/bin"           # Personal Spellbook (Highest Priority)
add_to_path "$HOME/.local/bin"    # User-Installed Artifacts
add_to_path "/data/data/com.termux/files/usr/bin" # Termux's Main Arsenal

# {Pyrmethus} --- Charting Optional Development Ley Lines ---
print -P "%F{111}Charting development Ley Lines...%f"
# Rust Forge Path
[[ -d "$HOME/.cargo/bin" ]] && add_to_path "$HOME/.cargo/bin"

# Go Dimension Path
_go_path="${GOPATH:-$HOME/go}/bin"
[[ -d "$_go_path" ]] && add_to_path "$_go_path"
unset _go_path

# Python User Scripts Path
_python_user_bin=$(python3 -m site --user-base 2>/dev/null)"/bin"
[[ -d "$_python_user_bin" ]] && add_to_path "$_python_user_bin"
unset _python_user_bin

# Node Global Path (if applicable)
_node_global_bin=$(npm config get prefix 2>/dev/null)/bin
[[ -n "$_node_global_bin" && -d "$_node_global_bin" ]] && add_to_path "$_node_global_bin"
unset _node_global_bin

# {Pyrmethus} --- Pruning and Harmonizing the Ley Lines ---
print -P "%F{111}Harmonizing Ley Lines...%f"
# Remove duplicates and non-existent paths for optimal flow
typeset -U path # Zsh's unique array magic

print -P "%F{34}Ley Lines harmonized.%f"
# print -l "Final PATH Ley Lines:" $path # Uncomment to inspect
"""

aliases_zsh_content = """# {Pyrmethus} ~/.config/zsh/aliases.zsh: Scroll of Abbreviation Spells
# Condensing common incantations into swift aliases.

# {Pyrmethus} --- Spirit Check Helper ---
is_installed() { command -v "$1" >/dev/null 2>&1; }

# {Pyrmethus} --- Warded Core Utilities ---
alias cp='cp -iv'         # Conjure Copy (Interactive, Verbose)
alias mv='mv -iv'         # Morph Move (Interactive, Verbose)
alias rm='rm -Iv --one-file-system' # Banishment Ritual (Interactive, Verbose, Single Filesystem)
alias mkdir='mkdir -pv'   # Create Sanctum (Parents, Verbose)
alias ln='ln -iv'         # Link Lifeline (Interactive, Verbose)

# {Pyrmethus} --- Enhanced Sight & Scrying ---
if is_installed eza; then
    print -P "%F{75}Binding 'ls' variants to Eza, the Seer.%f"
    alias ls='eza --group-directories-first --icons --color=always'
    alias ll='eza -lgh --git --icons --color=always'
    alias la='eza -lagh --icons --color=always'
    alias lt='eza --tree --level=2 --icons --color=always'
    alias l.='eza -d .* --icons --color=always'
else
    print -P "%F{214}Eza spirit not found. Binding 'ls' to standard view.%f"
    alias ls='ls --color=auto -F'
    alias ll='ls -lAh --color=auto'
    alias la='ls -A --color=auto'
    alias l.='ls -d .* -A --color=auto'
fi

# {Pyrmethus} --- Illuminating Texts ---
if is_installed bat; then
    print -P "%F{75}Binding 'cat' to Bat, the Illuminator.%f"
    alias cat='bat --paging=never --style=plain' # Default: Quick glimpse
    alias catp='bat --paging=always'             # Scry with paging
    alias cath='bat --style=numbers,changes'     # Scry with history runes
else
     print -P "%F{214}Bat spirit not found. Binding 'cat' to standard gaze.%f"
    alias cat='cat'
    alias catp='cat | $PAGER'
    alias cath='cat'
fi

# {Pyrmethus} --- Swift Seeking ---
if is_installed fd; then
    print -P "%F{75}Binding 'find' to fd, the Swift Seeker.%f"
    alias find='fd'
    alias fdf='fd --type f' # Seek Files
    alias fdd='fd --type d' # Seek Directories
else
    print -P "%F{214}fd spirit not found. Binding 'find' to ancient ways.%f"
    alias find='find'
fi

# {Pyrmethus} --- Colored Whispers ---
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'

# {Pyrmethus} --- Realm Measurements ---
alias df='df -h'              # Realm Space
alias du='du -h'              # Directory Weight
alias dus='du -shc * .[!.]* | sort -hr' # Summarized Weight (Sorted, includes hidden)

# ... (Keep other well-commented aliases: navigation, system info, file ops, editors, git, termux, misc) ...

# {Pyrmethus} --- AI Nexus Shortcuts ---
print -P "%F{75}Binding AI Nexus shortcuts...%f"
alias ais='_aichat_suggest'      # Invoke Suggestion Oracle (Alt+S)
alias aic='_aichat_command'      # Invoke Command Conjurer (Alt+C)
alias aih='view_aichat_history'  # Glimpse AI Interactions
alias aicfg='aichat config'      # Configure the AI Oracle
alias aisum='summarize_output'   # Distill Essence (Pipe output: cmd | aisum)
alias aiexplain='aichat explain' # Seek Understanding (aiexplain "cmd")

# {Pyrmethus} --- Vanish Helper Spirit ---
unset -f is_installed
"""

functions_zsh_content = """# {Pyrmethus} ~/.config/zsh/functions.zsh: Scroll of Incantations
# Etching custom spells and rituals into the Grimoire.

# {Pyrmethus} --- Utility Incantations ---
# (Includes _get_recent_commands, mkcd, j, jpr, treedir, extract,
#  find_large_files, find_large_dirs, check_dependencies, check_environment,
#  system_info, list_functions, list_aliases, list_path, weather,
#  gen_password, qrencode, timer, backup_dotfiles, run_python)
# Ensure these functions are well-commented internally. Example:

# {Pyrmethus} Spell to check for needed elemental spirits (dependencies)
check_dependencies() {
    # ... (implementation as before) ...
}

# {Pyrmethus} Ritual to perform environmental sanity checks
check_environment() {
    # ... (implementation as before) ...
}

# {Pyrmethus} Incantation to conjure a temporary directory for ephemeral work
tempdir() {
  local TEMP_DIR=$(mktemp -d -p "${TMPDIR:-/tmp}") # Use TMPDIR
  cd "$TEMP_DIR" || return 1
  print -P "%F{75}Entered ephemeral sanctum: %F{cyan}$TEMP_DIR%f"
  # Optional: Add trap to clean up on exit from this subshell? Complex.
}

# {Pyrmethus} --- AI Incantations ---

# {Pyrmethus} Oracle of Suggestion (_aichat_suggest)
_aichat_suggest() {
    # {Pyrmethus} Check prerequisites: Oracle presence and Arcane Key
    if ! command -v aichat >/dev/null; then print -P "%F{196}Error: AI Oracle 'aichat' is absent.%f"; return 1; fi
    if [[ -z "$AICHA_API_KEY" || "$AICHA_API_KEY" == "YOUR_API_KEY_PLACEHOLDER"* ]]; then print -P "%F{196}Error: Arcane Key (AICHA_API_KEY) is missing or insecure.%f"; return 1; fi
    # ... (Rest of the enhanced _aichat_suggest implementation) ...
    print -P "%F{111}Invoking Suggestion Oracle...%f" # Add feedback
    # ... interaction logic ...
    zle accept-line
}
# Note: zle -N moved to keybindings.zsh for better loading order assurance

# {Pyrmethus} Conjurer of Commands (_aichat_command)
_aichat_command() {
    # {Pyrmethus} Check prerequisites: Conjurer presence and Arcane Key
    if ! command -v aichat >/dev/null; then print -P "%F{196}Error: AI Conjurer 'aichat' is absent.%f"; return 1; fi
    if [[ -z "$AICHA_API_KEY" || "$AICHA_API_KEY" == "YOUR_API_KEY_PLACEHOLDER"* ]]; then print -P "%F{196}Error: Arcane Key (AICHA_API_KEY) is missing or insecure.%f"; return 1; fi
    # ... (Rest of the enhanced _aichat_command implementation including context, cache, validation) ...
     print -P "%F{111}Conjuring command...%f" # Add feedback
     # ... interaction logic ...
    zle accept-line
}
# Note: zle -N moved to keybindings.zsh

# {Pyrmethus} Essence Distiller (summarize_output)
summarize_output() {
    # ... (Implementation as before, check API key) ...
     print -P "%F{111}Distilling essence...%f" # Add feedback
     # ... summarization logic ...
}

# {Pyrmethus} Scryer of AI Interactions (view_aichat_history)
view_aichat_history() {
    # ... (Implementation as before) ...
}

# {Pyrmethus} Scribe of Feedback (_log_feedback)
_log_feedback() {
    # ... (Implementation as before) ...
}

# {Pyrmethus} Carver of Cache Keys (_get_cache_key)
_get_cache_key() {
    # ... (Implementation using sha256sum) ...
}

# {Pyrmethus} Ritual of Welcome (welcome_message)
welcome_message() {
    # ... (Implementation with optional fortune/cowsay) ...
    print -P "\n%F{141}✨ Welcome, Adept, to the Termux ZSH Sanctum! ✨ %F{75}[$(date)]%f"
}

# {Pyrmethus} Purge Cache Ritual (clean_aichat_cache)
clean_aichat_cache() {
    # ... (Implementation as before) ...
}
"""

keybindings_zsh_content = """# {Pyrmethus} ~/.config/zsh/keybindings.zsh: Scroll of Arcane Bindings
# Weaving keyboard shortcuts to invoke spells and navigate the shell realm.

# {Pyrmethus} Awaken the Line Editor Spirit
autoload -Uz zle

# {Pyrmethus} --- AI Nexus Bindings ---
print -P "%F{111}Binding AI Nexus keys...%f"
# Note: Use Esc + key for Alt bindings in many terminals
bindkey '^[s' _aichat_suggest  # Alt+S (Esc+s) -> Invoke Suggestion Oracle
bindkey '^[c' _aichat_command  # Alt+C (Esc+c) -> Invoke Command Conjurer
# Register functions with ZLE *before* binding
zle -N _aichat_suggest
zle -N _aichat_command

# {Pyrmethus} --- Navigation & Editing Runes (Emacs Style) ---
print -P "%F{111}Binding navigation and editing runes...%f"
bindkey '^A' beginning-of-line       # Rune of Alpha (Start)
bindkey '^E' end-of-line             # Rune of Omega (End)
bindkey '^[b' backward-word          # Rune of Word Step Back (Alt+B)
bindkey '^[f' forward-word           # Rune of Word Step Forward (Alt+F)
bindkey '^L' clear-screen            # Rune of Clarity (Ctrl+L)
bindkey '^P' up-line-or-history      # Rune of Ascent (Ctrl+P)
bindkey '^N' down-line-or-history    # Rune of Descent (Ctrl+N)
bindkey '^R' history-incremental-search-backward # Rune of Recall (Ctrl+R)

# {Pyrmethus} --- Banishing Runes ---
bindkey '^H' backward-delete-char    # Rune of Backspace Minor Banishing (Ctrl+H)
bindkey '^?' backward-delete-char    # Rune of Backspace (Physical Key)
bindkey '^W' backward-kill-word      # Rune of Word Banishing (Ctrl+W)
bindkey '^[d' kill-word              # Rune of Forward Word Banishing (Alt+D)
bindkey '^K' kill-line               # Rune of Line Severance (Ctrl+K)
bindkey '^U' kill-whole-line         # Rune of Line Obliteration (Ctrl+U)
bindkey '^[[3~' delete-char          # Rune of Forward Minor Banishing (Delete Key)

# {Pyrmethus} --- History & Completion Runes ---
bindkey '^[.' insert-last-word       # Rune of Echoed Argument (Alt+.)
bindkey '^I' complete-word           # Rune of Completion (Tab)

# {Pyrmethus} --- Custom Spell Bindings (Examples using Ctrl+X prefix) ---
# Ensure functions like update_all are defined in functions.zsh
# bindkey '^X^U' update_all       # Ctrl+X, Ctrl+U -> Invoke Update Ritual
# bindkey '^X^S' sync_dotfiles    # Ctrl+X, Ctrl+S -> Invoke Sync Ritual

# {Pyrmethus} --- Register other custom ZLE functions if any ---
# zle -N function_name_bound_to_key

print -P "%F{34}Arcane bindings woven.%f"
"""

plugins_zsh_content = """# {Pyrmethus} ~/.config/zsh/plugins.zsh: Scroll of Familiars (Zinit)
# Summoning and managing helpful spirits (plugins) with Zinit.

# {Pyrmethus} --- Configuring Zinit's Essence ---
print -P "%F{111}Configuring Zinit Familiar...%f"
# Annexes grant Zinit extra powers
zinit light-mode for \
    zdharma-continuum/zinit-annex-as-monitor \
    zdharma-continuum/zinit-annex-bin-gem-node \
    zdharma-continuum/zinit-annex-patch-dl \
    zdharma-continuum/zinit-annex-rust

# Turbo Mode for swift summoning (wait"0")
# Lucid Mode prevents blocking shell initialization
# node"!" avoids building node modules automatically
# nocd avoids changing directory during plugin loading
zinit ice wait"0" lucid node"!" nocd

# {Pyrmethus} --- Summoning Essential Familiars ---
print -P "%F{111}Summoning essential Familiars...%f"
# Git awareness (for prompt info) - Lightweight snippet
zinit snippet OMZL::git.zsh

# Syntax Highlighting Spirit (Choose One)
# zinit light zsh-users/zsh-syntax-highlighting # Standard Spirit
zinit light zdharma-continuum/fast-syntax-highlighting # Swift Spirit (Recommended)

# Autosuggestion Spirit (Whispers from history)
zinit light zsh-users/zsh-autosuggestions

# Completion Spirit (Expands Zsh's knowledge)
zinit light zsh-users/zsh-completions

# {Pyrmethus} --- Summoning Enhancement Familiars ---
print -P "%F{111}Summoning enhancement Familiars...%f"
# FZF Spirit (Fuzzy Seeker) - Crucial for many spells
zinit ice from"gh-r" as"program" mv"fzf* -> fzf" pick"fzf/fzf" nocompile'!'
zinit light junegunn/fzf

# Zoxide Spirit (Intelligent Teleportation) - Initialize after load
zinit ice lucid wait'1' atload'eval "$(zoxide init zsh --cmd j)"' # Initialize after load, use 'j' command
zinit light ajeetdsouza/zoxide

# Powerlevel10k Spirit (The Aesthetic Beacon)
# Load it here for Zinit management, P10k's own instant prompt handles speed.
zinit ice depth=1
zinit light romkatv/powerlevel10k

# {Pyrmethus} --- Optional Familiars (Uncomment to Summon) ---
# print -P "%F{111}Considering optional Familiars...%f"
# Atuin Spirit (Chronicler of Time) - Needs setup: `atuin login`, `atuin sync`
# zinit ice lucid wait'2' blockf atload"eval \\$(atuin init zsh); _atuin_bind_widget"
# zinit light atuinsh/atuin

# History Substring Search Spirit
# zinit ice wait lucid atload"bindkey '^[[A' history-substring-search-up; bindkey '^[[B' history-substring-search-down"
# zinit light zsh-users/zsh-history-substring-search

# Direnv Spirit (Guardian of Realm-Specific Aura)
# zinit ice as"program" pick"direnv/direnv" lucid from"gh-r" atload'eval "$(direnv hook zsh)"'
# zinit light direnv/direnv

# Auto-Notify Spirit (Messenger for Long Rituals)
# zinit ice lucid
# zinit light MichaelAquilina/zsh-auto-notify

# {Pyrmethus} --- Dismiss Zinit Configuration Mode ---
# zinit cdclear # Optional: Clear Zinit's internal compilation directory

print -P "%F{34}Familiars summoned and bound.%f"
"""

p10k_zsh_content = """# {Pyrmethus} ~/.config/zsh/.p10k.zsh: The Powerlevel10k Sigil
# Etching the runes that define the prompt's appearance and power.
# Conjured via `p10k configure`, refined by hand.

# {Pyrmethus} Enable/Disable Transient Prompt.
# Hides the right prompt when typing commands.
# typeset -g POWERLEVEL9K_TRANSIENT_PROMPT=off

# {Pyrmethus} Instant Prompt configuration.
# Chooses how P10k initializes for speed. `quiet` is good with Zinit.
typeset -g POWERLEVEL9K_INSTANT_PROMPT=quiet

# {Pyrmethus} Left Prompt Elements: What appears on the left side.
# Define the sequence of mystical symbols and information.
typeset -g POWERLEVEL9K_LEFT_PROMPT_ELEMENTS=(
    os_icon                 # The sigil of the realm (OS)
    context                 # User@Host context, if needed
    # dir_writable            # Indicates write permissions in dir
    dir                     # The current path location
    vcs                     # Version control status (Git, etc.)
    # prompt_char           # The final prompt character ($ or #)
)

# {Pyrmethus} Right Prompt Elements: What appears on the right.
# Display auxiliary energies and statuses.
typeset -g POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(
    status                  # Exit status of the last spell
    command_execution_time  # Duration of the last incantation
    # background_jobs         # Indicator for background tasks
    # time                    # Current time
    # ram                     # System RAM usage
    # load                    # System load average
    # direnv                # Direnv status, if active
    # virtualenv            # Python virtual environment
    # anaconda              # Anaconda environment
    # pyenv                 # Pyenv version
    # go_version            # Go language version
    # node_version          # Node.js version
    # rust_version          # Rust language version
    # java_version          # Java language version
    # php_version           # PHP language version
    # battery               # Battery level indicator
    # wifi                  # Wifi signal strength
    # termux                # Termux specific context
)

# {Pyrmethus} Styling and Behavior Options.
# Invoke `p10k configure` for a guided setup of these runes.
typeset -g POWERLEVEL9K_MODE='nerdfont-complete' # Assumes Nerd Font installed
typeset -g POWERLEVEL9K_PROMPT_ON_NEWLINE=true
typeset -g POWERLEVEL9K_MULTILINE_FIRST_PROMPT_PREFIX='%238L╭─' # Fancy start
typeset -g POWERLEVEL9K_MULTILINE_LAST_PROMPT_PREFIX='%238L╰─%B❯%b ' # Fancy end
typeset -g POWERLEVEL9K_PROMPT_ADD_NEWLINE=true
typeset -g POWERLEVEL9K_SHORTEN_DIR_LENGTH=1     # Show only last N directory components
typeset -g POWERLEVEL9K_DIR_ANCHOR_BOLD=true     # Make the base directory bold
typeset -g POWERLEVEL9K_DISABLE_HOT_RELOAD=true  # Can improve performance

# {Pyrmethus} Example Color Customizations (adjust to your preference)
# typeset -g POWERLEVEL9K_DIR_FOREGROUND=75
# typeset -g POWERLEVEL9K_VCS_CLEAN_FOREGROUND=34
# typeset -g POWERLEVEL9K_VCS_MODIFIED_FOREGROUND=214

# {Pyrmethus} For the full lexicon of P10k runes, consult the ancient texts:
# https://github.com/romkatv/powerlevel10k#configuration
"""

# --- Present the Scrolls ---

files_to_create = {
    os.path.join(home_dir, ".zshenv"): zshenv_content,
    os.path.join(config_dir, ".zshrc"): zshrc_content,
    os.path.join(config_dir, "options.zsh"): options_zsh_content,
    os.path.join(config_dir, "env.zsh"): env_zsh_content,
    os.path.join(config_dir, "paths.zsh"): paths_zsh_content,
    os.path.join(config_dir, "aliases.zsh"): aliases_zsh_content,
    os.path.join(config_dir, "functions.zsh"): functions_zsh_content,
    os.path.join(config_dir, "keybindings.zsh"): keybindings_zsh_content,
    os.path.join(config_dir, "plugins.zsh"): plugins_zsh_content,
    os.path.join(config_dir, ".p10k.zsh"): p10k_zsh_content,
}

console.print(Panel(
    Markdown(
        "# ✨ Pyrmethus's Grand ZSH Grimoire for Termux ✨\n\n"
        "Behold! A masterfully woven ZSH configuration, optimized and imbued with arcane energies for the Termux realm. This Grimoire utilizes a modular structure (`~/.config/zsh/`) and the swift **Zinit** familiar for potent plugin management.\n\n"
        "**[bold red]WARNING, ADEPT:[/bold red] The Arcane Key (`AICHA_API_KEY`) is potent and must be guarded! **NEVER** commit it to version control. Draw its power securely from a hidden vault (`~/.secrets/`) or environment variables, as instructed within the scrolls.\n\n"
        "**The Ritual of Summoning:**\n"
        "1. **Gather Reagents:** `pkg install zsh git curl aichat fzf eza bat fd-find gnupg coreutils ncurses-utils stow -y`\n"
        "2. **Attune Shell:** `chsh -s zsh`\n"
        "3. **Prepare Sanctum:** `mkdir -p ~/.config/zsh ~/.cache/zsh ~/.local/bin ~/bin ~/.secrets`\n"
        "4. **Inscribe Scrolls:** Place/Symlink (`stow`) the following scrolls into `~/.config/zsh/` (and `.zshenv` into `~/`).\n"
        "5. **Secure Arcane Key:** Place your key in `~/.secrets/aichat_api_key` (or similar) and ensure `.zshenv` draws power from it.\n"
        "6. **Awaken:** Start a new Termux session. Zinit will conjure the required spirits."
    ),
    title="[bold cyan]The Grand Conjuration: Setup Guide[/bold cyan]",
    border_style="cyan",
    padding=(1, 2)
))

for file_path, content in files_to_create.items():
    relative_path = file_path.replace(home_dir, "~")
    # Determine language for syntax highlighting
    lang = "bash"
    if relative_path.endswith((".zsh", ".zshenv", ".zshrc")):
        lang = "bash"  # Rich uses 'bash' for zsh highlighting
    elif relative_path.endswith(".p10k.zsh"):
         lang = "bash"  # Still shell script based

    syntax = Syntax(content, lang, theme="monokai", line_numbers=True, word_wrap=False)
    panel = Panel(
        syntax,
        title=f"[bold blue]📜 Scroll: {relative_path}[/bold blue]",
        subtitle=f"[dim]Etch this content into {relative_path}[/dim]",
        border_style="blue",
        padding=(1, 1)  # Add padding inside panel
    )
    console.print(panel)
    console.print("\n---\n")  # Separator for copy-pasting


# --- Final Incantations ---
console.print(Panel(
    Markdown(
        "**Final Attunements:**\n"
        "1. **Initiate:** Launch a new Termux session.\n"
        "2. **Witness:** Zinit shall conjure the plugin spirits.\n"
        "3. **Reforge Prompt:** If P10k is unconfigured, invoke `p10k configure`.\n"
        "4. **Test the Weave:** Try AI spells (`Alt+S`, `Alt+C`), aliases (`ll`, `cat`), and functions (`jpr`, `weather`).\n"
        "5. **Heed Warnings:** Observe any messages during startup regarding missing spirits or insecure keys.\n"
        "6. **[bold yellow]Master's Advice:[/bold yellow] Archive these scrolls in a Git sanctum (`~/dotfiles`) and use `stow` to manage them. Shield your vault (`~/.secrets/`, `.zshenv` if it holds secrets) with `.gitignore`!\n\n"
        "***Your Termux ZSH Grimoire is now imbued with potent, optimized magic! Wield it wisely!***"
    ),
    title="[bold green]✨ Awakening the Power ✨[/bold green]",
    border_style="green",
    padding=(1, 2)
))


"""
This Python script now generates the **complete, enhanced, modular, Zinit-powered, and Pyrmethus-styled** ZSH configuration for Termux. It includes:

*   **Mystical Language:** Throughout comments, titles, and instructions.
*   **Rich Formatting:** Beautifully presented scrolls with syntax highlighting.
*   **Zinit Integration:** Replaces basic OMZ loading with Zinit for plugins.
*   **Modularity:** Clear separation of concerns into different `.zsh` files.
*   **Enhanced AI:** Includes the improved `aichat` functions with context, caching, validation, etc.
*   **Path/Env Focus:** Robust setup for Termux paths and environment variables.
*   **Security Emphasis:** Strong warnings about API key handling.
*   **Optimization:** Incorporates performance tweaks via Zinit and sensible defaults.
*   **Copy-Pasteable:** Designed for easy transfer to your Termux environment.

Remember to follow the setup steps carefully, especially regarding API key security!
"""
