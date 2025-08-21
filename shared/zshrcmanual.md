# Complete .zshrc Configuration with Plugins and Customization

Here's a comprehensive `.zshrc` file that includes popular plugins, themes, and customizations for an enhanced terminal experience.

## Full .zshrc Configuration

```bash
# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded
ZSH_THEME="powerlevel10k/powerlevel10k"

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
HYPHEN_INSENSITIVE="true"

# Uncomment the following line to enable command auto-correction.
ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# History configuration
HIST_STAMPS="yyyy-mm-dd"
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=~/.zsh_history

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
plugins=(
    git
    docker
    docker-compose
    kubectl
    terraform
    aws
    npm
    node
    python
    pip
    virtualenv
    golang
    rust
    tmux
    fzf
    z
    colored-man-pages
    command-not-found
    sudo
    history
    extract
    aliases
    common-aliases
    jsontools
    web-search
    copypath
    copyfile
    copybuffer
    dirhistory
    zsh-autosuggestions
    zsh-syntax-highlighting
    fast-syntax-highlighting
    zsh-history-substring-search
    zsh-vi-mode
)

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

# User configuration

# Export paths
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"

# Preferred editor for local and remote sessions
if [[ -n $SSH_CONNECTION ]]; then
  export EDITOR='vim'
else
  export EDITOR='nvim'
fi

# Set language environment
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Compilation flags
export ARCHFLAGS="-arch x86_64"

# ============================================================================
# ALIASES
# ============================================================================

# Navigation aliases
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias .....="cd ../../../.."
alias ~="cd ~"
alias -- -="cd -"

# List directory contents
alias ls='ls --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias lt='ls -ltr'
alias lh='ls -lh'
alias lS='ls -lS'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gpl='git pull'
alias gco='git checkout'
alias gb='git branch'
alias glog='git log --oneline --decorate --graph'
alias gdiff='git diff'
alias gstash='git stash'

# Docker aliases
alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias drm='docker rm'
alias drmi='docker rmi'
alias dexec='docker exec -it'
alias dlogs='docker logs'
alias dprune='docker system prune -a'

# System aliases
alias update='sudo apt update && sudo apt upgrade'
alias install='sudo apt install'
alias search='apt search'
alias ports='netstat -tulanp'
alias meminfo='free -h'
alias cpuinfo='lscpu'
alias diskinfo='df -h'
alias processinfo='ps aux | grep'

# File operations
alias cp='cp -iv'
alias mv='mv -iv'
alias rm='rm -Iv'
alias mkdir='mkdir -pv'
alias chmod='chmod -v'
alias chown='chown -v'

# Utility aliases
alias h='history'
alias j='jobs -l'
alias which='type -a'
alias path='echo -e ${PATH//:/\\n}'
alias now='date +"%T"'
alias nowdate='date +"%d-%m-%Y"'
alias vi='vim'
alias svi='sudo vi'
alias edit='vim'
alias ping='ping -c 5'
alias fastping='ping -c 100 -s.2'
alias ports='netstat -tulanp'
alias wget='wget -c'
alias top='htop'
alias df='df -H'
alias du='du -ch'
alias free='free -m'

# Quick config editing
alias zshconfig="$EDITOR ~/.zshrc"
alias ohmyzsh="$EDITOR ~/.oh-my-zsh"
alias reload="source ~/.zshrc && echo 'ZSH config reloaded'"

# ============================================================================
# FUNCTIONS
# ============================================================================

# Create directory and cd into it
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# Extract archives
extract() {
    if [ -f $1 ] ; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)     echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Find file by name
ff() {
    find . -type f -name "*$1*"
}

# Find directory by name
fd() {
    find . -type d -name "*$1*"
}

# Git commit with message
gcm() {
    git commit -m "$1"
}

# Docker cleanup
docker-cleanup() {
    docker rm $(docker ps -a -q -f status=exited)
    docker rmi $(docker images -q -f dangling=true)
    docker volume rm $(docker volume ls -q -f dangling=true)
}

# Show most used commands
histtop() {
    history | awk '{print $2}' | sort | uniq -c | sort -rn | head -20
}

# ============================================================================
# PLUGIN CONFIGURATIONS
# ============================================================================

# ZSH Autosuggestions configuration
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=#666666"
ZSH_AUTOSUGGEST_STRATEGY=(history completion)
ZSH_AUTOSUGGEST_BUFFER_MAX_SIZE=20
bindkey '^[[Z' autosuggest-accept  # Shift+Tab to accept suggestion

# FZF configuration
export FZF_DEFAULT_COMMAND='find . -type f ! -path "*/\.git/*"'
export FZF_DEFAULT_OPTS='--height 40% --layout=reverse --border'
export FZF_CTRL_T_OPTS="--preview 'bat --style=numbers --color=always {}' --preview-window=right:60%"
export FZF_ALT_C_OPTS="--preview 'tree -C {} | head -200'"

# The fuck configuration
eval $(thefuck --alias)
alias f='fuck'

# Vi mode configuration
bindkey -v
export KEYTIMEOUT=1

# Change cursor shape for different vi modes.
function zle-keymap-select {
  if [[ ${KEYMAP} == vicmd ]] ||
     [[ $1 = 'block' ]]; then
    echo -ne '\e[1 q'
  elif [[ ${KEYMAP} == main ]] ||
       [[ ${KEYMAP} == viins ]] ||
       [[ ${KEYMAP} = '' ]] ||
       [[ $1 = 'beam' ]]; then
    echo -ne '\e[5 q'
  fi
}
zle -N zle-keymap-select

# Use vim keys in tab complete menu
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history
bindkey -v '^?' backward-delete-char

# History substring search bindings
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey -M vicmd 'k' history-substring-search-up
bindkey -M vicmd 'j' history-substring-search-down

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Node.js
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Python
export PYTHONDONTWRITEBYTECODE=1
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Go
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Rust
export PATH="$HOME/.cargo/bin:$PATH"

# Ruby
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)" 2>/dev/null

# Java
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

# ============================================================================
# COMPLETION SETTINGS
# ============================================================================

# Basic auto/tab complete
autoload -U compinit && compinit
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' rehash true
zstyle ':completion:*' accept-exact '*(N)'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path ~/.zsh/cache

# Kill completion
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'
zstyle ':completion:*:kill:*' command 'ps -u $USER -o pid,%cpu,tty,cputime,cmd'

# ============================================================================
# PROMPT CUSTOMIZATION (if not using Powerlevel10k)
# ============================================================================

# If you want to use a custom prompt instead of Powerlevel10k, uncomment:
# autoload -Uz vcs_info
# precmd() { vcs_info }
# zstyle ':vcs_info:git:*' formats '%b '
# setopt PROMPT_SUBST
# PROMPT='%F{green}%*%f %F{blue}%~%f %F{red}${vcs_info_msg_0_}%f$ '

# ============================================================================
# ADDITIONAL TOOLS
# ============================================================================

# Load zoxide (better cd)
eval "$(zoxide init zsh)" 2>/dev/null

# Load starship prompt (alternative to powerlevel10k)
# eval "$(starship init zsh)"

# Load direnv
eval "$(direnv hook zsh)" 2>/dev/null

# ============================================================================
# LOCAL CUSTOMIZATIONS
# ============================================================================

# Source local configuration if it exists
[ -f ~/.zshrc.local ] && source ~/.zshrc.local

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
```



## Installation Instructions

To use this configuration, you'll need to install the required components:

### **1. Install Oh My Zsh**
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### **2. Install Powerlevel10k Theme**
```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

### **3. Install Essential Plugins**
```bash
# zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# fast-syntax-highlighting
git clone https://github.com/zdharma-continuum/fast-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting

# zsh-history-substring-search
git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search

# zsh-vi-mode
git clone https://github.com/jeffreytse/zsh-vi-mode ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-vi-mode
```

### **4. Install Additional Tools**
```bash
# FZF (fuzzy finder)
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install

# The Fuck (command correction)
pip install thefuck

# Zoxide (better cd)
curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

# Install fonts for Powerlevel10k
# Download and install a Nerd Font like "MesloLGS NF" from:
# https://github.com/romkatv/powerlevel10k#meslo-nerd-font-patched-for-powerlevel10k
```

## Key Features Included

This configuration provides:

- **Powerlevel10k theme** with instant prompt for blazing-fast startup
- **Auto-suggestions** that predict commands based on history
- **Syntax highlighting** for better command visibility
- **Vi-mode** for vim-like navigation in the terminal
- **FZF integration** for fuzzy finding files and history
- **Git integration** with extensive aliases and status information
- **Docker and Kubernetes** shortcuts for container management
- **History substring search** for finding previous commands
- **Smart tab completion** with case-insensitive matching
- **The Fuck** integration for correcting mistyped commands
- **Extensive aliases** for common operations
- **Custom functions** for productivity enhancement

After saving the `.zshrc` file, run `source ~/.zshrc` to apply the changes, then run `p10k configure` to set up your Powerlevel10k prompt style.


Here's a complete .zshrc file with comprehensive plugins, functionality, and customization:

```bash
# ===============================
# Oh My Zsh Configuration
# ===============================

# Path to your oh-my-zsh installation
export ZSH="$HOME/.oh-my-zsh"

# Theme configuration
ZSH_THEME="powerlevel10k/powerlevel10k"

# Uncomment for random theme each time
# ZSH_THEME="random"

# Update behavior
DISABLE_AUTO_UPDATE="false"
DISABLE_UPDATE_PROMPT="false"
UPDATE_ZSH_DAYS=7

# Plugin configuration
COMPLETION_WAITING_DOTS="true"
DISABLE_UNTRACKED_FILES_DIRTY="false"
HIST_STAMPS="yyyy-mm-dd"

# ===============================
# Plugins
# ===============================

plugins=(
    # Essential plugins
    git
    zsh-syntax-highlighting
    zsh-autosuggestions
    zsh-completions
    
    # Development tools
    docker
    docker-compose
    kubectl
    helm
    terraform
    aws
    gcloud
    
    # Programming languages
    python
    pip
    node
    npm
    rust
    golang
    
    # Productivity
    z
    fzf
    thefuck
    command-not-found
    colored-man-pages
    extract
    
    # Git extras
    git-flow
    github
    gitignore
    
    # System utilities
    sudo
    systemd
    ubuntu
    macos
    brew
    
    # Shell enhancements
    history-substring-search
    vi-mode
    safe-paste
    copypath
    copyfile
    copybuffer
    dirhistory
    jsontools
    urltools
    web-search
)

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

# ===============================
# Environment Variables
# ===============================

# Default programs
export EDITOR="nvim"
export VISUAL="nvim"
export PAGER="less"
export BROWSER="firefox"
export TERMINAL="alacritty"

# Language settings
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

# Development paths
export GOPATH="$HOME/go"
export GOBIN="$GOPATH/bin"
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export NVM_DIR="$HOME/.nvm"
export PYENV_ROOT="$HOME/.pyenv"

# XDG Base Directory
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_STATE_HOME="$HOME/.local/state"

# Other configurations
export DOTFILES="$HOME/.dotfiles"
export PROJECTS="$HOME/projects"

# ===============================
# PATH Configuration
# ===============================

# Add custom paths
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"
export PATH="$GOBIN:$PATH"
export PATH="$CARGO_HOME/bin:$PATH"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="/usr/local/sbin:$PATH"

# ===============================
# History Configuration
# ===============================

HISTFILE="$HOME/.zsh_history"
HISTSIZE=100000
SAVEHIST=100000
HISTDUP=erase

setopt APPEND_HISTORY
setopt SHARE_HISTORY
setopt HIST_IGNORE_SPACE
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_SAVE_NO_DUPS
setopt HIST_IGNORE_DUPS
setopt HIST_FIND_NO_DUPS
setopt HIST_REDUCE_BLANKS
setopt HIST_VERIFY
setopt INC_APPEND_HISTORY

# ===============================
# Zsh Options
# ===============================

# Directory navigation
setopt AUTO_CD
setopt AUTO_PUSHD
setopt PUSHD_IGNORE_DUPS
setopt PUSHD_SILENT
setopt PUSHD_TO_HOME
setopt CDABLE_VARS

# Completion
setopt ALWAYS_TO_END
setopt COMPLETE_IN_WORD
setopt AUTO_MENU
setopt MENU_COMPLETE
setopt LIST_AMBIGUOUS

# Globbing
setopt EXTENDED_GLOB
setopt GLOB_DOTS
setopt NO_CASE_GLOB

# Correction
setopt CORRECT
setopt CORRECT_ALL

# Job control
setopt NO_BG_NICE
setopt NO_HUP
setopt NO_LIST_BEEP
setopt LOCAL_OPTIONS
setopt LOCAL_TRAPS

# Other options
setopt PROMPT_SUBST
setopt INTERACTIVE_COMMENTS
setopt RC_QUOTES
setopt LONG_LIST_JOBS
setopt NOTIFY
setopt NO_BEEP

# ===============================
# Key Bindings
# ===============================

# Enable Vi mode
bindkey -v
export KEYTIMEOUT=1

# Better searching in command mode
bindkey -M vicmd '?' history-incremental-search-backward
bindkey -M vicmd '/' history-incremental-search-forward

# Edit line in vim with ctrl-e
autoload edit-command-line
zle -N edit-command-line
bindkey '^e' edit-command-line

# Use vim keys in tab complete menu
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history

# History substring search
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey -M vicmd 'k' history-substring-search-up
bindkey -M vicmd 'j' history-substring-search-down

# Common shortcuts
bindkey '^A' beginning-of-line
bindkey '^E' end-of-line
bindkey '^K' kill-line
bindkey '^W' backward-kill-word
bindkey '^R' history-incremental-search-backward
bindkey '^S' history-incremental-search-forward
bindkey '^P' up-history
bindkey '^N' down-history
bindkey '^Y' accept-and-hold
bindkey '^Q' push-line-or-edit
bindkey -s '^O' 'lfcd\n'

# ===============================
# Completion System
# ===============================

# Initialize completion
autoload -Uz compinit && compinit
autoload -Uz bashcompinit && bashcompinit

# Completion styling
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' menu select
zstyle ':completion:*' rehash true
zstyle ':completion:*' accept-exact '*(N)'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$XDG_CACHE_HOME/zsh/zcompcache"

# Completion categories
zstyle ':completion:*:*:*:*:*' menu select
zstyle ':completion:*:matches' group 'yes'
zstyle ':completion:*:options' description 'yes'
zstyle ':completion:*:options' auto-description '%d'
zstyle ':completion:*:corrections' format ' %F{green}-- %d (errors: %e) --%f'
zstyle ':completion:*:descriptions' format ' %F{yellow}-- %d --%f'
zstyle ':completion:*:messages' format ' %F{purple} -- %d --%f'
zstyle ':completion:*:warnings' format ' %F{red}-- no matches found --%f'
zstyle ':completion:*:default' list-prompt '%S%M matches%s'
zstyle ':completion:*' format ' %F{yellow}-- %d --%f'
zstyle ':completion:*' group-name ''
zstyle ':completion:*' verbose yes

# Kill completion
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#) ([0-9a-z-]#)*=01;34=0=01'
zstyle ':completion:*:*:*:*:processes' command "ps -u $USER -o pid,user,comm -w -w"

# ===============================
# Aliases
# ===============================

# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'
alias ~='cd ~'
alias -- -='cd -'

# List directory contents
alias ls='ls --color=auto'
alias l='ls -lFh'
alias la='ls -lAFh'
alias lr='ls -tRFh'
alias lt='ls -ltFh'
alias ll='ls -l'
alias lla='ls -la'
alias lld='ls -ld'
alias ldot='ls -ld .*'
alias lS='ls -1FSsh'
alias lart='ls -1Fcart'
alias lrt='ls -1Fcrt'

# File operations
alias cp='cp -iv'
alias mv='mv -iv'
alias rm='rm -Iv'
alias mkdir='mkdir -pv'
alias rmdir='rmdir -v'

# Archives
alias mktar='tar -cvf'
alias mkbz2='tar -cvjf'
alias mkgz='tar -cvzf'
alias untar='tar -xvf'
alias unbz2='tar -xvjf'
alias ungz='tar -xvzf'

# Git shortcuts
alias g='git'
alias ga='git add'
alias gaa='git add --all'
alias gb='git branch'
alias gc='git commit -v'
alias gcm='git commit -m'
alias gco='git checkout'
alias gd='git diff'
alias gf='git fetch'
alias gl='git pull'
alias glog='git log --oneline --decorate --graph'
alias gp='git push'
alias gst='git status'
alias gs='git status -s'

# Docker shortcuts
alias d='docker'
alias dc='docker-compose'
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dex='docker exec -it'
alias dlog='docker logs'
alias dstop='docker stop $(docker ps -q)'
alias drm='docker rm $(docker ps -aq)'
alias drmi='docker rmi $(docker images -q)'

# System monitoring
alias df='df -h'
alias du='du -h'
alias top='htop'
alias free='free -h'
alias ps='ps auxf'
alias psg='ps aux | grep -v grep | grep -i -e VSZ -e'

# Network
alias ping='ping -c 5'
alias fastping='ping -c 100 -i 0.2'
alias ports='netstat -tulanp'
alias ipinfo='curl ipinfo.io'
alias myip='curl ifconfig.me'

# Package management (Ubuntu/Debian)
alias update='sudo apt update && sudo apt upgrade'
alias install='sudo apt install'
alias search='apt search'
alias remove='sudo apt remove'

# Text editors
alias v='nvim'
alias vi='nvim'
alias vim='nvim'
alias sv='sudo nvim'
alias edit='$EDITOR'

# Python
alias py='python3'
alias pip='pip3'
alias venv='python3 -m venv'
alias activate='source venv/bin/activate'

# Misc utilities
alias h='history'
alias help='man'
alias p='ps -f'
alias sortnr='sort -n -r'
alias unexport='unset'
alias whereami='pwd'
alias j='jobs -l'
alias path='echo -e ${PATH//:/\\n}'
alias reload='source ~/.zshrc'
alias zshrc='$EDITOR ~/.zshrc'
alias please='sudo'
alias fucking='sudo'
alias week='date +%V'
alias timer='echo "Timer started. Stop with Ctrl-D." && date && time cat && date'

# ===============================
# Functions
# ===============================

# Create directory and cd into it
mkcd() {
    mkdir -p "$@" && cd "$_"
}

# Extract archives
extract() {
    if [ -f $1 ]; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)     echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Find file by name
ff() {
    find . -type f -iname "*$1*"
}

# Find directory by name
fd() {
    find . -type d -iname "*$1*"
}

# Git commit with message
gcmsg() {
    git commit -m "$*"
}

# Kill process by name
killp() {
    ps aux | grep -i $1 | grep -v grep | awk '{print $2}' | xargs kill -9
}

# Backup file with timestamp
backup() {
    cp "$1" "$1.backup.$(date +%Y%m%d_%H%M%S)"
}

# Show PATH in readable format
path() {
    echo $PATH | tr ':' '\n' | nl
}

# Weather
weather() {
    curl "wttr.in/${1:-}"
}

# Cheat sheet
cheat() {
    curl "cheat.sh/$1"
}

# Quick note taking
note() {
    echo "$(date): $*" >> ~/notes.txt
}

# Docker cleanup
docker-cleanup() {
    docker system prune -af
    docker volume prune -f
}

# ===============================
# FZF Configuration
# ===============================

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS="
--layout=reverse
--height=40%
--multi
--preview='[[ \$(file --mime {}) =~ binary ]] && echo {} is a binary file || (bat --style=numbers --color=always {} || cat {}) 2> /dev/null | head -300'
--preview-window='right:hidden:wrap'
--bind='f3:execute(bat --style=numbers {} || less -f {}),f2:toggle-preview,ctrl-d:half-page-down,ctrl-u:half-page-up'
"

# fzf shortcuts
alias fzfp='fzf --preview "bat --style=numbers --color=always {}"'
alias fzfv='vim $(fzf)'

# ===============================
# Load Additional Configurations
# ===============================

# Load local configuration if exists
[ -f ~/.zshrc.local ] && source ~/.zshrc.local

# Load work-specific configuration
[ -f ~/.zshrc.work ] && source ~/.zshrc.work

# Load private/secret configuration
[ -f ~/.zshrc.private ] && source ~/.zshrc.private

# ===============================
# Final Configurations
# ===============================

# Enable Powerlevel10k instant prompt
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
    source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# Load syntax highlighting (should be last)
source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null || \
source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null || \
source $HOME/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null

# Welcome message
echo "Welcome back, $USER! 🚀"
echo "Today is $(date '+%A, %B %d, %Y')"
echo "---"
```

## Installation Instructions

To use this configuration effectively, you'll need to install the following dependencies:

**Oh My Zsh Framework**
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Essential Plugins**
```bash
# Zsh Syntax Highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Zsh Autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Zsh Completions
git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions

# Powerlevel10k Theme
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

**Additional Tools**
```bash
# FZF (Fuzzy Finder)
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# Install fd (better find)
# On Ubuntu/Debian:
sudo apt install fd-find

# On macOS:
brew install fd

# Install bat (better cat)
# On Ubuntu/Debian:
sudo apt install bat

# On macOS:
brew install bat
```

## Key Features

This configuration includes comprehensive functionality for:

- **Smart command completion** with case-insensitive matching and menu selection
- **Extensive plugin support** for development tools, programming languages, and productivity enhancements
- **Vi mode editing** with custom key bindings for efficient command-line editing
- **Advanced history management** with substring search and deduplication
- **Docker and Kubernetes integration** with helpful aliases and functions
- **Git workflow enhancements** with shortcuts and visual status indicators
- **FZF integration** for fuzzy file finding and previewing
- **Custom functions** for common tasks like archive extraction, directory creation, and backup management
- **Modular configuration** support for work-specific or private settings
- **Performance optimizations** with lazy loading and instant prompt support

The configuration is designed to be both powerful and maintainable, with clear sections for different aspects of shell customization. You can easily modify or extend it by editing specific sections or adding your own custom configurations in the local files that are sourced at the end.


Based on the search results, there isn't a specific "aichat suggestions widget" mentioned in the standard Zsh plugin ecosystem. However, I can help you set up AI-powered command suggestions and integrate AI chat functionality into your Zsh configuration using custom functions and existing AI tools.

## Setting Up AI-Powered Command Assistance in Zsh

While the search results primarily discuss traditional autosuggestion plugins like `zsh-autosuggestions`, which provides fish-like suggestions based on your command history, you can create a custom AI chat integration for your shell. Here's how to set it up:

### **Custom AI Chat Function**

Add this custom function to your `.zshrc` file to create an AI chat interface:

```bash
# AI Chat function - using OpenAI's API or similar
aichat() {
    local query="$*"
    
    # Using curl to interact with AI API (example with OpenAI)
    # Replace YOUR_API_KEY with your actual API key
    local response=$(curl -s -X POST https://api.openai.com/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer YOUR_API_KEY" \
        -d "{
            \"model\": \"gpt-3.5-turbo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$query\"}],
            \"temperature\": 0.7
        }" | jq -r '.choices.message.content')
    
    echo "$response"
}

# AI command suggestion function
ai_suggest() {
    local current_buffer="$BUFFER"
    local suggestion=$(aichat "Suggest a shell command for: $current_buffer")
    BUFFER="$suggestion"
    zle end-of-line
}

# Register the widget
zle -N ai_suggest
```

### **Key Binding Configuration**

Bind the AI suggestion function to a key combination:

```bash
# Bind Ctrl+G to get AI suggestions
bindkey '^G' ai_suggest

# Alternative: Bind Alt+A for AI assistance
bindkey '\ea' ai_suggest
```

## Enhanced Setup with Traditional Autosuggestions

For the best experience, combine AI assistance with traditional Zsh autosuggestions. The `zsh-autosuggestions` plugin provides suggestions based on your command history and can work alongside your AI integration:

### **Installation of zsh-autosuggestions**

```bash
# Clone the plugin
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

### **Configuration in .zshrc**

```bash
# Add to your plugins list
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
)

# Configure autosuggestion appearance
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=#ff00ff,bg=cyan,bold,underline"

# Set suggestion strategy
ZSH_AUTOSUGGEST_STRATEGY=(history completion)

# Custom key bindings for autosuggestions
bindkey '^ ' autosuggest-accept  # Ctrl+Space to accept suggestion
```

## Alternative AI Integration Using External Tools

If you have AI CLI tools like GitHub Copilot CLI or other AI assistants installed, you can create wrapper functions:

```bash
# GitHub Copilot CLI integration (if installed)
copilot_suggest() {
    local query="$BUFFER"
    local suggestion=$(gh copilot suggest "$query" 2>/dev/null)
    if [[ -n "$suggestion" ]]; then
        BUFFER="$suggestion"
        zle end-of-line
    fi
}
zle -N copilot_suggest
bindkey '^G' copilot_suggest




Here's a complete .zshrc file with comprehensive plugins, functionality, and customization:

```bash
# ===============================
# Oh My Zsh Configuration
# ===============================

# Path to your oh-my-zsh installation
export ZSH="$HOME/.oh-my-zsh"

# Theme configuration
ZSH_THEME="powerlevel10k/powerlevel10k"

# Uncomment for random theme each time
# ZSH_THEME="random"

# Update behavior
DISABLE_AUTO_UPDATE="false"
DISABLE_UPDATE_PROMPT="false"
UPDATE_ZSH_DAYS=7

# Plugin configuration
COMPLETION_WAITING_DOTS="true"
DISABLE_UNTRACKED_FILES_DIRTY="false"
HIST_STAMPS="yyyy-mm-dd"

# ===============================
# Plugins
# ===============================

plugins=(
    # Essential plugins
    git
    zsh-syntax-highlighting
    zsh-autosuggestions
    zsh-completions
    
    # Development tools
    docker
    docker-compose
    kubectl
    helm
    terraform
    aws
    gcloud
    
    # Programming languages
    python
    pip
    node
    npm
    rust
    golang
    
    # Productivity
    z
    fzf
    thefuck
    command-not-found
    colored-man-pages
    extract
    
    # Git extras
    git-flow
    github
    gitignore
    
    # System utilities
    sudo
    systemd
    ubuntu
    macos
    brew
    
    # Shell enhancements
    history-substring-search
    vi-mode
    safe-paste
    copypath
    copyfile
    copybuffer
    dirhistory
    jsontools
    urltools
    web-search
)

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

# ===============================
# Environment Variables
# ===============================

# Default programs
export EDITOR="nvim"
export VISUAL="nvim"
export PAGER="less"
export BROWSER="firefox"
export TERMINAL="alacritty"

# Language settings
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"

# Development paths
export GOPATH="$HOME/go"
export GOBIN="$GOPATH/bin"
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export NVM_DIR="$HOME/.nvm"
export PYENV_ROOT="$HOME/.pyenv"

# XDG Base Directory
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_STATE_HOME="$HOME/.local/state"

# Other configurations
export DOTFILES="$HOME/.dotfiles"
export PROJECTS="$HOME/projects"

# ===============================
# PATH Configuration
# ===============================

# Add custom paths
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"
export PATH="$GOBIN:$PATH"
export PATH="$CARGO_HOME/bin:$PATH"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="/usr/local/sbin:$PATH"

# ===============================
# History Configuration
# ===============================

HISTFILE="$HOME/.zsh_history"
HISTSIZE=100000
SAVEHIST=100000
HISTDUP=erase

setopt APPEND_HISTORY
setopt SHARE_HISTORY
setopt HIST_IGNORE_SPACE
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_SAVE_NO_DUPS
setopt HIST_IGNORE_DUPS
setopt HIST_FIND_NO_DUPS
setopt HIST_REDUCE_BLANKS
setopt HIST_VERIFY
setopt INC_APPEND_HISTORY

# ===============================
# Zsh Options
# ===============================

# Directory navigation
setopt AUTO_CD
setopt AUTO_PUSHD
setopt PUSHD_IGNORE_DUPS
setopt PUSHD_SILENT
setopt PUSHD_TO_HOME
setopt CDABLE_VARS

# Completion
setopt ALWAYS_TO_END
setopt COMPLETE_IN_WORD
setopt AUTO_MENU
setopt MENU_COMPLETE
setopt LIST_AMBIGUOUS

# Globbing
setopt EXTENDED_GLOB
setopt GLOB_DOTS
setopt NO_CASE_GLOB

# Correction
setopt CORRECT
setopt CORRECT_ALL

# Job control
setopt NO_BG_NICE
setopt NO_HUP
setopt NO_LIST_BEEP
setopt LOCAL_OPTIONS
setopt LOCAL_TRAPS

# Other options
setopt PROMPT_SUBST
setopt INTERACTIVE_COMMENTS
setopt RC_QUOTES
setopt LONG_LIST_JOBS
setopt NOTIFY
setopt NO_BEEP

# ===============================
# Key Bindings
# ===============================

# Enable Vi mode
bindkey -v
export KEYTIMEOUT=1

# Better searching in command mode
bindkey -M vicmd '?' history-incremental-search-backward
bindkey -M vicmd '/' history-incremental-search-forward

# Edit line in vim with ctrl-e
autoload edit-command-line
zle -N edit-command-line
bindkey '^e' edit-command-line

# Use vim keys in tab complete menu
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history

# History substring search
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey -M vicmd 'k' history-substring-search-up
bindkey -M vicmd 'j' history-substring-search-down

# Common shortcuts
bindkey '^A' beginning-of-line
bindkey '^E' end-of-line
bindkey '^K' kill-line
bindkey '^W' backward-kill-word
bindkey '^R' history-incremental-search-backward
bindkey '^S' history-incremental-search-forward
bindkey '^P' up-history
bindkey '^N' down-history
bindkey '^Y' accept-and-hold
bindkey '^Q' push-line-or-edit
bindkey -s '^O' 'lfcd\n'

# ===============================
# Completion System
# ===============================

# Initialize completion
autoload -Uz compinit && compinit
autoload -Uz bashcompinit && bashcompinit

# Completion styling
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' menu select
zstyle ':completion:*' rehash true
zstyle ':completion:*' accept-exact '*(N)'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$XDG_CACHE_HOME/zsh/zcompcache"

# Completion categories
zstyle ':completion:*:*:*:*:*' menu select
zstyle ':completion:*:matches' group 'yes'
zstyle ':completion:*:options' description 'yes'
zstyle ':completion:*:options' auto-description '%d'
zstyle ':completion:*:corrections' format ' %F{green}-- %d (errors: %e) --%f'
zstyle ':completion:*:descriptions' format ' %F{yellow}-- %d --%f'
zstyle ':completion:*:messages' format ' %F{purple} -- %d --%f'
zstyle ':completion:*:warnings' format ' %F{red}-- no matches found --%f'
zstyle ':completion:*:default' list-prompt '%S%M matches%s'
zstyle ':completion:*' format ' %F{yellow}-- %d --%f'
zstyle ':completion:*' group-name ''
zstyle ':completion:*' verbose yes

# Kill completion
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#) ([0-9a-z-]#)*=01;34=0=01'
zstyle ':completion:*:*:*:*:processes' command "ps -u $USER -o pid,user,comm -w -w"

# ===============================
# Aliases
# ===============================

# Navigation
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'
alias ~='cd ~'
alias -- -='cd -'

# List directory contents
alias ls='ls --color=auto'
alias l='ls -lFh'
alias la='ls -lAFh'
alias lr='ls -tRFh'
alias lt='ls -ltFh'
alias ll='ls -l'
alias lla='ls -la'
alias lld='ls -ld'
alias ldot='ls -ld .*'
alias lS='ls -1FSsh'
alias lart='ls -1Fcart'
alias lrt='ls -1Fcrt'

# File operations
alias cp='cp -iv'
alias mv='mv -iv'
alias rm='rm -Iv'
alias mkdir='mkdir -pv'
alias rmdir='rmdir -v'

# Archives
alias mktar='tar -cvf'
alias mkbz2='tar -cvjf'
alias mkgz='tar -cvzf'
alias untar='tar -xvf'
alias unbz2='tar -xvjf'
alias ungz='tar -xvzf'

# Git shortcuts
alias g='git'
alias ga='git add'
alias gaa='git add --all'
alias gb='git branch'
alias gc='git commit -v'
alias gcm='git commit -m'
alias gco='git checkout'
alias gd='git diff'
alias gf='git fetch'
alias gl='git pull'
alias glog='git log --oneline --decorate --graph'
alias gp='git push'
alias gst='git status'
alias gs='git status -s'

# Docker shortcuts
alias d='docker'
alias dc='docker-compose'
alias dps='docker ps'
alias dpsa='docker ps -a'
alias dimg='docker images'
alias dex='docker exec -it'
alias dlog='docker logs'
alias dstop='docker stop $(docker ps -q)'
alias drm='docker rm $(docker ps -aq)'
alias drmi='docker rmi $(docker images -q)'

# System monitoring
alias df='df -h'
alias du='du -h'
alias top='htop'
alias free='free -h'
alias ps='ps auxf'
alias psg='ps aux | grep -v grep | grep -i -e VSZ -e'

# Network
alias ping='ping -c 5'
alias fastping='ping -c 100 -i 0.2'
alias ports='netstat -tulanp'
alias ipinfo='curl ipinfo.io'
alias myip='curl ifconfig.me'

# Package management (Ubuntu/Debian)
alias update='sudo apt update && sudo apt upgrade'
alias install='sudo apt install'
alias search='apt search'
alias remove='sudo apt remove'

# Text editors
alias v='nvim'
alias vi='nvim'
alias vim='nvim'
alias sv='sudo nvim'
alias edit='$EDITOR'

# Python
alias py='python3'
alias pip='pip3'
alias venv='python3 -m venv'
alias activate='source venv/bin/activate'

# Misc utilities
alias h='history'
alias help='man'
alias p='ps -f'
alias sortnr='sort -n -r'
alias unexport='unset'
alias whereami='pwd'
alias j='jobs -l'
alias path='echo -e ${PATH//:/\\n}'
alias reload='source ~/.zshrc'
alias zshrc='$EDITOR ~/.zshrc'
alias please='sudo'
alias fucking='sudo'
alias week='date +%V'
alias timer='echo "Timer started. Stop with Ctrl-D." && date && time cat && date'

# ===============================
# Functions
# ===============================

# Create directory and cd into it
mkcd() {
    mkdir -p "$@" && cd "$_"
}

# Extract archives
extract() {
    if [ -f $1 ]; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)     echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Find file by name
ff() {
    find . -type f -iname "*$1*"
}

# Find directory by name
fd() {
    find . -type d -iname "*$1*"
}

# Git commit with message
gcmsg() {
    git commit -m "$*"
}

# Kill process by name
killp() {
    ps aux | grep -i $1 | grep -v grep | awk '{print $2}' | xargs kill -9
}

# Backup file with timestamp
backup() {
    cp "$1" "$1.backup.$(date +%Y%m%d_%H%M%S)"
}

# Show PATH in readable format
path() {
    echo $PATH | tr ':' '\n' | nl
}

# Weather
weather() {
    curl "wttr.in/${1:-}"
}

# Cheat sheet
cheat() {
    curl "cheat.sh/$1"
}

# Quick note taking
note() {
    echo "$(date): $*" >> ~/notes.txt
}

# Docker cleanup
docker-cleanup() {
    docker system prune -af
    docker volume prune -f
}

# ===============================
# FZF Configuration
# ===============================

[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh

export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_DEFAULT_OPTS="
--layout=reverse
--height=40%
--multi
--preview='[[ \$(file --mime {}) =~ binary ]] && echo {} is a binary file || (bat --style=numbers --color=always {} || cat {}) 2> /dev/null | head -300'
--preview-window='right:hidden:wrap'
--bind='f3:execute(bat --style=numbers {} || less -f {}),f2:toggle-preview,ctrl-d:half-page-down,ctrl-u:half-page-up'
"

# fzf shortcuts
alias fzfp='fzf --preview "bat --style=numbers --color=always {}"'
alias fzfv='vim $(fzf)'

# ===============================
# Load Additional Configurations
# ===============================

# Load local configuration if exists
[ -f ~/.zshrc.local ] && source ~/.zshrc.local

# Load work-specific configuration
[ -f ~/.zshrc.work ] && source ~/.zshrc.work

# Load private/secret configuration
[ -f ~/.zshrc.private ] && source ~/.zshrc.private

# ===============================
# Final Configurations
# ===============================

# Enable Powerlevel10k instant prompt
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
    source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh

# Load syntax highlighting (should be last)
source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null || \
source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null || \
source $HOME/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh 2>/dev/null

# Welcome message
echo "Welcome back, $USER! 🚀"
echo "Today is $(date '+%A, %B %d, %Y')"
echo "---"
```

## Installation Instructions

To use this configuration effectively, you'll need to install the following dependencies:

**Oh My Zsh Framework**
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

**Essential Plugins**
```bash
# Zsh Syntax Highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# Zsh Autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Zsh Completions
git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions

# Powerlevel10k Theme
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

**Additional Tools**
```bash
# FZF (Fuzzy Finder)
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# Install fd (better find)
# On Ubuntu/Debian:
sudo apt install fd-find

# On macOS:
brew install fd

# Install bat (better cat)
# On Ubuntu/Debian:
sudo apt install bat

# On macOS:
brew install bat
```

## Key Features

This configuration includes comprehensive functionality for:

- **Smart command completion** with case-insensitive matching and menu selection
- **Extensive plugin support** for development tools, programming languages, and productivity enhancements
- **Vi mode editing** with custom key bindings for efficient command-line editing
- **Advanced history management** with substring search and deduplication
- **Docker and Kubernetes integration** with helpful aliases and functions
- **Git workflow enhancements** with shortcuts and visual status indicators
- **FZF integration** for fuzzy file finding and previewing
- **Custom functions** for common tasks like archive extraction, directory creation, and backup management
- **Modular configuration** support for work-specific or private settings
- **Performance optimizations** with lazy loading and instant prompt support

The configuration is designed to be both powerful and maintainable, with clear sections for different aspects of shell customization. You can easily modify or extend it by editing specific sections or adding your own custom configurations in the local files that are sourced at the end.
# Complete .zshrc Configuration with Plugins and Customization

Here's a comprehensive `.zshrc` file that includes popular plugins, themes, and customizations for an enhanced terminal experience.

## Full .zshrc Configuration

```bash
# Enable Powerlevel10k instant prompt. Should stay close to the top of ~/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load --- if set to "random", it will
# load a random theme each time oh-my-zsh is loaded
ZSH_THEME="powerlevel10k/powerlevel10k"

# Uncomment the following line to use case-sensitive completion.
# CASE_SENSITIVE="true"

# Uncomment the following line to use hyphen-insensitive completion.
# Case-sensitive completion must be off. _ and - will be interchangeable.
HYPHEN_INSENSITIVE="true"

# Uncomment the following line to enable command auto-correction.
ENABLE_CORRECTION="true"

# Uncomment the following line to display red dots whilst waiting for completion.
COMPLETION_WAITING_DOTS="true"

# Uncomment the following line if you want to disable marking untracked files
# under VCS as dirty. This makes repository status check for large repositories
# much, much faster.
# DISABLE_UNTRACKED_FILES_DIRTY="true"

# History configuration
HIST_STAMPS="yyyy-mm-dd"
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=~/.zsh_history

# Which plugins would you like to load?
# Standard plugins can be found in $ZSH/plugins/
# Custom plugins may be added to $ZSH_CUSTOM/plugins/
plugins=(
    git
    docker
    docker-compose
    kubectl
    terraform
    aws
    npm
    node
    python
    pip
    virtualenv
    golang
    rust
    tmux
    fzf
    z
    colored-man-pages
    command-not-found
    sudo
    history
    extract
    aliases
    common-aliases
    jsontools
    web-search
    copypath
    copyfile
    copybuffer
    dirhistory
    zsh-autosuggestions
    zsh-syntax-highlighting
    fast-syntax-highlighting
    zsh-history-substring-search
    zsh-vi-mode
)

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

# User configuration

# Export paths
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"

# Preferred editor for local and remote sessions
if [[ -n $SSH_CONNECTION ]]; then
  export EDITOR='vim'
else
  export EDITOR='nvim'
fi

# Set language environment
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Compilation flags
export ARCHFLAGS="-arch x86_64"

# ============================================================================
# ALIASES
# ============================================================================

# Navigation aliases
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias .....="cd ../../../.."
alias ~="cd ~"
alias -- -="cd -"

# List directory contents
alias ls='ls --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias lt='ls -ltr'
alias lh='ls -lh'
alias lS='ls -lS'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gpl='git pull'
alias gco='git checkout'
alias gb='git branch'
alias glog='git log --oneline --decorate --graph'
alias gdiff='git diff'
alias gstash='git stash'

# Docker aliases
alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias drm='docker rm'
alias drmi='docker rmi'
alias dexec='docker exec -it'
alias dlogs='docker logs'
alias dprune='docker system prune -a'

# System aliases
alias update='sudo apt update && sudo apt upgrade'
alias install='sudo apt install'
alias search='apt search'
alias ports='netstat -tulanp'
alias meminfo='free -h'
alias cpuinfo='lscpu'
alias diskinfo='df -h'
alias processinfo='ps aux | grep'

# File operations
alias cp='cp -iv'
alias mv='mv -iv'
alias rm='rm -Iv'
alias mkdir='mkdir -pv'
alias chmod='chmod -v'
alias chown='chown -v'

# Utility aliases
alias h='history'
alias j='jobs -l'
alias which='type -a'
alias path='echo -e ${PATH//:/\\n}'
alias now='date +"%T"'
alias nowdate='date +"%d-%m-%Y"'
alias vi='vim'
alias svi='sudo vi'
alias edit='vim'
alias ping='ping -c 5'
alias fastping='ping -c 100 -s.2'
alias ports='netstat -tulanp'
alias wget='wget -c'
alias top='htop'
alias df='df -H'
alias du='du -ch'
alias free='free -m'

# Quick config editing
alias zshconfig="$EDITOR ~/.zshrc"
alias ohmyzsh="$EDITOR ~/.oh-my-zsh"
alias reload="source ~/.zshrc && echo 'ZSH config reloaded'"

# ============================================================================
# FUNCTIONS
# ============================================================================

# Create directory and cd into it
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# Extract archives
extract() {
    if [ -f $1 ] ; then
        case $1 in
            *.tar.bz2)   tar xjf $1     ;;
            *.tar.gz)    tar xzf $1     ;;
            *.bz2)       bunzip2 $1     ;;
            *.rar)       unrar e $1     ;;
            *.gz)        gunzip $1      ;;
            *.tar)       tar xf $1      ;;
            *.tbz2)      tar xjf $1     ;;
            *.tgz)       tar xzf $1     ;;
            *.zip)       unzip $1       ;;
            *.Z)         uncompress $1  ;;
            *.7z)        7z x $1        ;;
            *)     echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# Find file by name
ff() {
    find . -type f -name "*$1*"
}

# Find directory by name
fd() {
    find . -type d -name "*$1*"
}

# Git commit with message
gcm() {
    git commit -m "$1"
}

# Docker cleanup
docker-cleanup() {
    docker rm $(docker ps -a -q -f status=exited)
    docker rmi $(docker images -q -f dangling=true)
    docker volume rm $(docker volume ls -q -f dangling=true)
}

# Show most used commands
histtop() {
    history | awk '{print $2}' | sort | uniq -c | sort -rn | head -20
}

# ============================================================================
# PLUGIN CONFIGURATIONS
# ============================================================================

# ZSH Autosuggestions configuration
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE="fg=#666666"
ZSH_AUTOSUGGEST_STRATEGY=(history completion)
ZSH_AUTOSUGGEST_BUFFER_MAX_SIZE=20
bindkey '^[[Z' autosuggest-accept  # Shift+Tab to accept suggestion

# FZF configuration
export FZF_DEFAULT_COMMAND='find . -type f ! -path "*/\.git/*"'
export FZF_DEFAULT_OPTS='--height 40% --layout=reverse --border'
export FZF_CTRL_T_OPTS="--preview 'bat --style=numbers --color=always {}' --preview-window=right:60%"
export FZF_ALT_C_OPTS="--preview 'tree -C {} | head -200'"

# The fuck configuration
eval $(thefuck --alias)
alias f='fuck'

# Vi mode configuration
bindkey -v
export KEYTIMEOUT=1

# Change cursor shape for different vi modes.
function zle-keymap-select {
  if [[ ${KEYMAP} == vicmd ]] ||
     [[ $1 = 'block' ]]; then
    echo -ne '\e[1 q'
  elif [[ ${KEYMAP} == main ]] ||
       [[ ${KEYMAP} == viins ]] ||
       [[ ${KEYMAP} = '' ]] ||
       [[ $1 = 'beam' ]]; then
    echo -ne '\e[5 q'
  fi
}
zle -N zle-keymap-select

# Use vim keys in tab complete menu
bindkey -M menuselect 'h' vi-backward-char
bindkey -M menuselect 'k' vi-up-line-or-history
bindkey -M menuselect 'l' vi-forward-char
bindkey -M menuselect 'j' vi-down-line-or-history
bindkey -v '^?' backward-delete-char

# History substring search bindings
bindkey '^[[A' history-substring-search-up
bindkey '^[[B' history-substring-search-down
bindkey -M vicmd 'k' history-substring-search-up
bindkey -M vicmd 'j' history-substring-search-down

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Node.js
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# Python
export PYTHONDONTWRITEBYTECODE=1
export VIRTUAL_ENV_DISABLE_PROMPT=1

# Go
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Rust
export PATH="$HOME/.cargo/bin:$PATH"

# Ruby
export PATH="$HOME/.rbenv/bin:$PATH"
eval "$(rbenv init -)" 2>/dev/null

# Java
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$PATH:$JAVA_HOME/bin

# ============================================================================
# COMPLETION SETTINGS
# ============================================================================

# Basic auto/tab complete
autoload -U compinit && compinit
zstyle ':completion:*' menu select
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' rehash true
zstyle ':completion:*' accept-exact '*(N)'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path ~/.zsh/cache

# Kill completion
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'
zstyle ':completion:*:kill:*' command 'ps -u $USER -o pid,%cpu,tty,cputime,cmd'

# ============================================================================
# PROMPT CUSTOMIZATION (if not using Powerlevel10k)
# ============================================================================

# If you want to use a custom prompt instead of Powerlevel10k, uncomment:
# autoload -Uz vcs_info
# precmd() { vcs_info }
# zstyle ':vcs_info:git:*' formats '%b '
# setopt PROMPT_SUBST
# PROMPT='%F{green}%*%f %F{blue}%~%f %F{red}${vcs_info_msg_0_}%f$ '

# ============================================================================
# ADDITIONAL TOOLS
# ============================================================================

# Load zoxide (better cd)
eval "$(zoxide init zsh)" 2>/dev/null

# Load starship prompt (alternative to powerlevel10k)
# eval "$(starship init zsh)"

# Load direnv
eval "$(direnv hook zsh)" 2>/dev/null

# ============================================================================
# LOCAL CUSTOMIZATIONS
# ============================================================================

# Source local configuration if it exists
[ -f ~/.zshrc.local ] && source ~/.zshrc.local

# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh.
[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh
```



## Installation Instructions

To use this configuration, you'll need to install the required components:

### **1. Install Oh My Zsh**
```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### **2. Install Powerlevel10k Theme**
```bash
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
```

### **3. Install Essential Plugins**
```bash
# zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# fast-syntax-highlighting
git clone https://github.com/zdharma-continuum/fast-syntax-highlighting ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/fast-syntax-highlighting

# zsh-history-substring-search
git clone https://github.com/zsh-users/zsh-history-substring-search ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-history-substring-search

# zsh-vi-mode
git clone https://github.com/jeffreytse/zsh-vi-mode ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-vi-mode
```

### **4. Install Additional Tools**
```bash
# FZF (fuzzy finder)
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install

# The Fuck (command correction)
pip install thefuck

# Zoxide (better cd)
curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

# Install fonts for Powerlevel10k
# Download and install a Nerd Font like "MesloLGS NF" from:
# https://github.com/romkatv/powerlevel10k#meslo-nerd-font-patched-for-powerlevel10k
```

## Key Features Included

This configuration provides:

- **Powerlevel10k theme** with instant prompt for blazing-fast startup
- **Auto-suggestions** that predict commands based on history
- **Syntax highlighting** for better command visibility
- **Vi-mode** for vim-like navigation in the terminal
- **FZF integration** for fuzzy finding files and history
- **Git integration** with extensive aliases and status information
- **Docker and Kubernetes** shortcuts for container management
- **History substring search** for finding previous commands
- **Smart tab completion** with case-insensitive matching
- **The Fuck** integration for correcting mistyped commands
- **Extensive aliases** for common operations
- **Custom functions** for productivity enhancement

After saving the `.zshrc` file, run `source ~/.zshrc` to apply the changes, then run `p10k configure` to set up your Powerlevel10k prompt style.



Spaceship is a minimalistic, powerful, and extremely customizable Zsh prompt that provides useful information about your current environment. Before setting it up, ensure you have Zsh version 5.2 or newer installed by running `echo $ZSH_VERSION`.

## Installation Methods

There are several ways to install Spaceship prompt, depending on your setup and preference. Here are the most common approaches:

### **Oh My Zsh Installation**

The most popular method for Oh My Zsh users involves cloning the repository and creating a symlink:

```bash
# Clone the Spaceship repository
git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$ZSH_CUSTOM/themes/spaceship-prompt" --depth=1

# Create a symbolic link
ln -s "$ZSH_CUSTOM/themes/spaceship-prompt/spaceship.zsh-theme" "$ZSH_CUSTOM/themes/spaceship.zsh-theme"

# Set the theme in your .zshrc
ZSH_THEME="spaceship"
```

### **NPM Installation**

For those who prefer using npm as a package manager:

```bash
npm install -g spaceship-prompt
```

This command will download Spaceship and prompt you to source it in your `~/.zshrc` file.

### **Homebrew Installation**

If you're on macOS and use Homebrew:

```bash
# Install via Homebrew
brew install spaceship

# Add initialization to .zshrc
echo "source $(brew --prefix)/opt/spaceship/spaceship.zsh" >> ~/.zshrc
```

### **Manual Installation**

For a completely manual setup:

```bash
# Create directory and clone
mkdir -p "$HOME/.zsh"
git clone --depth=1 https://github.com/spaceship-prompt/spaceship-prompt.git "$HOME/.zsh/spaceship"

# Add to .zshrc
echo 'source "$HOME/.zsh/spaceship/spaceship.zsh"' >> ~/.zshrc
```

## Plugin Manager Support

Spaceship works seamlessly with various Zsh plugin managers:

### **Zinit**
```bash
zinit light spaceship-prompt/spaceship-prompt
```

### **Antigen**
```bash
antigen theme spaceship-prompt/spaceship-prompt
```

### **Antibody**
```bash
antibody bundle spaceship-prompt/spaceship-prompt
```

### **Zplug**
```bash
zplug "spaceship-prompt/spaceship-prompt", use:spaceship.zsh, from:github, as:theme
```

## Configuration

After installation, you can customize Spaceship's behavior by adding configuration to your `.zshrc` file or creating a separate `.spaceshiprc.zsh` file.

### **Basic Configuration**

Add these settings to your `.zshrc` for a customized setup:

```bash
# Theme setting
ZSH_THEME="spaceship"

# Basic Spaceship configuration
SPACESHIP_PROMPT_ASYNC=true
SPACESHIP_PROMPT_ADD_NEWLINE=true
SPACESHIP_CHAR_SYMBOL="⚡"
SPACESHIP_USER_SHOW=always
SPACESHIP_DIR_TRUNC_REPO=false
```

### **Advanced Configuration with Custom Order**

Create a more detailed configuration by specifying the prompt order and individual section settings:

```bash
# Create a .spaceshiprc.zsh file
cat > ~/.spaceshiprc.zsh << 'EOF'
SPACESHIP_USER_SHOW=always
SPACESHIP_PROMPT_ADD_NEWLINE=false
SPACESHIP_CHAR_SYMBOL="λ"
SPACESHIP_CHAR_SUFFIX=" "

SPACESHIP_PROMPT_ORDER=(
    user          # Username section
    dir           # Current directory section
    host          # Hostname section
    git           # Git section (git_branch + git_status)
    package       # Package version
    node          # Node.js section
    python        # Python section
    rust          # Rust section
    docker        # Docker section
    terraform     # Terraform section
    exec_time     # Execution time
    line_sep      # Line break
    jobs          # Background jobs indicator
    exit_code     # Exit code section
    char          # Prompt character
)
EOF

# Source the configuration in .zshrc
echo "source ~/.spaceshiprc.zsh" >> ~/.zshrc
```

### **Performance Optimization**

For improved performance, you can use a minimal configuration:

```bash
# Minimal spaceship sections for performance
SPACESHIP_PROMPT_ORDER=(
    time
    user
    dir
    git
    line_sep
    char
)
```

## Font Requirements

Spaceship works best with Powerline fonts or Nerd Fonts for proper symbol rendering. Install a compatible font:

```bash
# On macOS with Homebrew
brew tap homebrew/cask-fonts
brew install font-hack-nerd-font

# Configure your terminal to use the installed font
```

## Complete .zshrc Example

Here's a complete example section for your `.zshrc` file:

```bash
# Oh My Zsh configuration
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="spaceship"

# Spaceship prompt configuration
SPACESHIP_PROMPT_ASYNC=true
SPACESHIP_PROMPT_ADD_NEWLINE=true
SPACESHIP_CHAR_SYMBOL="➜"
SPACESHIP_CHAR_SUFFIX=" "
SPACESHIP_TIME_SHOW=false
SPACESHIP_USER_SHOW=always
SPACESHIP_HOST_SHOW=full
SPACESHIP_DIR_TRUNC=3
SPACESHIP_DIR_TRUNC_REPO=false
SPACESHIP_GIT_BRANCH_SHOW=true
SPACESHIP_GIT_STATUS_SHOW=true
SPACESHIP_EXEC_TIME_SHOW=true
SPACESHIP_EXIT_CODE_SHOW=true

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

# Source additional Spaceship configuration if exists
[[ -f ~/.spaceshiprc.zsh ]] && source ~/.spaceshiprc.zsh
```

After making these changes, reload your terminal or run `source ~/.zshrc` to apply the new configuration. The prompt will now display contextual information about your current directory, Git repository status, execution time, and other relevant details based on your environment.

# Spaceship Prompt Theme Setup for Zsh

Spaceship is a minimalistic, powerful, and highly customizable Zsh prompt that provides a clean terminal experience with useful information displayed intelligently. Here's how to set it up with your `.zshrc` configuration.

## Prerequisites

Before installing Spaceship, ensure you have:

- **Zsh** (v5.2 or later) installed. Check your version with: `echo $ZSH_VERSION`
- **Powerline Font** or **Nerd Font** installed and configured in your terminal for proper icon display

## Installation Methods

### **Method 1: With Oh My Zsh** (Recommended)

If you're using Oh My Zsh, follow these steps:

1. **Clone the Spaceship repository**:
```bash
git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$ZSH_CUSTOM/themes/spaceship-prompt" --depth=1
```

2. **Create a symbolic link**:
```bash
ln -s "$ZSH_CUSTOM/themes/spaceship-prompt/spaceship.zsh-theme" "$ZSH_CUSTOM/themes/spaceship.zsh-theme"
```

3. **Set the theme in your `.zshrc`**:
```bash
ZSH_THEME="spaceship"
```

### **Method 2: Using Homebrew** (macOS/Linux)

For a simpler installation via Homebrew:

```bash
brew install spaceship
```

Then add to your `.zshrc`:
```bash
source "$(brew --prefix)/opt/spaceship/spaceship.zsh"
```

### **Method 3: Using npm**

Install globally via npm:
```bash
npm install -g spaceship-prompt
```

## Configuration

### Basic Configuration

After installation, you can customize Spaceship by adding configuration to your `.zshrc` file. Here's a popular configuration example:

```bash
# Spaceship prompt configuration
SPACESHIP_PROMPT_ADD_NEWLINE=false
SPACESHIP_CHAR_SYMBOL="❯"
SPACESHIP_CHAR_SUFFIX=" "
SPACESHIP_USER_SHOW=always

SPACESHIP_PROMPT_ORDER=(
  user          # Username section
  dir           # Current directory section
  host          # Hostname section
  git           # Git section (git_branch + git_status)
  package       # Package version
  node          # Node.js section
  exec_time     # Execution time
  line_sep      # Line break
  jobs          # Background jobs indicator
  exit_code     # Exit code section
  char          # Prompt character
)
```

### Advanced Configuration with Separate File

For better organization, create a dedicated configuration file:

1. **Create `.spaceshiprc.zsh`**:
```bash
nano ~/.spaceshiprc.zsh
```

2. **Add your configuration** (example with more options):
```bash
SPACESHIP_USER_SHOW=always
SPACESHIP_PROMPT_ADD_NEWLINE=false
SPACESHIP_CHAR_SYMBOL="λ"
SPACESHIP_CHAR_SUFFIX=" "
SPACESHIP_DIR_TRUNC_REPO=false

SPACESHIP_PROMPT_ORDER=(
  user
  dir
  host
  git
  package
  node
  bun
  elixir
  rust
  docker
  terraform
  exec_time
  line_sep
  jobs
  exit_code
  char
)
```

3. **Source the configuration in `.zshrc`**:
```bash
echo "source ~/.spaceshiprc.zsh" >> ~/.zshrc
source ~/.zshrc
```

### Performance Optimization

For faster prompt rendering, enable asynchronous mode:

```bash
SPACESHIP_PROMPT_ASYNC=true
```

## Plugin Manager Installation

### **Antigen**
```bash
antigen theme spaceship-prompt/spaceship-prompt
```

### **Zinit**
```bash
zinit light spaceship-prompt/spaceship-prompt
```

### **Zplug**
```bash
zplug "spaceship-prompt/spaceship-prompt", use:spaceship.zsh, from:github, as:theme
```

## Verification

After installation and configuration:

1. **Reload your shell**:
```bash
source ~/.zshrc
```

2. **Verify the installation** by checking if the prompt displays correctly with git information, directory path, and other configured sections

## Common Customization Options

| Option | Description | Example Value |
|--------|-------------|---------------|
| `SPACESHIP_CHAR_SYMBOL` | Prompt character | `"⚡"`, `"❯"`, `"λ"` |
| `SPACESHIP_PROMPT_ADD_NEWLINE` | Add newline before prompt | `true` / `false` |
| `SPACESHIP_USER_SHOW` | When to show username | `always`, `needed`, `false` |
| `SPACESHIP_DIR_TRUNC_REPO` | Truncate directory in git repos | `true` / `false` |

The theme will now provide you with a clean, informative command prompt showing relevant information like your current git branch, uncommitted changes, execution time, and more.
# Generic AI CLI tool wrapper
ai_complete() {
    # Assuming you have an 'ai' CLI tool installed
    local completion=$(ai complete "$BUFFER" 2>/dev/null)
    if [[ -n "$completion" ]]; then
        BUFFER="$completion"
        zle end-of-line
    fi
}
zle -N ai_complete
bindkey '\ei' ai_complete  # Alt+I for AI completion
```

## Creating a Custom AI Widget

For a more sophisticated setup, create a dedicated widget file:

```bash
# Create a file: ~/.config/zsh/ai-widget.zsh

# AI suggestion widget with visual feedback
ai-suggestion-widget() {
    local current_command="$BUFFER"
    
    # Show loading indicator
    zle -M "Getting AI suggestion..."
    
    # Get AI suggestion (implement your preferred AI backend)
    local suggestion=$(get_ai_suggestion "$current_command")
    
    if [[ -n "$suggestion" ]]; then
        # Display suggestion below current line
        zle -M "AI Suggestion: $suggestion"
        
        # Optionally replace buffer
        read -k 1 "?Press 'y' to accept, any other key to cancel: "
        if [[ $REPLY == 'y' ]]; then
            BUFFER="$suggestion"
            zle end-of-line
        fi
    else
        zle -M "No AI suggestion available"
    fi
}

# Helper function for AI backend
get_ai_suggestion() {
    local query="$1"
    # Implement your AI backend call here
    # This could be OpenAI, Claude, local LLM, etc.
    echo "suggested command based on: $query"
}

# Register and bind the widget
zle -N ai-suggestion-widget
bindkey '^[a' ai-suggestion-widget  # Alt+A
```

### **Loading the Widget**

Add this to your `.zshrc`:

```bash
# Source AI widget
[[ -f ~/.config/zsh/ai-widget.zsh ]] && source ~/.config/zsh/ai-widget.zsh

# Enable widget on startup
autoload -Uz ai-suggestion-widget
```

## Configuration Tips

The search results mention several configuration options for autosuggestions that you can adapt:

- **Visual styling**: Customize how suggestions appear using `ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE`
- **Buffer size limits**: Set `ZSH_AUTOSUGGEST_BUFFER_MAX_SIZE` to disable suggestions for large inputs
- **Custom key bindings**: Define specific keys for accepting, executing, or clearing suggestions

Remember to source your `.zshrc` file after making changes:

```bash
source ~/.zshrc
```

This setup provides both traditional history-based suggestions and AI-powered command assistance, giving you the best of both worlds for command-line productivity.
# Setting Up AIChat Suggestions Widget for Zsh

## Installing AIChat

First, you'll need to install AIChat itself. You can install it through various package managers or download it directly from the GitHub repository.

## Enabling Shell Integration for Zsh

AIChat provides **shell integration scripts** that enable intelligent command suggestions directly in your terminal. This feature allows you to type natural language descriptions and have them converted to shell commands.

### Step 1: Download the Integration Script

AIChat offers shell integration scripts for multiple shells including zsh. These scripts are available in the GitHub repository at `https://github.com/sigoden/aichat/tree/main/scripts/shell-integration`.

The zsh integration file is named `integration.zsh` and needs to be sourced in your `.zshrc` file.

### Step 2: Configure Your .zshrc

Add the following to your `.zshrc` file to enable the AIChat suggestions widget:

1. **Download the integration script** to a location on your system (e.g., `~/.config/aichat/`)
2. **Source the script** in your `.zshrc`:
   ```bash
   source ~/.config/aichat/integration.zsh
   ```
3. **Reload your shell configuration**:
   ```bash
   source ~/.zshrc
   ```

## How the Widget Works

Once configured, the AIChat suggestions widget provides several powerful features:

- **Natural Language Input**: Type what you want to do in plain English
- **Command Conversion**: Press **Alt+E** to convert your natural language request into an executable shell command
- **OS-Aware**: AIChat recognizes your operating system and shell, providing appropriate commands for your specific environment



## Additional Features

The integration also supports:

- **Command History**: Suggestions based on your command history
- **Context Awareness**: Understanding of your current shell and OS environment
- **Quick Execution**: After conversion, you can review and execute the generated command immediately

## Complementary Tools

While AIChat provides excellent command suggestions, you might also consider **zsh-autosuggestions** for traditional history-based completions. This plugin offers:

- Fish-like autosuggestions displayed in gray text
- Acceptance with arrow keys or End key
- Custom key bindings for various suggestion actions

To use both tools together, you can install zsh-autosuggestions alongside AIChat for a comprehensive command-line experience that combines AI-powered suggestions with traditional history-based completions.

## Verification

After setup, test the integration by:

1. Opening a new terminal window
2. Typing a natural language command like "list all files in current directory"
3. Pressing **Alt+E** to see it converted to the appropriate shell command (e.g., `ls -la`)

This setup will significantly enhance your command-line productivity by allowing you to describe what you want to accomplish without memorizing exact command syntax.
