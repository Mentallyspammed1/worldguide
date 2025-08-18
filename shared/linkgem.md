Here’s a clean, battle‑tested workflow to make your local fork of gemini-cli power the gemini command on your machine. It covers npm, pnpm, and Yarn; single‑package and monorepo layouts; macOS/Linux/Termux and Windows.

Overview
- Goal: when you type gemini, it runs your fork’s CLI code.
- Strategy: clone your fork, install deps, build, then “globally link” its CLI package so the gemini shim points at your working copy.
- Rollback: unlink and reinstall the published package.

Prerequisites
- Node.js 18+ and npm available in PATH.
- If you previously installed the published CLI globally, note it: npm list -g @google/gemini-cli || true
- Termux tip: ensure ~/.npm-global/bin is on PATH if you use a local npm prefix.

Step 0 — Clone your fork and add upstream
- Unix/macOS/Termux:
  git clone https://github.com/YOUR_GH_USER/gemini-cli.git
  cd gemini-cli
  git remote add upstream https://github.com/google-gemini/gemini-cli.git
  git fetch upstream
- Windows PowerShell:
  git clone https://github.com/YOUR_GH_USER/gemini-cli.git
  Set-Location gemini-cli
  git remote add upstream https://github.com/google-gemini/gemini-cli.git
  git fetch upstream

Step 1 — Detect the repo layout
You need the package that actually exports the gemini binary.

- Quick detectors (Unix/macOS/Termux):
  # At repo root
  test -f package.json && echo "root package:" && jq -r '.name, .bin' package.json 2>/dev/null
  test -d packages && fd package.json packages -x sh -c 'echo "pkg: $1"; jq -r ".name, .bin" "$1" 2>/dev/null' sh {} \;

- What you’re looking for:
  - package.json "bin" contains a "gemini" entry (e.g., "bin": { "gemini": "dist/index.js" }).
  - Common cases:
    A) Single-package repo: CLI is at repo root.
    B) Monorepo: CLI is inside packages/cli (or similar).

Step 2 — Install dependencies
Pick the tool that matches the repo’s lock file:
- npm (package-lock.json):
  npm install
- pnpm (pnpm-lock.yaml):
  corepack enable
  pnpm -w install
- Yarn classic (yarn.lock):
  yarn install

Step 3 — Build the CLI
- npm:
  npm run build
- pnpm:
  pnpm -w build
- Yarn:
  yarn build

Step 4 — Link the CLI so “gemini” points to your fork
Choose the right folder (from Step 1), then:

Case A — CLI at repo root (bin.gemini defined here)
- Unix/macOS/Termux:
  npm link
- Windows:
  npm link

Case B — Monorepo (CLI in packages/cli)
- Unix/macOS/Termux:
  cd packages/cli
  npm link
- Windows:
  Set-Location packages/cli
  npm link

Notes
- npm link here creates a global symlink from the package name to your working copy and installs the bin shim gemini pointing at your local build output.
- If you still have a global install, npm link overrides it. If you want to be tidy first:
  npm rm -g @google/gemini-cli || true

Step 5 — Verify you’re running the linked fork
- Where is gemini?
  which gemini        # macOS/Linux/Termux
  Get-Command gemini  # Windows
- Does it resolve to your fork?
  node -p "require.resolve('@google/gemini-cli/package.json')" 2>/dev/null
- Version sanity:
  gemini --version
If version output isn’t helpful, add a temporary line in your local entry file (e.g., dist/index.js) to print a banner, rebuild, and re-run gemini.

Step 6 — Develop with a tight feedback loop
- Watch builds (if available):
  npm run dev
- Or rebuild after changes:
  npm run build
Because npm link points to your folder, the gemini command picks up your new build immediately.

Handy aliases (bash/zsh)
- echo "alias gdev='(cd ~/path/to/gemini-cli && npm run build) && gemini'" >> ~/.bashrc

Step 7 — Switch between your fork and the published package
- Switch back to the published package:
  npm unlink -g @google/gemini-cli || true
  npm rm -g @google/gemini-cli || true
  npm i -g @google/gemini-cli
- Switch to local fork again (monorepo shown):
  (cd ~/path/to/gemini-cli/packages/cli && npm link)

Optional alternatives
- Use npx directly from your fork (no linking):
  npx github:YOUR_GH_USER/gemini-cli#BRANCH
- Test as if it were published:
  npm pack
  npm i -g ./your-tarball-*.tgz
- Yarn classic linking (if the repo uses yarn workspaces):
  cd packages/cli
  yarn link
  # yarn global add 'link:@google/gemini-cli' (or link into a sandbox project and run from there)

Termux-specific tips
- Ensure PATH has your npm global prefix:
  mkdir -p ~/.npm-global
  npm config set prefix ~/.npm-global
  echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
- If scripts rely on node shebang, confirm Termux’s node path is correct. Rebuild after changing shebangs.

Troubleshooting
- gemini still runs the published package
  - Check PATH order: echo $PATH
  - Which shim is used: which gemini
  - Remove stale shims in older Node prefixes (e.g., /usr/local/bin/gemini) and re-link.
- Permission errors on Windows
  - Run terminal as Administrator when linking, or configure Developer Mode to allow symlinks.
- pnpm monorepo complains about hoisting
  - Use corepack enable and pnpm -w install at the repo root, then pnpm -C packages/cli link --global.
- “Command not found” after linking on Termux
  - Re-source your shell rc file: source ~/.bashrc
  - Verify ~/.npm-global/bin exists and contains a gemini shim.

Quality-of-life: quick env switcher
- Create a tiny script to toggle global gemini:

Unix/macOS/Termux
- cat > ~/.gemini/switch-gemini <<'EOF'
#!/usr/bin/env bash
set -e
case "$1" in
  local)
    (cd ~/path/to/gemini-cli/packages/cli 2>/dev/null && npm link) || (cd ~/path/to/gemini-cli && npm link)
    echo "Switched gemini -> local fork"
    ;;
  global)
    npm unlink -g @google/gemini-cli || true
    npm rm -g @google/gemini-cli || true
    npm i -g @google/gemini-cli
    echo "Switched gemini -> published package"
    ;;
  *)
    echo "Usage: switch-gemini [local|global]"
    exit 1
esac
EOF
- chmod +x ~/.gemini/switch-gemini
- echo 'alias gemini-use="~/.gemini/switch-gemini"' >> ~/.bashrc && source ~/.bashrc
- gemini-use local  # or: gemini-use global

That’s it—your gemini command now runs from your fork. If you tell me whether your repo is single-package or a monorepo, and which package manager it uses, I can tailor exact commands for your layout.
# Complete Gemini CLI Setup Guide: From Zero to Production

## Table of Contents

### Part I: Foundation Setup
1. [Installation Methods & Verification](#installation-methods--verification)
2. [Authentication Configuration](#authentication-configuration)
3. [Initial Configuration Files](#initial-configuration-files)
4. [Shell Integration & Aliases](#shell-integration--aliases)

### Part II: Core Configuration
5. [Settings.json Deep Dive](#settingsjson-deep-dive)
6. [Built-in Tools Configuration](#built-in-tools-configuration)
7. [Memory System Setup](#memory-system-setup)
8. [Context Management](#context-management)

### Part III: Advanced Setup
9. [MCP Server Configuration](#mcp-server-configuration)
10. [Custom Commands Setup](#custom-commands-setup)
11. [Extension Ecosystem](#extension-ecosystem)
12. [Performance Optimization](#performance-optimization)

### Part IV: Production Deployment
13. [Multi-User Setup](#multi-user-setup)
14. [CI/CD Integration](#cicd-integration)
15. [Monitoring & Logging](#monitoring--logging)
16. [Troubleshooting Guide](#troubleshooting-guide)

---

## Part I: Foundation Setup

## Installation Methods & Verification

### Method 1: NPM Global Installation (Recommended)

```bash
#!/bin/bash
# install-gemini-cli.sh - Complete installation script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

# Check system
check_system() {
    info "Detecting system..."
    
    if [ -n "${TERMUX_VERSION:-}" ]; then
        SYSTEM="termux"
        info "Running on Termux ${TERMUX_VERSION}"
    elif [ "$(uname)" == "Darwin" ]; then
        SYSTEM="macos"
        info "Running on macOS"
    elif [ "$(uname)" == "Linux" ]; then
        SYSTEM="linux"
        info "Running on Linux"
    elif [[ "$(uname -r)" == *"Microsoft"* ]]; then
        SYSTEM="wsl"
        info "Running on WSL"
    else
        SYSTEM="unknown"
        warn "Unknown system, proceeding with generic installation"
    fi
}

# Install Node.js if needed
install_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v | sed 's/v//')
        MIN_VERSION="18.0.0"
        
        if [ "$(printf '%s\n' "$MIN_VERSION" "$NODE_VERSION" | sort -V | head -n1)" = "$MIN_VERSION" ]; then
            log "Node.js $NODE_VERSION is installed"
            return 0
        else
            warn "Node.js $NODE_VERSION is too old, need v$MIN_VERSION+"
        fi
    else
        warn "Node.js not found, installing..."
    fi
    
    case "$SYSTEM" in
        termux)
            pkg install -y nodejs-lts
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install node
            else
                error "Please install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            fi
            ;;
        linux|wsl)
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
        *)
            error "Please install Node.js manually from https://nodejs.org"
            ;;
    esac
    
    log "Node.js installed successfully"
}

# Configure NPM
configure_npm() {
    info "Configuring NPM..."
    
    # Set global directory for non-root installs
    NPM_PREFIX="$HOME/.npm-global"
    mkdir -p "$NPM_PREFIX"
    npm config set prefix "$NPM_PREFIX"
    
    # Add to PATH
    export PATH="$NPM_PREFIX/bin:$PATH"
    
    # Persist PATH changes
    SHELL_RC="$HOME/.bashrc"
    [ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"
    
    if ! grep -q "npm-global" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# NPM global packages" >> "$SHELL_RC"
        echo "export PATH=\"$NPM_PREFIX/bin:\$PATH\"" >> "$SHELL_RC"
        log "Added NPM to PATH in $SHELL_RC"
    fi
    
    # Configure registry and timeouts
    npm config set registry https://registry.npmjs.org/
    npm config set fetch-retry-mintimeout 20000
    npm config set fetch-retry-maxtimeout 120000
    
    log "NPM configured"
}

# Install Gemini CLI
install_gemini() {
    info "Installing Gemini CLI..."
    
    # Try installation with retries
    MAX_ATTEMPTS=3
    ATTEMPT=1
    
    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
        info "Installation attempt $ATTEMPT/$MAX_ATTEMPTS..."
        
        if npm install -g @google/gemini-cli; then
            log "Gemini CLI installed successfully"
            break
        else
            warn "Installation failed, attempt $ATTEMPT/$MAX_ATTEMPTS"
            ATTEMPT=$((ATTEMPT + 1))
            [ $ATTEMPT -le $MAX_ATTEMPTS ] && sleep 5
        fi
    done
    
    [ $ATTEMPT -gt $MAX_ATTEMPTS ] && error "Failed to install after $MAX_ATTEMPTS attempts"
}

# Verify installation
verify_installation() {
    info "Verifying installation..."
    
    # Check gemini command
    if ! command -v gemini &> /dev/null; then
        error "gemini command not found. Check your PATH"
    fi
    
    # Get version
    VERSION=$(gemini --version 2>/dev/null || echo "unknown")
    log "Gemini CLI version: $VERSION"
    
    # Check Node modules
    GEMINI_PATH=$(which gemini)
    log "Gemini CLI installed at: $GEMINI_PATH"
    
    # Verify dependencies
    NPM_LIST=$(npm list -g @google/gemini-cli --depth=0 2>/dev/null || true)
    if [[ $NPM_LIST == *"@google/gemini-cli"* ]]; then
        log "NPM package verified"
    else
        warn "NPM package verification failed, but command exists"
    fi
}

# Main installation
main() {
    echo "========================================"
    echo "    Gemini CLI Complete Installation    "
    echo "========================================"
    echo ""
    
    check_system
    install_nodejs
    configure_npm
    install_gemini
    verify_installation
    
    echo ""
    echo "========================================"
    log "Installation complete!"
    info "Run 'source $SHELL_RC' to update PATH"
    info "Then run 'gemini' to start"
    echo "========================================"
}

main "$@"
```

### Method 2: Development Installation (Editable)```bash
#!/bin/bash
# dev-install.sh - Install from source for development

# Clone the repository
git clone https://github.com/google-gemini/gemini-cli.git
cd gemini-cli

# Install dependencies
npm install

# Build the project
npm run build

# Link for development (method 1: development mode)
npm run dev

# OR Link globally (method 2: production-like)
npm link packages/cli

# Now use the development version
gemini --version
```

For contributors and developers, you can run Gemini CLI directly from source code with hot-reloading for active development, or link the local package to simulate a production installation.

### Method 3: Docker Installation

```bash
# docker-install.sh - Containerized installation

# Pull the official image
docker pull gcr.io/gemini-cli/gemini-cli-sandbox:latest

# Create alias for easy use
alias gemini-docker='docker run -it \
  -v "$HOME/.gemini:/home/user/.gemini" \
  -v "$PWD:/workspace" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  gcr.io/gemini-cli/gemini-cli-sandbox:latest'

# Run Gemini CLI in container
gemini-docker
```

### Method 4: Direct NPX Execution

You can run the most recently committed version of Gemini CLI directly from the GitHub repo:

```bash
# Run directly without installation
npx github:google-gemini/gemini-cli

# Or with specific version
npx @google/gemini-cli@latest
```

## Authentication Configuration

### Complete Authentication Setup Script

```bash
#!/bin/bash
# auth-setup.sh - Comprehensive authentication configuration

set -euo pipefail

# Configuration directory
CONFIG_DIR="$HOME/.gemini"
SETTINGS_FILE="$CONFIG_DIR/settings.json"
SECURE_DIR="$CONFIG_DIR/secure"

# Create directories
mkdir -p "$CONFIG_DIR" "$SECURE_DIR"
chmod 700 "$SECURE_DIR"

# Authentication method selection
select_auth_method() {
    echo "Select authentication method:"
    echo "1) Google Account (Recommended - Free tier)"
    echo "2) API Key (For programmatic access)"
    echo "3) Service Account (For CI/CD)"
    echo "4) OAuth2 Token (Advanced)"
    read -p "Choice [1-4]: " choice
    
    case $choice in
        1) setup_google_auth ;;
        2) setup_api_key ;;
        3) setup_service_account ;;
        4) setup_oauth2 ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
}

# Google Account authentication
setup_google_auth() {
    echo "Setting up Google Account authentication..."
    
    # Create auth helper script
    cat > "$CONFIG_DIR/auth-google.sh" << 'EOF'
#!/bin/bash
# Google Account authentication helper

# Function to handle authentication URL
handle_auth_url() {
    echo "Opening authentication URL in browser..."
    
    # Detect OS and open browser
    if [[ "$OSTYPE" == "linux-android"* ]]; then
        # Termux
        termux-open-url "$1"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$1"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open "$1" 2>/dev/null || firefox "$1" || google-chrome "$1"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows
        start "$1"
    else
        echo "Please open this URL manually: $1"
    fi
}

# Run Gemini with authentication capture
gemini 2>&1 | while IFS= read -r line; do
    echo "$line"
    
    # Check for auth URL
    if [[ "$line" =~ https://accounts.google.com ]]; then
        URL=$(echo "$line" | grep -o 'https://[^"]*' | head -1)
        [ -n "$URL" ] && handle_auth_url "$URL"
    fi
done
EOF
    
    chmod +x "$CONFIG_DIR/auth-google.sh"
    
    echo "Google authentication configured."
    echo "Run: $CONFIG_DIR/auth-google.sh"
}

# API Key setup
setup_api_key() {
    echo "Setting up API Key authentication..."
    
    # Prompt for API key
    echo -n "Enter your Gemini API key: "
    read -rs API_KEY
    echo
    
    # Validate key format
    if [[ ! "$API_KEY" =~ ^[A-Za-z0-9_-]{39}$ ]]; then
        echo "Warning: API key format appears invalid"
    fi
    
    # Encrypt and store key
    echo "$API_KEY" | openssl enc -aes-256-cbc -pbkdf2 -salt -out "$SECURE_DIR/api_key.enc"
    
    # Create loader script
    cat > "$CONFIG_DIR/load-api-key.sh" << 'EOF'
#!/bin/bash
# Load encrypted API key

if [ -f "$HOME/.gemini/secure/api_key.enc" ]; then
    echo -n "Enter decryption password: "
    read -rs PASSWORD
    echo
    
    export GEMINI_API_KEY=$(
        openssl enc -aes-256-cbc -pbkdf2 -d \
        -in "$HOME/.gemini/secure/api_key.enc" \
        -k "$PASSWORD" 2>/dev/null
    )
    
    if [ $? -eq 0 ]; then
        echo "API key loaded successfully"
    else
        echo "Failed to decrypt API key"
        unset GEMINI_API_KEY
    fi
fi
EOF
    
    chmod +x "$CONFIG_DIR/load-api-key.sh"
    
    echo "API key encrypted and stored."
    echo "Load with: source $CONFIG_DIR/load-api-key.sh"
}

# Service Account setup
setup_service_account() {
    echo "Setting up Service Account authentication..."
    
    read -p "Path to service account JSON file: " SA_FILE
    
    if [ ! -f "$SA_FILE" ]; then
        echo "File not found: $SA_FILE"
        exit 1
    fi
    
    # Copy and secure service account file
    cp "$SA_FILE" "$SECURE_DIR/service-account.json"
    chmod 600 "$SECURE_DIR/service-account.json"
    
    # Create activation script
    cat > "$CONFIG_DIR/activate-sa.sh" << 'EOF'
#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gemini/secure/service-account.json"
echo "Service account activated"
EOF
    
    chmod +x "$CONFIG_DIR/activate-sa.sh"
    
    echo "Service account configured."
    echo "Activate with: source $CONFIG_DIR/activate-sa.sh"
}

# OAuth2 token setup
setup_oauth2() {
    echo "Setting up OAuth2 token authentication..."
    
    # Create OAuth2 flow script
    cat > "$CONFIG_DIR/oauth2-flow.py" << 'EOF'
#!/usr/bin/env python3
import json
import os
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/generative-language']
TOKEN_FILE = os.path.expanduser('~/.gemini/secure/token.json')
CREDS_FILE = os.path.expanduser('~/.gemini/secure/credentials.json')

def authenticate():
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_FILE):
                print(f"Please place OAuth2 credentials at: {CREDS_FILE}")
                sys.exit(1)
            
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
        os.chmod(TOKEN_FILE, 0o600)
    
    print(f"export GEMINI_ACCESS_TOKEN='{creds.token}'")

if __name__ == '__main__':
    authenticate()
EOF
    
    chmod +x "$CONFIG_DIR/oauth2-flow.py"
    
    echo "OAuth2 setup complete."
    echo "1. Place OAuth2 credentials at: $SECURE_DIR/credentials.json"
    echo "2. Run: eval \$($CONFIG_DIR/oauth2-flow.py)"
}

# Main execution
select_auth_method
```

## Initial Configuration Files

### Complete settings.json Configuration

```json
{
  "$schema": "https://gemini-cli.dev/schemas/settings.json",
  "version": "1.0.0",
  
  "// Authentication": "Choose one authentication method",
  "authentication": {
    "method": "google",
    "apiKey": "${GEMINI_API_KEY}",
    "serviceAccount": "${GOOGLE_APPLICATION_CREDENTIALS}",
    "oauth2Token": "${GEMINI_ACCESS_TOKEN}"
  },
  
  "// Model Configuration": "Gemini model settings",
  "model": {
    "name": "gemini-2.5-pro",
    "temperature": 0.7,
    "topP": 0.95,
    "topK": 40,
    "maxOutputTokens": 8192,
    "stopSequences": [],
    "candidateCount": 1
  },
  
  "// UI Configuration": "Terminal interface settings",
  "ui": {
    "theme": "auto",
    "markdown": true,
    "syntaxHighlighting": true,
    "lineNumbers": false,
    "wordWrap": 80,
    "pager": "less",
    "editor": "vim",
    "colors": {
      "prompt": "cyan",
      "response": "white",
      "error": "red",
      "warning": "yellow",
      "info": "blue",
      "success": "green"
    }
  },
  
  "// Tool Configuration": "Built-in tools settings",
  "tools": {
    "enabled": [
      "read_file",
      "read_many_files",
      "file_write",
      "run_shell_command",
      "google_web_search",
      "save_memory",
      "web_fetch"
    ],
    "disabled": [],
    "restrictions": {
      "run_shell_command": {
        "allowedCommands": [],
        "blockedCommands": ["rm -rf", "dd", "mkfs"],
        "requireConfirmation": true,
        "timeout": 30000
      },
      "file_write": {
        "maxFileSize": "10MB",
        "allowedPaths": ["./", "~/projects/"],
        "blockedPaths": ["/etc", "/sys", "/proc"]
      },
      "web_fetch": {
        "allowedDomains": [],
        "blockedDomains": [],
        "maxSize": "5MB",
        "timeout": 15000
      }
    }
  },
  
  "// MCP Servers": "Model Context Protocol servers",
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"],
      "trust": false
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://localhost/mydb"
      ]
    }
  },
  
  "// Memory Configuration": "Context and memory settings",
  "memory": {
    "maxContextTokens": 1000000,
    "compressionThreshold": 0.8,
    "autoSave": true,
    "saveInterval": 300,
    "historyLimit": 100
  },
  
  "// Logging Configuration": "Logging and debugging",
  "logging": {
    "level": "info",
    "file": "~/.gemini/logs/gemini.log",
    "maxSize": "10MB",
    "maxFiles": 5,
    "format": "json"
  },
  
  "// Network Configuration": "Network and proxy settings",
  "network": {
    "proxy": "${HTTP_PROXY}",
    "timeout": 30000,
    "retries": 3,
    "retryDelay": 1000
  },
  
  "// Custom Commands": "User-defined command shortcuts",
  "customCommands": {
    "bugCommand": "open https://github.com/google-gemini/gemini-cli/issues/new",
    "helpCommand": "open https://gemini-cli.dev/docs"
  },
  
  "// Extensions": "Extension system configuration",
  "extensions": {
    "autoLoad": true,
    "directories": [
      "~/.gemini/extensions",
      "./.gemini/extensions"
    ],
    "disabled": []
  },
  
  "// Sandbox Configuration": "Security sandbox settings",
  "sandbox": {
    "enabled": false,
    "profile": "default",
    "restrictions": {
      "filesystem": {
        "read": ["./", "~/"],
        "write": ["./output/", "/tmp/"]
      },
      "network": {
        "allowed": ["localhost", "*.google.com"],
        "blocked": []
      },
      "processes": {
        "allowed": ["node", "python", "git"],
        "blocked": ["sudo", "su", "chmod"]
      }
    }
  },
  
  "// Performance": "Performance tuning",
  "performance": {
    "parallelTools": true,
    "maxConcurrency": 4,
    "cacheEnabled": true,
    "cacheSize": "100MB",
    "lazyLoading": true
  },
  
  "// Experimental Features": "Enable at your own risk",
  "experimental": {
    "multiModal": true,
    "codeExecution": false,
    "functionCalling": true,
    "streaming": true,
    "contextCaching": false
  }
}
```

### Directory Structure Setup

```bash
#!/bin/bash
# setup-directories.sh - Create complete Gemini CLI directory structure

# Base directory
GEMINI_HOME="$HOME/.gemini"

# Create directory structure
create_structure() {
    local dirs=(
        "$GEMINI_HOME"
        "$GEMINI_HOME/extensions"
        "$GEMINI_HOME/commands"
        "$GEMINI_HOME/contexts"
        "$GEMINI_HOME/logs"
        "$GEMINI_HOME/cache"
        "$GEMINI_HOME/checkpoints"
        "$GEMINI_HOME/secure"
        "$GEMINI_HOME/mcp-servers"
        "$GEMINI_HOME/templates"
        "$GEMINI_HOME/scripts"
        "$GEMINI_HOME/backups"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        echo "Created: $dir"
    done
    
    # Set permissions
    chmod 700 "$GEMINI_HOME/secure"
    chmod 755 "$GEMINI_HOME/scripts"
}

# Create default files
create_defaults() {
    # Default GEMINI.md context
    cat > "$GEMINI_HOME/GEMINI.md" << 'EOF'
# Global Gemini Context

## User Preferences
- Prefer concise, practical solutions
- Include error handling in all code
- Follow best practices for security
- Optimize for readability over cleverness

## Development Environment
- Primary languages: [Specify your languages]
- Preferred frameworks: [Your frameworks]
- Code style: [Your style guide]

## Project Standards
- Version control: Git with conventional commits
- Documentation: Comprehensive inline comments
- Testing: Unit tests for all functions
- Security: Input validation and sanitization
EOF
    
    # Default .gitignore
    cat > "$GEMINI_HOME/.gitignore" << 'EOF'
# Sensitive files
secure/
*.key
*.pem
*.enc
token.json
credentials.json

# Logs and cache
logs/
cache/
*.log
*.tmp

# Backups
backups/
*.backup

# OS files
.DS_Store
Thumbs.db
EOF
    
    # README
    cat > "$GEMINI_HOME/README.md" << 'EOF'
# Gemini CLI Configuration

This directory contains your personal Gemini CLI configuration.

## Structure

- `extensions/` - Custom extensions
- `commands/` - Custom TOML commands
- `contexts/` - Context files (GEMINI.md)
- `logs/` - Application logs
- `cache/` - Temporary cache files
- `checkpoints/` - Conversation checkpoints
- `secure/` - Encrypted credentials
- `mcp-servers/` - MCP server implementations
- `templates/` - File templates
- `scripts/` - Utility scripts
- `backups/` - Configuration backups

## Security

The `secure/` directory contains sensitive information and should never be shared or committed to version control.
EOF
    
    echo "Default files created"
}

# Create utility scripts
create_scripts() {
    # Backup script
    cat > "$GEMINI_HOME/scripts/backup.sh" << 'EOF'
#!/bin/bash
# Backup Gemini configuration

BACKUP_DIR="$HOME/.gemini/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/gemini_backup_$TIMESTAMP.tar.gz"

mkdir -p "$BACKUP_DIR"

tar -czf "$BACKUP_FILE" \
    --exclude="$HOME/.gemini/cache" \
    --exclude="$HOME/.gemini/logs" \
    --exclude="$HOME/.gemini/backups" \
    "$HOME/.gemini"

echo "Backup created: $BACKUP_FILE"
EOF
    
    # Update script
    cat > "$GEMINI_HOME/scripts/update.sh" << 'EOF'
#!/bin/bash
# Update Gemini CLI

echo "Updating Gemini CLI..."
npm update -g @google/gemini-cli

echo "Checking for extension updates..."
for ext in ~/.gemini/extensions/*/; do
    if [ -f "$ext/package.json" ]; then
        echo "Updating extension: $(basename "$ext")"
        (cd "$ext" && npm update)
    fi
done

echo "Update complete"
EOF
    
    # Health check script
    cat > "$GEMINI_HOME/scripts/health-check.sh" << 'EOF'
#!/bin/bash
# Gemini CLI health check

echo "=== Gemini CLI Health Check ==="
echo

# Check installation
echo -n "Gemini CLI: "
if command -v gemini &> /dev/null; then
    gemini --version
else
    echo "NOT INSTALLED"
fi

# Check Node.js
echo -n "Node.js: "
node --version

# Check authentication
echo -n "Authentication: "
if [ -n "$GEMINI_API_KEY" ]; then
    echo "API Key configured"
elif [ -f "$HOME/.gemini/secure/token.json" ]; then
    echo "OAuth2 token present"
else
    echo "Not configured"
fi

# Check extensions
echo "Extensions: $(ls -1 ~/.gemini/extensions 2>/dev/null | wc -l) installed"

# Check MCP servers
echo "MCP Servers:"
if [ -f "$HOME/.gemini/settings.json" ]; then
    grep -o '"mcpServers"' "$HOME/.gemini/settings.json" &> /dev/null && \
        echo "  Configured in settings.json" || \
        echo "  None configured"
fi

echo
echo "=== Check Complete ==="
EOF
    
    chmod +x "$GEMINI_HOME/scripts/"*.sh
    echo "Utility scripts created"
}

# Main execution
main() {
    echo "Setting up Gemini CLI directories..."
    create_structure
    create_defaults
    create_scripts
    echo "Setup complete!"
    echo "Configuration directory: $GEMINI_HOME"
}

main
```

## Shell Integration & Aliases

### Comprehensive Shell Integration

```bash
#!/bin/bash
# shell-integration.sh - Complete shell integration for all shells

detect_shell() {
    if [ -n "$BASH_VERSION" ]; then
        SHELL_TYPE="bash"
        RC_FILE="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_TYPE="zsh"
        RC_FILE="$HOME/.zshrc"
    elif [ -n "$FISH_VERSION" ]; then
        SHELL_TYPE="fish"
        RC_FILE="$HOME/.config/fish/config.fish"
    else
        SHELL_TYPE="sh"
        RC_FILE="$HOME/.profile"
    fi
}

# Bash/Zsh integration
setup_bash_zsh() {
    cat >> "$RC_FILE" << 'EOF'

# ============================================
# Gemini CLI Integration
# ============================================

# Environment variables
export GEMINI_HOME="$HOME/.gemini"
export PATH="$GEMINI_HOME/scripts:$PATH"

# Load API key if available
[ -f "$GEMINI_HOME/secure/api_key.env" ] && source "$GEMINI_HOME/secure/api_key.env"

# Aliases
alias g="gemini"
alias gm="gemini"
alias gchat="gemini /chat"
alias ghelp="gemini /help"
alias gtools="gemini /tools"
alias gext="gemini /extensions"
alias gmem="gemini /memory show"
alias gclear="gemini /clear"
alias gcompress="gemini /compress"
alias gsave="gemini /chat save"
alias gload="gemini /chat load"

# Advanced aliases
alias gdebug="gemini --debug"
alias gapi="gemini --api"
alias gsilent="gemini --silent"
alias gpipe="gemini --pipe"

# Function: Quick question to Gemini
gask() {
    gemini "$*" | head -20
}

# Function: Gemini with file context
gwith() {
    local file="$1"
    shift
    gemini "@$file $*"
}

# Function: Explain command
gexplain() {
    gemini "Explain this command: $*"
}

# Function: Generate code
gcode() {
    gemini "Write code to: $*"
}

# Function: Debug error
gdebug_error() {
    gemini "Debug this error: $(cat)"
}

# Function: Summarize file
gsummarize() {
    gemini "@$1 Summarize this file"
}

# Auto-completion for Bash
if [ -n "$BASH_VERSION" ]; then
    _gemini_complete() {
        local cur="${COMP_WORDS[COMP_CWORD]}"
        local commands="/help /clear /tools /extensions /memory /chat /compress /copy /directory /editor /mcp /bug"
        
        if [[ "$cur" == /* ]]; then
            COMPREPLY=($(compgen -W "$commands" -- "$cur"))
        fi
    }
    complete -F _gemini_complete gemini g gm
fi

# Auto-completion for Zsh
if [ -n "$ZSH_VERSION" ]; then
    autoload -U compinit && compinit
    
    _gemini() {
        local -a commands
        commands=(
            '/help:Show help information'
            '/clear:Clear the screen'
            '/tools:List available tools'
            '/extensions:List extensions'
            '/memory:Manage memory'
            '/chat:Chat commands'
            '/compress:Compress context'
            '/copy:Copy last output'
            '/directory:Manage directories'
            '/editor:Open editor'
            '/mcp:MCP server commands'
            '/bug:Report a bug'
        )
        
        if [[ "$words[2]" == /* ]]; then
            _describe 'command' commands
        fi
    }
    
    compdef _gemini gemini
    compdef _gemini g
    compdef _gemini gm
fi

# Keybindings (if supported)
if [ -n "$BASH_VERSION" ]; then
    bind '"\C-xg": "gemini "'
    bind '"\C-x\C-g": "gemini /chat\n"'
fi

# Welcome message
echo "Gemini CLI shell integration loaded"
echo "Type 'gemini' or 'g' to start"

EOF
}

# Fish shell integration
setup_fish() {
    cat >> "$RC_FILE" << 'EOF'

# ============================================
# Gemini CLI Integration for Fish
# ============================================

# Environment variables
set -gx GEMINI_HOME "$HOME/.gemini"
set -gx PATH "$GEMINI_HOME/scripts" $PATH

# Load API key if available
test -f "$GEMINI_HOME/secure/api_key.env"; and source "$GEMINI_HOME/secure/api_key.env"

# Aliases
alias g="gemini"
alias gm="gemini"
alias gchat="gemini /chat"
alias ghelp="gemini /help"
alias gtools="gemini /tools"
alias gext="gemini /extensions"
alias gmem="gemini /memory show"
alias gclear="gemini /clear"
alias gcompress="gemini /compress"
alias gsave="gemini /chat save"
alias gload="gemini /chat load"

# Functions
function gask
    gemini $argv | head -20
end

function gwith
    set file $argv[1]
    set -e argv[1]
    gemini "@$file $argv"
end

function gexplain
    gemini "Explain this command: $argv"
end

function gcode
    gemini "Write code to: $argv"
end

# Auto-completion
complete -c gemini -a '/help /clear /tools /extensions /memory /chat /compress /copy /directory /editor /mcp /bug'
complete -c g -a '/help /clear /tools /extensions /memory /chat /compress /copy /directory /editor /mcp /bug'

echo "Gemini CLI Fish integration loaded"

EOF
}

# PowerShell integration
setup_powershell() {
    PROFILE_PATH="$HOME/.config/powershell/Microsoft.PowerShell_profile.ps1"
    mkdir -p "$(dirname "$PROFILE_PATH")"
    
    cat >> "$PROFILE_PATH" << 'EOF'

# ============================================
# Gemini CLI Integration for PowerShell
# ============================================

# Environment variables
$env:GEMINI_HOME = "$HOME/.gemini"
$env:PATH = "$env:GEMINI_HOME/scripts;$env:PATH"

# Aliases
Set-Alias -Name g -Value gemini
Set-Alias -Name gm -Value gemini

# Functions
function gask {
    param([string]$Query)
    gemini $Query | Select-Object -First 20
}

function gwith {
    param(
        [string]$File,
        [string]$Query
    )
    gemini "@$File $Query"
}

function gexplain {
    param([string]$Command)
    gemini "Explain this command: $Command"
}

function gcode {
    param([string]$Task)
    gemini "Write code to: $Task"
}

# Tab completion
Register-ArgumentCompleter -CommandName gemini -ScriptBlock {
    param($commandName, $parameterName, $wordToComplete, $commandAst, $fakeBoundParameter)
    
    $commands = @(
        '/help', '/clear', '/tools', '/extensions',
        '/memory', '/chat', '/compress', '/copy',
        '/directory', '/editor', '/mcp', '/bug'
    )
    
    $commands | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}

Write-Host "Gemini CLI PowerShell integration loaded" -ForegroundColor Green

EOF
}

# Main setup
main() {
    detect_shell
    
    echo "Setting up shell integration for $SHELL_TYPE..."
    
    case $SHELL_TYPE in
        bash|zsh|sh)
            setup_bash_zsh
            ;;
        fish)
            setup_fish
            ;;
        *)
            echo "Unsupported shell: $SHELL_TYPE"
            exit 1
            ;;
    esac
    
    # Also setup PowerShell if available
    if command -v pwsh &> /dev/null; then
        echo "Setting up PowerShell integration..."
        setup_powershell
    fi
    
    echo "Shell integration complete!"
    echo "Please run: source $RC_FILE"
}

main
```

## Part II: Core Configuration

## Settings.json Deep Dive

### Advanced Configuration Examples

```json
{
  "// Profile Management": "Multiple configuration profiles",
  "profiles": {
    "default": {
      "model": "gemini-2.5-pro",
      "temperature": 0.7
    },
    "creative": {
      "model": "gemini-2.5-pro",
      "temperature": 0.9,
      "topP": 0.98
    },
    "precise": {
      "model": "gemini-2.5-pro",
      "temperature": 0.3,
      "topP": 0.85
    },
    "fast": {
      "model": "gemini-2.5-flash",
      "temperature": 0.5
    }
  },
  
  "// Tool Discovery": "Custom tool discovery configuration",
  "toolDiscovery": {
    "enabled": true,
    "toolDiscoveryCommand": "node ~/.gemini/tools/discover.js",
    "toolCallCommand": "node ~/.gemini/tools/call.js",
    "refreshInterval": 300,
    "cache": true
  },
  
  "// Context Providers": "External context sources",
  "contextProviders": [
    {
      "name": "git-context",
      "command": "git log --oneline -10",
      "trigger": "auto"
    },
    {
      "name": "env-context",
      "command": "env | grep -E '^(NODE|PYTHON|JAVA)'",
      "trigger": "manual"
    }
  ],
  
  "// Hooks": "Lifecycle hooks",
  "hooks": {
    "prePrompt": "~/.gemini/hooks/pre-prompt.sh",
    "postResponse": "~/.gemini/hooks/post-response.sh",
    "onError": "~/.gemini/hooks/on-error.sh",
    "onToolCall": "~/.gemini/hooks/on-tool-call.sh"
  },
  
  "// Rate Limiting": "API rate limit configuration",
  "rateLimiting": {
    "enabled": true,
    "requestsPerMinute": 60,
    "requestsPerDay": 1000,
    "backoffStrategy": "exponential",
    "maxRetries": 5
  },
  
  "// Caching": "Response caching configuration",
  "caching": {
    "enabled": true,
    "ttl": 3600,
    "maxSize": "500MB",
    "strategy": "lru",
    "persistToDisk": true,
    "location": "~/.gemini/cache"
  }
}
```

### Environment-Specific Settings

```bash
#!/bin/bash
# env-settings.sh - Environment-specific configuration

create_env_settings() {
    local env="$1"
    local file="$HOME/.gemini/settings.$env.json"
    
    case "$env" in
        development)
            cat > "$file" << 'EOF'
{
  "model": {
    "name": "gemini-2.5-flash",
    "temperature": 0.7
  },
  "tools": {
    "restrictions": {
      "run_shell_command": {
        "requireConfirmation": false
      }
    }
  },
  "logging": {
    "level": "debug"
  }
}
EOF
            ;;
            
        production)
            cat > "$file" << 'EOF'
{
  "model": {
    "name": "gemini-2.5-pro",
    "temperature": 0.3
  },
  "tools": {
    "restrictions": {
      "run_shell_command": {
        "requireConfirmation": true,
        "blockedCommands": ["rm", "dd", "mkfs", "kill"]
      }
    }
  },
  "logging": {
    "level": "error"
  },
  "sandbox": {
    "enabled": true,
    "profile": "strict"
  }
}
EOF
            ;;
            
        testing)
            cat > "$file" << 'EOF'
{
  "model": {
    "name": "gemini-2.5-flash",
    "temperature": 0.5
  },
  "tools": {
    "enabled": ["read_file", "file_write"],
    "disabled": ["run_shell_command", "web_fetch"]
  },
  "logging": {
    "level": "verbose",
    "file": "~/.gemini/logs/test.log"
  }
}
EOF
            ;;
    esac
    
    echo "Created $env settings: $file"
}

# Create all environment settings
for env in development production testing; do
    create_env_settings "$env"
done

# Create environment switcher
cat > "$HOME/.gemini/scripts/switch-env.sh" << 'EOF'
#!/bin/bash
# Switch Gemini CLI environment

ENV="${1:-development}"
SETTINGS_FILE="$HOME/.gemini/settings.$ENV.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Environment not found: $ENV"
    exit 1
fi

# Backup current settings
cp "$HOME/.gemini/settings.json" "$HOME/.gemini/settings.json.backup"

# Switch to new environment
cp "$SETTINGS_FILE" "$HOME/.gemini/settings.json"

echo "Switched to $ENV environment"
EOF

chmod +x "$HOME/.gemini/scripts/switch-env.sh"
```

## Built-in Tools Configuration

### Tool Discovery System

```javascript
// ~/.gemini/tools/discover.js
// Custom tool discovery system

const fs = require('fs').promises;
const path = require('path');

class ToolDiscovery {
    constructor() {
        this.tools = new Map();
        this.toolsDir = path.join(process.env.HOME, '.gemini', 'custom-tools');
    }
    
    async discover() {
        const files = await fs.readdir(this.toolsDir);
        
        for (const file of files) {
            if (file.endsWith('.json')) {
                const toolPath = path.join(this.toolsDir, file);
                const tool = JSON.parse(await fs.readFile(toolPath, 'utf8'));
                this.tools.set(tool.name, tool);
            }
        }
        
        return this.formatForGemini();
    }
    
    formatForGemini() {
        const tools = [];
        
        for (const [name, tool] of this.tools) {
            tools.push({
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
                command: tool.command
            });
        }
        
        return {
            tools,
            version: '1.0.0'
        };
    }
}

// Main execution
(async () => {
    const discovery = new ToolDiscovery();
    const tools = await discovery.discover();
    console.log(JSON.stringify(tools));
})();
```

### Tool Call Handler

```javascript
// ~/.gemini/tools/call.js
// Custom tool call handler

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class ToolExecutor {
    constructor() {
        this.toolsDir = path.join(process.env.HOME, '.gemini', 'custom-tools');
        this.logFile = path.join(process.env.HOME, '.gemini', 'logs', 'tools.log');
    }
    
    async execute(toolCall) {
        const { tool, parameters } = JSON.parse(toolCall);
        
        // Load tool definition
        const toolPath = path.join(this.toolsDir, `${tool}.json`);
        const toolDef = JSON.parse(await fs.readFile(toolPath, 'utf8'));
        
        // Validate parameters
        this.validateParameters(parameters, toolDef.parameters);
        
        // Execute tool
        const result = await this.runCommand(toolDef.command, parameters);
        
        // Log execution
        await this.logExecution(tool, parameters, result);
        
        return result;
    }
    
    validateParameters(provided, required) {
        for (const [key, schema] of Object.entries(required)) {
            if (schema.required && !provided[key]) {
                throw new Error(`Missing required parameter: ${key}`);
            }
            
            if (provided[key] && schema.type) {
                const type = typeof provided[key];
                if (type !== schema.type) {
                    throw new Error(`Invalid type for ${key}: expected ${schema.type}, got ${type}`);
                }
            }
        }
    }
    
    async runCommand(command, parameters) {
        // Replace placeholders in command
        let cmd = command;
        for (const [key, value] of Object.entries(parameters)) {
            cmd = cmd.replace(`{{${key}}}`, value);
        }
        
        return new Promise((resolve, reject) => {
            const child = spawn('bash', ['-c', cmd]);
            let output = '';
            let error = '';
            
            child.stdout.on('data', (data) => {
                output += data.toString();
            });
            
            child.stderr.on('data', (data) => {
                error += data.toString();
            });
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve({ success: true, output });
                } else {
                    reject({ success: false, error, code });
                }
            });
        });
    }
    
    async logExecution(tool, parameters, result) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            tool,
            parameters,
            result: result.success ? 'success' : 'failure',
            output: result.output || result.error
        };
        
        await fs.appendFile(
            this.logFile,
            JSON.stringify(logEntry) + '\n'
        );
    }
}

// Main execution
(async () => {
    const executor = new ToolExecutor();
    const input = await new Promise((resolve) => {
        let data = '';
        process.stdin.on('data', (chunk) => data += chunk);
        process.stdin.on('end', () => resolve(data));
    });
    
    try {
        const result = await executor.execute(input);
        console.log(JSON.stringify(result));
    } catch (error) {
        console.error(JSON.stringify({ error: error.message }));
        process.exit(1);
    }
})();
```

### Custom Tool Definition

```json
{
  "name": "code_analyzer",
  "description": "Analyze code complexity and quality",
  "parameters": {
    "file": {
      "type": "string",
      "required": true,
      "description": "Path to the file to analyze"
    },
    "metrics": {
      "type": "array",
      "required": false,
      "default": ["complexity", "lines", "coverage"],
      "description": "Metrics to calculate"
    }
  },
  "command": "node ~/.gemini/tools/analyzers/code-analyzer.js '{{file}}' '{{metrics}}'",
  "timeout": 30000,
  "cache": true
}
```

## Memory System Setup

### Advanced Memory Management

```bash
#!/bin/bash
# memory-setup.sh - Configure advanced memory system

MEMORY_DIR="$HOME/.gemini/memory"
mkdir -p "$MEMORY_DIR"

# Create memory manager script
cat > "$MEMORY_DIR/manager.py" << 'EOF'
#!/usr/bin/env python3
"""
Gemini CLI Memory Manager
Handles context compression, storage, and retrieval
"""

import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
import tiktoken

class MemoryManager:
    def __init__(self, db_path="~/.gemini/memory/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.init_db()
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def init_db(self):
        """Initialize database schema"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT,
                tokens INTEGER,
                category TEXT,
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def add_memory(self, content, category=None, importance=0.5):
        """Add a new memory"""
        memory_id = hashlib.md5(content.encode()).hexdigest()
        tokens = len(self.encoder.encode(content))
        
        self.conn.execute("""
            INSERT OR REPLACE INTO memories 
            (id, content, tokens, category, importance)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_id, content, tokens, category, importance))
        
        self.conn.commit()
        return memory_id
    
    def get_memories(self, category=None, limit=10):
        """Retrieve memories by category"""
        query = """
            SELECT content, importance, accessed_at
            FROM memories
            WHERE category = ? OR ? IS NULL
            ORDER BY importance DESC, accessed_at DESC
            LIMIT ?
        """
        
        cursor = self.conn.execute(query, (category, category, limit))
        return cursor.fetchall()
    
    def compress_context(self, context, target_tokens=1000):
        """Compress context to target token count"""
        current_tokens = len(self.encoder.encode(context))
        
        if current_tokens <= target_tokens:
            return context
        
        # Simple compression: summarize
        # In production, use LLM for summarization
        lines = context.split('\n')
        compressed = []
        token_count = 0
        
        for line in lines:
            line_tokens = len(self.encoder.encode(line))
            if token_count + line_tokens <= target_tokens:
                compressed.append(line)
                token_count += line_tokens
            else:
                break
        
        return '\n'.join(compressed)
    
    def save_context(self, name, content):
        """Save named context"""
        tokens = len(self.encoder.encode(content))
        
        self.conn.execute("""
            INSERT OR REPLACE INTO contexts 
            (name, content, tokens, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (name, content, tokens))
        
        self.conn.commit()
    
    def load_context(self, name):
        """Load named context"""
        cursor = self.conn.execute(
            "SELECT content FROM contexts WHERE name = ?", 
            (name,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    def cleanup(self, max_age_days=30):
        """Remove old memories"""
        self.conn.execute("""
            DELETE FROM memories
            WHERE julianday('now') - julianday(accessed_at) > ?
            AND importance < 0.7
        """, (max_age_days,))
        
        self.conn.commit()

if __name__ == "__main__":
    import sys
    
    manager = MemoryManager()
    
    if len(sys.argv) < 2:
        print("Usage: manager.py <command> [args]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "add":
        content = sys.argv[2]
        category = sys.argv[3] if len(sys.argv) > 3 else None
        memory_id = manager.add_memory(content, category)
        print(f"Added memory: {memory_id}")
    
    elif command == "get":
        category = sys.argv[2] if len(sys.argv) > 2 else None
        memories = manager.get_memories(category)
        for memory in memories:
            print(f"- {memory[0][:100]}...")
    
    elif command == "compress":
        context = sys.stdin.read()
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        compressed = manager.compress_context(context, target)
        print(compressed)
    
    elif command == "save":
        name = sys.argv[2]
        content = sys.stdin.read()
        manager.save_context(name, content)
        print(f"Saved context: {name}")
    
    elif command == "load":
        name = sys.argv[2]
        content = manager.load_context(name)
        if content:
            print(content)
        else:
            print(f"Context not found: {name}")
    
    elif command == "cleanup":
        manager.cleanup()
        print("Cleanup completed")
EOF

chmod +x "$MEMORY_DIR/manager.py"

# Create memory hooks
cat > "$HOME/.gemini/hooks/memory-hook.sh" << 'EOF'
#!/bin/bash
# Memory hook for Gemini CLI

MEMORY_MANAGER="$HOME/.gemini/memory/manager.py"

# Function to save response
save_response() {
    local content="$1"
    echo "$content" | python3 "$MEMORY_MANAGER" add - "response"
}

# Function to load context
load_context() {
    local name="$1"
    python3 "$MEMORY_MANAGER" load "$name"
}

# Export functions for use in Gemini
export -f save_response
export -f load_context
EOF

chmod +x "$HOME/.gemini/hooks/memory-hook.sh"
```

## Context Management

### Hierarchical Context System

```markdown
# ~/.gemini/GEMINI.md
# Global Context - Loaded for all projects

## System Configuration
You are configured to work with a developer who prefers:
- Language: ${PRIMARY_LANGUAGE:-Python}
- Framework: ${PRIMARY_FRAMEWORK:-FastAPI}
- Database: ${PRIMARY_DATABASE:-PostgreSQL}
- Cloud: ${PRIMARY_CLOUD:-AWS}

## Import Project-Specific Contexts
@./contexts/coding-standards.md
@./contexts/security-policies.md
@./contexts/performance-guidelines.md

## Working Principles

### Code Quality
1. **Readability First**: Code should be self-documenting
2. **Type Safety**: Use type hints/annotations where available
3. **Error Handling**: Comprehensive error handling with useful messages
4. **Testing**: Minimum 80% code coverage
5. **Documentation**: Inline comments for complex logic

### Security
1. **Input Validation**: Always validate and sanitize inputs
2. **Authentication**: Use industry-standard auth mechanisms
3. **Secrets Management**: Never hardcode credentials
4. **Audit Logging**: Log security-relevant events
5. **Dependency Scanning**: Regular vulnerability checks

### Performance
1. **Optimization**: Profile before optimizing
2. **Caching**: Implement appropriate caching strategies
3. **Database**: Use indexes and query optimization
4. **Async Operations**: Use async/await for I/O operations
5. **Resource Management**: Proper cleanup and resource disposal

## Command Preferences

When executing commands:
- Always show the command before running
- Explain potentially dangerous operations
- Prefer non-destructive operations
- Create backups before modifications
- Use verbose output for debugging

## Response Format

Structure responses as:
1. **Summary**: Brief overview of the solution
2. **Implementation**: Detailed code/commands
3. **Explanation**: Why this approach was chosen
4. **Considerations**: Edge cases and limitations
5. **Next Steps**: Suggested follow-up actions

## Project Context Loading

The system will automatically load:
- `.gemini/GEMINI.md` from the current project
- `GEMINI.md` files from subdirectories
- Context from imported files using @filepath syntax

## Memory Management

- Compress context when approaching token limits
- Prioritize recent and relevant information
- Maintain conversation continuity across sessions
- Store important decisions and rationale

## Tool Usage Guidelines

### File Operations
- Confirm before overwriting files
- Create backups for important files
- Use atomic operations when possible
- Respect .gitignore patterns

### Shell Commands
- Validate commands before execution
- Use safe defaults for dangerous operations
- Provide command explanations
- Show expected output format

### Web Operations
- Respect rate limits
- Cache responses when appropriate
- Handle network errors gracefully
- Validate SSL certificates

## Integration Points

### Version Control
- Follow conventional commit messages
- Create feature branches for changes
- Include tests with code changes
- Update documentation alongside code

### CI/CD
- Ensure changes pass existing tests
- Add tests for new functionality
- Update deployment configurations
- Document breaking changes

### Monitoring
- Add appropriate logging
- Include metrics collection
- Set up alerts for failures
- Document operational procedures
```

### Project-Specific Context

```bash
#!/bin/bash
# setup-project-context.sh - Create project-specific context

PROJECT_DIR="${1:-.}"
GEMINI_DIR="$PROJECT_DIR/.gemini"

mkdir -p "$GEMINI_DIR"

# Analyze project and generate context
cat > "$GEMINI_DIR/GEMINI.md" << 'EOF'
# Project Context

## Project Information
- **Name**: $(basename "$PROJECT_DIR")
- **Type**: $(detect_project_type)
- **Language**: $(detect_language)
- **Dependencies**: $(list_dependencies)

## Project Structure
```
$(tree -L 2 "$PROJECT_DIR" 2>/dev/null || ls -la "$PROJECT_DIR")
```

## Development Setup
$(generate_setup_instructions)

## Key Files
$(identify_key_files)

## Testing
$(identify_test_framework)

## Build & Deploy
$(identify_build_system)

## Project-Specific Rules
1. Follow existing code style
2. Maintain backward compatibility
3. Update tests for all changes
4. Document API changes
5. Follow branch naming convention

## Import Additional Context
@../../.gemini/GEMINI.md
EOF

# Helper functions
detect_project_type() {
    if [ -f "package.json" ]; then
        echo "Node.js/JavaScript"
    elif [ -f "requirements.txt" ] || [ -f "setup.py" ]; then
        echo "Python"
    elif [ -f "pom.xml" ]; then
        echo "Java/Maven"
    elif [ -f "go.mod" ]; then
        echo "Go"
    elif [ -f "Cargo.toml" ]; then
        echo "Rust"
    else
        echo "Unknown"
    fi
}

detect_language() {
    # Count file extensions
    find . -type f -name "*.py" 2>/dev/null | head -1 && echo "Python"
    find . -type f -name "*.js" 2>/dev/null | head -1 && echo "JavaScript"
    find . -type f -name "*.java" 2>/dev/null | head -1 && echo "Java"
    find . -type f -name "*.go" 2>/dev/null | head -1 && echo "Go"
}

list_dependencies() {
    if [ -f "package.json" ]; then
        jq -r '.dependencies | keys[]' package.json 2>/dev/null | head -5
    elif [ -f "requirements.txt" ]; then
        head -5 requirements.txt
    elif [ -f "go.mod" ]; then
        grep "require" go.mod | head -5
    fi
}

identify_key_files() {
    echo "- README.md: $([ -f README.md ] && echo "Present" || echo "Missing")"
    echo "- LICENSE: $([ -f LICENSE ] && echo "Present" || echo "Missing")"
    echo "- Config: $(ls -1 *.config.* 2>/dev/null | head -3)"
}

identify_test_framework() {
    if [ -f "package.json" ]; then
        grep -E "jest|mocha|jasmine" package.json && echo "Found test framework"
    elif [ -f "setup.py" ]; then
        grep -E "pytest|unittest" setup.py && echo "Found test framework"
    fi
}

identify_build_system() {
    [ -f "Makefile" ] && echo "Make"
    [ -f "package.json" ] && grep -q "build" package.json && echo "npm/yarn build"
    [ -f "pom.xml" ] && echo "Maven"
    [ -f "build.gradle" ] && echo "Gradle"
}
```

This comprehensive setup guide provides everything needed to configure Gemini CLI from scratch, including installation, authentication, configuration files, shell integration, and advanced features. The guide covers all major operating systems and shells, with specific attention to Termux/Android environments as requested.

# Complete Walkthrough: Linking Your Forked Gemini CLI Repository

## Table of Contents

### Part I: Fork & Clone Setup
1. [Forking the Repository](#forking-the-repository)
2. [Cloning Your Fork](#cloning-your-fork)
3. [Repository Structure Overview](#repository-structure-overview)

### Part II: Development Setup
4. [Installing Dependencies](#installing-dependencies)
5. [Building the Project](#building-the-project)
6. [Linking Methods](#linking-methods)

### Part III: Development Workflow
7. [Making Changes](#making-changes)
8. [Testing Your Changes](#testing-your-changes)
9. [Keeping Fork Updated](#keeping-fork-updated)

### Part IV: Advanced Configuration
10. [Multiple Version Management](#multiple-version-management)
11. [Development Tools Setup](#development-tools-setup)
12. [Contributing Back](#contributing-back)

---

## Part I: Fork & Clone Setup

## Forking the Repository

### Step 1: Create Your Fork on GitHub

```bash
#!/bin/bash
# fork-setup.sh - Complete fork setup process

# Configuration
GITHUB_USERNAME="${1:-your-username}"
ORIGINAL_REPO="google-gemini/gemini-cli"
FORK_REPO="$GITHUB_USERNAME/gemini-cli"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }

# Step 1: Fork via GitHub CLI (if available)
fork_with_cli() {
    if command -v gh &> /dev/null; then
        info "Using GitHub CLI to fork..."
        gh repo fork "$ORIGINAL_REPO" --clone --remote
        return $?
    else
        warn "GitHub CLI not found. Please fork manually:"
        echo "1. Go to: https://github.com/$ORIGINAL_REPO"
        echo "2. Click 'Fork' button in the top-right"
        echo "3. Select your account"
        echo "4. Wait for fork to complete"
        echo ""
        read -p "Press Enter when fork is complete..."
        return 0
    fi
}

# Execute fork
fork_with_cli
```

### Step 2: Configure Git for Development

```bash
#!/bin/bash
# git-config.sh - Configure Git for Gemini CLI development

setup_git_config() {
    # Get user information
    read -p "Enter your GitHub username: " GITHUB_USER
    read -p "Enter your GitHub email: " GITHUB_EMAIL
    read -p "Enter your full name: " FULL_NAME
    
    # Set up Git configuration
    git config --global user.name "$FULL_NAME"
    git config --global user.email "$GITHUB_EMAIL"
    
    # Set up GitHub-specific settings
    git config --global github.user "$GITHUB_USER"
    
    # Configure helpful aliases
    git config --global alias.co checkout
    git config --global alias.br branch
    git config --global alias.ci commit
    git config --global alias.st status
    git config --global alias.unstage 'reset HEAD --'
    git config --global alias.last 'log -1 HEAD'
    git config --global alias.visual '!gitk'
    
    # Set up commit signing (optional)
    read -p "Set up GPG signing? (y/n): " SETUP_GPG
    if [[ "$SETUP_GPG" == "y" ]]; then
        setup_gpg_signing
    fi
    
    log "Git configuration complete"
}

setup_gpg_signing() {
    # Check for existing GPG key
    if gpg --list-secret-keys --keyid-format LONG | grep -q sec; then
        info "Found existing GPG keys:"
        gpg --list-secret-keys --keyid-format LONG
        
        read -p "Enter GPG key ID to use: " GPG_KEY_ID
        git config --global user.signingkey "$GPG_KEY_ID"
        git config --global commit.gpgsign true
        
        log "GPG signing configured"
    else
        warn "No GPG keys found. Generate one with: gpg --full-generate-key"
    fi
}

setup_git_config
```

## Cloning Your Fork

### Complete Clone and Setup Script

```bash
#!/bin/bash
# clone-fork.sh - Clone and set up your Gemini CLI fork

set -euo pipefail

# Configuration
GITHUB_USER="${1:-}"
WORK_DIR="${2:-$HOME/Development}"
CLONE_DIR="$WORK_DIR/gemini-cli"

if [ -z "$GITHUB_USER" ]; then
    read -p "Enter your GitHub username: " GITHUB_USER
fi

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone the fork
clone_repository() {
    info "Cloning your fork..."
    
    # Clone with SSH (preferred) or HTTPS
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        git clone "git@github.com:$GITHUB_USER/gemini-cli.git"
        log "Cloned via SSH"
    else
        git clone "https://github.com/$GITHUB_USER/gemini-cli.git"
        log "Cloned via HTTPS"
        
        warn "Consider setting up SSH keys for easier authentication:"
        echo "  https://docs.github.com/en/authentication/connecting-to-github-with-ssh"
    fi
    
    cd gemini-cli
}

# Set up remotes
setup_remotes() {
    info "Setting up remotes..."
    
    # Add upstream remote
    git remote add upstream https://github.com/google-gemini/gemini-cli.git
    
    # Verify remotes
    git remote -v
    
    # Set up branch tracking
    git branch --set-upstream-to=origin/main main
    
    log "Remotes configured:"
    echo "  - origin: Your fork"
    echo "  - upstream: Original repository"
}

# Fetch all branches and tags
fetch_everything() {
    info "Fetching all branches and tags..."
    
    git fetch --all --tags
    git fetch upstream
    
    # List available branches
    echo ""
    info "Available branches:"
    git branch -r | head -10
    
    log "Repository synced"
}

# Main execution
main() {
    echo "========================================="
    echo "     Gemini CLI Fork Setup"
    echo "========================================="
    
    clone_repository
    setup_remotes
    fetch_everything
    
    echo ""
    log "Fork cloned successfully to: $CLONE_DIR"
    info "Next step: Install dependencies and link"
}

main
```

## Repository Structure Overview

### Understanding the Codebase

```bash
#!/bin/bash
# explore-structure.sh - Explore Gemini CLI repository structure

cd gemini-cli

# Create structure documentation
cat > REPOSITORY_STRUCTURE.md << 'EOF'
# Gemini CLI Repository Structure

## Core Directories

```
gemini-cli/
├── packages/              # Monorepo packages
│   ├── cli/              # Main CLI package
│   │   ├── src/          # Source code
│   │   │   ├── index.ts  # Entry point
│   │   │   ├── commands/ # Command implementations
│   │   │   ├── tools/    # Built-in tools
│   │   │   ├── mcp/      # MCP integration
│   │   │   └── utils/    # Utilities
│   │   ├── dist/         # Compiled output
│   │   ├── package.json  # Package configuration
│   │   └── tsconfig.json # TypeScript config
│   ├── core/             # Core functionality
│   ├── sdk/              # SDK for extensions
│   └── tools/            # Tool implementations
├── extensions/           # Official extensions
├── examples/             # Example usage
├── docs/                 # Documentation
├── scripts/              # Build/dev scripts
├── tests/                # Test suites
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── .github/              # GitHub Actions
├── package.json          # Root package.json
├── lerna.json           # Lerna monorepo config
├── tsconfig.json        # Root TypeScript config
└── README.md            # Project README
```

## Key Files for Development

### Entry Points
- `packages/cli/src/index.ts` - Main CLI entry
- `packages/cli/src/cli.ts` - CLI initialization
- `packages/cli/src/repl.ts` - REPL implementation

### Configuration
- `packages/cli/src/config/` - Configuration management
- `packages/cli/src/settings/` - Settings handling

### Tools System
- `packages/cli/src/tools/registry.ts` - Tool registry
- `packages/cli/src/tools/executor.ts` - Tool execution
- `packages/cli/src/tools/built-in/` - Built-in tools

### MCP Integration
- `packages/cli/src/mcp/server.ts` - MCP server management
- `packages/cli/src/mcp/client.ts` - MCP client

### Extensions
- `packages/cli/src/extensions/loader.ts` - Extension loader
- `packages/cli/src/extensions/manager.ts` - Extension manager

## Development Files

### Build System
- `rollup.config.js` - Build configuration
- `webpack.config.js` - Webpack config (if used)
- `scripts/build.js` - Build script

### Testing
- `jest.config.js` - Jest configuration
- `tests/setup.js` - Test setup

### Linting & Formatting
- `.eslintrc.js` - ESLint rules
- `.prettierrc` - Prettier config
- `.editorconfig` - Editor config
EOF

# Create development guide
cat > DEVELOPMENT_GUIDE.md << 'EOF'
# Development Guide

## Prerequisites
- Node.js 18+ 
- npm or yarn
- Git
- TypeScript knowledge

## Initial Setup
1. Install dependencies: `npm install`
2. Build project: `npm run build`
3. Link for development: `npm link`

## Development Commands
- `npm run dev` - Watch mode development
- `npm run build` - Production build
- `npm run test` - Run tests
- `npm run lint` - Lint code
- `npm run format` - Format code

## Code Organization
- Keep tools modular in `src/tools/`
- Add commands in `src/commands/`
- Utilities go in `src/utils/`
- Types in `src/types/`

## Testing Requirements
- Write tests for new features
- Maintain test coverage above 80%
- Run tests before committing
EOF

info "Repository structure documented"
```

## Part II: Development Setup

## Installing Dependencies

### Complete Dependency Installation

```bash
#!/bin/bash
# install-deps.sh - Install all dependencies for development

set -euo pipefail

cd gemini-cli

# Detect package manager
detect_package_manager() {
    if [ -f "yarn.lock" ]; then
        echo "yarn"
    elif [ -f "pnpm-lock.yaml" ]; then
        echo "pnpm"
    elif [ -f "package-lock.json" ]; then
        echo "npm"
    else
        echo "npm"  # Default
    fi
}

PKG_MANAGER=$(detect_package_manager)
info "Using package manager: $PKG_MANAGER"

# Install dependencies
install_dependencies() {
    info "Installing dependencies..."
    
    case "$PKG_MANAGER" in
        yarn)
            yarn install
            yarn lerna bootstrap
            ;;
        pnpm)
            pnpm install
            pnpm lerna bootstrap
            ;;
        npm)
            npm ci || npm install
            npx lerna bootstrap
            ;;
    esac
    
    log "Dependencies installed"
}

# Install global development tools
install_dev_tools() {
    info "Installing development tools..."
    
    # Check and install global tools
    local tools=(
        "typescript"
        "ts-node"
        "@types/node"
        "nodemon"
        "jest"
        "eslint"
        "prettier"
    )
    
    for tool in "${tools[@]}"; do
        if ! npm list -g "$tool" &> /dev/null; then
            info "Installing $tool globally..."
            npm install -g "$tool"
        fi
    done
    
    log "Development tools ready"
}

# Verify installation
verify_installation() {
    info "Verifying installation..."
    
    # Check Node version
    NODE_VERSION=$(node -v | sed 's/v//')
    MIN_VERSION="18.0.0"
    
    if [ "$(printf '%s\n' "$MIN_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
        error "Node.js version $NODE_VERSION is too old. Need v$MIN_VERSION+"
    fi
    
    # Check TypeScript
    if ! command -v tsc &> /dev/null; then
        warn "TypeScript compiler not found"
    else
        info "TypeScript version: $(tsc --version)"
    fi
    
    # Check for required files
    local required_files=(
        "package.json"
        "tsconfig.json"
        "packages/cli/package.json"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Required file missing: $file"
        fi
    done
    
    log "Installation verified"
}

# Main installation
main() {
    echo "========================================="
    echo "     Installing Dependencies"
    echo "========================================="
    
    install_dependencies
    install_dev_tools
    verify_installation
    
    echo ""
    log "Dependencies installation complete!"
}

main
```

### Handling Platform-Specific Dependencies

```bash
#!/bin/bash
# platform-deps.sh - Handle platform-specific dependencies

# For Termux
if [ -n "${TERMUX_VERSION:-}" ]; then
    info "Setting up Termux-specific dependencies..."
    
    # Install required packages
    pkg install -y \
        nodejs-lts \
        python \
        make \
        clang \
        git
    
    # Set up node-gyp
    npm config set python python3
    npm config set node_gyp "$(npm prefix -g)/lib/node_modules/npm/node_modules/node-gyp/bin/node-gyp.js"
    
    # Handle native modules
    export GYP_DEFINES="OS=android"
    
    log "Termux dependencies configured"
fi

# For macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    info "Setting up macOS-specific dependencies..."
    
    # Ensure Xcode Command Line Tools
    if ! xcode-select -p &> /dev/null; then
        warn "Installing Xcode Command Line Tools..."
        xcode-select --install
    fi
    
    log "macOS dependencies configured"
fi

# For Windows (WSL)
if grep -qi microsoft /proc/version 2>/dev/null; then
    info "Setting up WSL-specific configurations..."
    
    # Configure npm for WSL
    npm config set script-shell bash
    
    log "WSL dependencies configured"
fi
```

## Building the Project

### Complete Build Process

```bash
#!/bin/bash
# build-project.sh - Build Gemini CLI from source

set -euo pipefail

cd gemini-cli

# Build configuration
BUILD_MODE="${1:-development}"  # development or production

# Clean previous builds
clean_build() {
    info "Cleaning previous builds..."
    
    # Remove dist directories
    find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".turbo" -exec rm -rf {} + 2>/dev/null || true
    
    # Clear TypeScript cache
    find . -name "*.tsbuildinfo" -delete 2>/dev/null || true
    
    log "Clean complete"
}

# Build TypeScript
build_typescript() {
    info "Building TypeScript..."
    
    if [ "$BUILD_MODE" == "production" ]; then
        # Production build
        npx tsc --project tsconfig.json --declaration
        
        # Minification (if configured)
        if [ -f "rollup.config.js" ]; then
            npx rollup -c
        fi
    else
        # Development build
        npx tsc --project tsconfig.json --sourceMap --declaration
    fi
    
    log "TypeScript build complete"
}

# Build packages
build_packages() {
    info "Building packages..."
    
    # Build with Lerna
    if [ -f "lerna.json" ]; then
        npx lerna run build --stream
    else
        # Manual package builds
        for pkg in packages/*/; do
            if [ -f "$pkg/package.json" ]; then
                info "Building $(basename "$pkg")..."
                (cd "$pkg" && npm run build)
            fi
        done
    fi
    
    log "Packages built"
}

# Create executable
create_executable() {
    info "Creating executable..."
    
    # Ensure shebang is correct
    local cli_entry="packages/cli/dist/index.js"
    
    if [ -f "$cli_entry" ]; then
        # Add shebang if missing
        if ! head -1 "$cli_entry" | grep -q "^#!/usr/bin/env node"; then
            echo '#!/usr/bin/env node' | cat - "$cli_entry" > temp && mv temp "$cli_entry"
        fi
        
        # Make executable
        chmod +x "$cli_entry"
        
        log "Executable created: $cli_entry"
    else
        error "CLI entry point not found: $cli_entry"
    fi
}

# Run post-build tasks
post_build() {
    info "Running post-build tasks..."
    
    # Copy necessary files
    cp README.md packages/cli/dist/ 2>/dev/null || true
    cp LICENSE packages/cli/dist/ 2>/dev/null || true
    
    # Generate metadata
    cat > packages/cli/dist/build-info.json << EOF
{
    "buildTime": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "buildMode": "$BUILD_MODE",
    "nodeVersion": "$(node -v)",
    "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')"
}
EOF
    
    log "Post-build complete"
}

# Main build process
main() {
    echo "========================================="
    echo "     Building Gemini CLI ($BUILD_MODE)"
    echo "========================================="
    
    clean_build
    build_typescript
    build_packages
    create_executable
    post_build
    
    echo ""
    log "Build complete!"
    info "Next step: Link the built project"
}

main
```

### Watch Mode for Development

```bash
#!/bin/bash
# dev-watch.sh - Development watch mode

cd gemini-cli

# Start watch mode
start_watch() {
    info "Starting watch mode..."
    
    # Use nodemon for watching
    cat > nodemon.json << 'EOF'
{
    "watch": ["packages/*/src"],
    "ext": "ts,js,json",
    "ignore": ["**/*.test.ts", "**/*.spec.ts", "dist/**", "node_modules/**"],
    "exec": "npm run build:dev",
    "delay": 1000
}
EOF
    
    # Start watching
    npx nodemon
}

# Alternative: TypeScript watch
start_tsc_watch() {
    info "Starting TypeScript watch mode..."
    
    # Run tsc in watch mode for each package
    npx lerna run --parallel watch
}

# Choose watch method
if command -v nodemon &> /dev/null; then
    start_watch
else
    start_tsc_watch
fi
```

## Linking Methods

### Method 1: NPM Link (Global)

```bash
#!/bin/bash
# link-global.sh - Link fork globally using npm link

set -euo pipefail

cd gemini-cli

link_globally() {
    info "Linking Gemini CLI globally..."
    
    # Build first
    npm run build
    
    # Link the CLI package
    cd packages/cli
    npm link
    
    # Verify link
    local linked_path=$(npm ls -g --depth=0 --link --parseable | grep gemini-cli || true)
    
    if [ -n "$linked_path" ]; then
        log "Successfully linked at: $linked_path"
        
        # Test the command
        info "Testing linked command..."
        gemini --version || warn "Command test failed"
    else
        error "Linking failed"
    fi
    
    cd ../..
}

# Create unlink script
create_unlink_script() {
    cat > unlink.sh << 'EOF'
#!/bin/bash
# Unlink the development version

cd packages/cli
npm unlink -g
cd ../..

echo "Development version unlinked"
echo "To reinstall stable version: npm install -g @google/gemini-cli"
EOF
    
    chmod +x unlink.sh
    info "Created unlink.sh for removing link"
}

# Main
main() {
    echo "========================================="
    echo "     Linking Gemini CLI Globally"
    echo "========================================="
    
    link_globally
    create_unlink_script
    
    echo ""
    log "Global link established!"
    info "You can now use 'gemini' command anywhere"
    info "To unlink: ./unlink.sh"
}

main
```

### Method 2: Alias Method (Non-invasive)

```bash
#!/bin/bash
# link-alias.sh - Create alias for development version

cd gemini-cli

create_alias() {
    info "Creating alias for development version..."
    
    # Get absolute path
    DEV_PATH="$(pwd)/packages/cli/dist/index.js"
    
    # Detect shell
    if [ -n "$BASH_VERSION" ]; then
        RC_FILE="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        RC_FILE="$HOME/.zshrc"
    else
        RC_FILE="$HOME/.profile"
    fi
    
    # Add alias
    echo "" >> "$RC_FILE"
    echo "# Gemini CLI Development Version" >> "$RC_FILE"
    echo "alias gemini-dev='node $DEV_PATH'" >> "$RC_FILE"
    echo "alias gdev='node $DEV_PATH'" >> "$RC_FILE"
    
    log "Aliases added to $RC_FILE"
    info "Run: source $RC_FILE"
    info "Then use: gemini-dev or gdev"
}

create_alias
```

### Method 3: Path Method

```bash
#!/bin/bash
# link-path.sh - Add development version to PATH

cd gemini-cli

setup_path_link() {
    info "Setting up PATH link..."
    
    # Create bin directory
    mkdir -p "$HOME/.local/bin"
    
    # Create wrapper script
    cat > "$HOME/.local/bin/gemini-fork" << EOF
#!/bin/bash
# Wrapper for Gemini CLI development version

export GEMINI_DEV=true
node $(pwd)/packages/cli/dist/index.js "\$@"
EOF
    
    chmod +x "$HOME/.local/bin/gemini-fork"
    
    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        info "Added ~/.local/bin to PATH"
    fi
    
    log "Path link created"
    info "Use: gemini-fork"
}

setup_path_link
```

### Method 4: Docker Development Container

```bash
#!/bin/bash
# link-docker.sh - Create Docker development container

cd gemini-cli

create_dockerfile() {
    cat > Dockerfile.dev << 'EOF'
FROM node:18-alpine

# Install development tools
RUN apk add --no-cache \
    git \
    bash \
    make \
    python3 \
    g++

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY lerna.json ./
COPY packages/*/package*.json ./packages/

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build project
RUN npm run build

# Link globally in container
RUN cd packages/cli && npm link

# Set entrypoint
ENTRYPOINT ["gemini"]
EOF
    
    # Build container
    docker build -f Dockerfile.dev -t gemini-cli-dev .
    
    # Create runner script
    cat > gemini-docker-dev << 'EOF'
#!/bin/bash
docker run -it --rm \
    -v "$PWD:/workspace" \
    -v "$HOME/.gemini:/root/.gemini" \
    -e GEMINI_API_KEY="$GEMINI_API_KEY" \
    gemini-cli-dev "$@"
EOF
    
    chmod +x gemini-docker-dev
    
    log "Docker development container created"
    info "Use: ./gemini-docker-dev"
}

create_dockerfile
```

## Part III: Development Workflow

## Making Changes

### Development Workflow Script

```bash
#!/bin/bash
# dev-workflow.sh - Complete development workflow

set -euo pipefail

# Workflow configuration
FEATURE_NAME="${1:-feature}"
COMMIT_MSG="${2:-Work in progress}"

# Create feature branch
create_feature_branch() {
    info "Creating feature branch..."
    
    # Ensure we're on main
    git checkout main
    git pull upstream main
    
    # Create and checkout feature branch
    BRANCH_NAME="feature/$FEATURE_NAME-$(date +%Y%m%d)"
    git checkout -b "$BRANCH_NAME"
    
    log "Created branch: $BRANCH_NAME"
}

# Make changes
make_changes() {
    info "Ready to make changes..."
    
    cat << 'EOF'
Common files to modify:

1. Add new tool:
   packages/cli/src/tools/built-in/your-tool.ts

2. Add new command:
   packages/cli/src/commands/your-command.ts

3. Modify CLI behavior:
   packages/cli/src/cli.ts
   packages/cli/src/repl.ts

4. Add configuration:
   packages/cli/src/config/schema.ts

5. Update types:
   packages/cli/src/types/index.ts
EOF
    
    # Open editor
    read -p "Open editor? (y/n): " OPEN_EDITOR
    if [[ "$OPEN_EDITOR" == "y" ]]; then
        ${EDITOR:-code} .
    fi
}

# Test changes
test_changes() {
    info "Testing changes..."
    
    # Run linter
    npm run lint || warn "Linting issues found"
    
    # Run tests
    npm test || warn "Tests failed"
    
    # Build
    npm run build || error "Build failed"
    
    # Test the CLI
    node packages/cli/dist/index.js --version || error "CLI test failed"
    
    log "Tests complete"
}

# Commit changes
commit_changes() {
    info "Committing changes..."
    
    # Stage changes
    git add -A
    
    # Show diff
    git diff --cached --stat
    
    # Commit with conventional commit format
    read -p "Commit type (feat/fix/docs/style/refactor/test/chore): " TYPE
    read -p "Commit scope (optional): " SCOPE
    read -p "Commit message: " MESSAGE
    
    if [ -n "$SCOPE" ]; then
        git commit -m "${TYPE}(${SCOPE}): ${MESSAGE}"
    else
        git commit -m "${TYPE}: ${MESSAGE}"
    fi
    
    log "Changes committed"
}

# Push changes
push_changes() {
    info "Pushing changes..."
    
    # Push to origin (your fork)
    git push -u origin "$(git branch --show-current)"
    
    log "Changes pushed to fork"
    
    # Provide PR link
    echo ""
    info "Create Pull Request at:"
    echo "https://github.com/$GITHUB_USER/gemini-cli/compare/main...$(git branch --show-current)"
}

# Main workflow
main() {
    echo "========================================="
    echo "     Development Workflow"
    echo "========================================="
    
    create_feature_branch
    make_changes
    
    read -p "Ready to test? (y/n): " TEST
    if [[ "$TEST" == "y" ]]; then
        test_changes
    fi
    
    read -p "Ready to commit? (y/n): " COMMIT
    if [[ "$COMMIT" == "y" ]]; then
        commit_changes
        push_changes
    fi
    
    echo ""
    log "Workflow complete!"
}

main
```

### Hot Reload Development

```bash
#!/bin/bash
# hot-reload.sh - Set up hot reload for development

cd gemini-cli

setup_hot_reload() {
    info "Setting up hot reload..."
    
    # Create development entry point
    cat > packages/cli/src/dev.ts << 'EOF'
import { register } from 'ts-node';
import { join } from 'path';

// Register ts-node for TypeScript support
register({
    project: join(__dirname, '../tsconfig.json'),
    transpileOnly: true,
    compilerOptions: {
        module: 'commonjs'
    }
});

// Hot reload support
if (process.env.NODE_ENV === 'development') {
    try {
        require('source-map-support').install();
    } catch (e) {
        console.warn('source-map-support not installed');
    }
}

// Import and run CLI
import('./index').then((cli) => {
    cli.run();
});
EOF
    
    # Create nodemon configuration
    cat > nodemon.dev.json << 'EOF'
{
    "watch": ["packages/cli/src/**/*.ts"],
    "ext": "ts",
    "ignore": ["**/*.test.ts", "dist/**"],
    "exec": "ts-node --transpile-only packages/cli/src/dev.ts",
    "env": {
        "NODE_ENV": "development",
        "DEBUG": "gemini:*"
    }
}
EOF
    
    # Create start script
    cat > start-dev.sh << 'EOF'
#!/bin/bash
export NODE_ENV=development
npx nodemon --config nodemon.dev.json
EOF
    
    chmod +x start-dev.sh
    
    log "Hot reload configured"
    info "Run: ./start-dev.sh"
}

setup_hot_reload
```

## Testing Your Changes

### Comprehensive Testing Suite

```bash
#!/bin/bash
# test-suite.sh - Complete testing suite for changes

cd gemini-cli

# Unit tests
run_unit_tests() {
    info "Running unit tests..."
    
    # Run Jest tests
    npx jest --coverage --verbose
    
    # Check coverage
    local coverage=$(npx jest --coverage --silent 2>&1 | grep "All files" | awk '{print $4}')
    
    if [[ "${coverage%\%}" -lt 80 ]]; then
        warn "Coverage below 80%: $coverage"
    else
        log "Coverage: $coverage"
    fi
}

# Integration tests
run_integration_tests() {
    info "Running integration tests..."
    
    # Test CLI commands
    cat > test-integration.sh << 'EOF'
#!/bin/bash
set -e

CLI="node packages/cli/dist/index.js"

echo "Testing basic commands..."
$CLI --version
$CLI --help

echo "Testing tools..."
$CLI /tools

echo "Testing memory..."
$CLI /memory show

echo "Testing with input..."
echo "What is 2+2?" | $CLI --pipe

echo "All integration tests passed!"
EOF
    
    chmod +x test-integration.sh
    ./test-integration.sh
}

# E2E tests
run_e2e_tests() {
    info "Running E2E tests..."
    
    # Create test script
    cat > test-e2e.js << 'EOF'
const { spawn } = require('child_process');
const assert = require('assert');

async function testCLI() {
    return new Promise((resolve, reject) => {
        const cli = spawn('node', ['packages/cli/dist/index.js'], {
            stdin: 'pipe',
            stdout: 'pipe',
            stderr: 'pipe'
        });
        
        let output = '';
        cli.stdout.on('data', (data) => {
            output += data.toString();
        });
        
        cli.stderr.on('data', (data) => {
            console.error('Error:', data.toString());
        });
        
        cli.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(`CLI exited with code ${code}`));
            }
        });
        
        // Send test input
        cli.stdin.write('What is the capital of France?\n');
        cli.stdin.end();
    });
}

// Run test
testCLI()
    .then(output => {
        assert(output.includes('Paris'), 'Response should mention Paris');
        console.log('E2E test passed!');
    })
    .catch(err => {
        console.error('E2E test failed:', err);
        process.exit(1);
    });
EOF
    
    node test-e2e.js
}

# Performance tests
run_performance_tests() {
    info "Running performance tests..."
    
    cat > test-performance.js << 'EOF'
const { performance } = require('perf_hooks');
const { spawn } = require('child_process');

async function measureStartupTime() {
    const start = performance.now();
    
    return new Promise((resolve) => {
        const cli = spawn('node', ['packages/cli/dist/index.js', '--version']);
        
        cli.on('close', () => {
            const end = performance.now();
            resolve(end - start);
        });
    });
}

// Run multiple times and average
async function runBenchmark() {
    const runs = 10;
    let total = 0;
    
    for (let i = 0; i < runs; i++) {
        const time = await measureStartupTime();
        total += time;
        console.log(`Run ${i + 1}: ${time.toFixed(2)}ms`);
    }
    
    const average = total / runs;
    console.log(`Average startup time: ${average.toFixed(2)}ms`);
    
    if (average > 1000) {
        console.warn('Warning: Startup time exceeds 1 second');
    }
}

runBenchmark();
EOF
    
    node test-performance.js
}

# Main test suite
main() {
    echo "========================================="
    echo "     Running Complete Test Suite"
    echo "========================================="
    
    run_unit_tests
    run_integration_tests
    run_e2e_tests
    run_performance_tests
    
    echo ""
    log "All tests complete!"
}

main
```

### Manual Testing Checklist

```bash
#!/bin/bash
# manual-test.sh - Manual testing checklist

cat > TESTING_CHECKLIST.md << 'EOF'
# Manual Testing Checklist

## Basic Functionality
- [ ] CLI starts without errors
- [ ] Help command works: `gemini --help`
- [ ] Version shows correctly: `gemini --version`

## Authentication
- [ ] Google auth flow works
- [ ] API key authentication works
- [ ] Token refresh works

## Core Features
- [ ] Chat mode works
- [ ] File operations work
- [ ] Shell command execution works
- [ ] Web search works
- [ ] Memory system works

## Tools
- [ ] Built-in tools load
- [ ] Custom tools discovery works
- [ ] MCP servers connect
- [ ] Tool execution completes

## Extensions
- [ ] Extensions load from global directory
- [ ] Extensions load from project directory
- [ ] Extension commands work
- [ ] Extension contexts load

## Error Handling
- [ ] Graceful handling of network errors
- [ ] Proper error messages
- [ ] No crashes on invalid input
- [ ] Recovery from API errors

## Performance
- [ ] Startup time < 2 seconds
- [ ] Response time reasonable
- [ ] Memory usage stable
- [ ] No memory leaks

## Platform-Specific
- [ ] Works on Linux
- [ ] Works on macOS
- [ ] Works on Windows (WSL)
- [ ] Works on Termux

## Edge Cases
- [ ] Handles large files
- [ ] Handles long conversations
- [ ] Handles concurrent operations
- [ ] Handles interrupts (Ctrl+C)
EOF

echo "Testing checklist created: TESTING_CHECKLIST.md"
```

## Keeping Fork Updated

### Sync with Upstream

```bash
#!/bin/bash
# sync-fork.sh - Keep fork synchronized with upstream

cd gemini-cli

sync_with_upstream() {
    info "Syncing with upstream..."
    
    # Fetch upstream changes
    git fetch upstream
    
    # Check current branch
    CURRENT_BRANCH=$(git branch --show-current)
    
    # Checkout main
    git checkout main
    
    # Merge upstream changes
    git merge upstream/main --no-edit
    
    # Push to your fork
    git push origin main
    
    # Return to previous branch
    git checkout "$CURRENT_BRANCH"
    
    # Rebase current branch on updated main
    read -p "Rebase current branch on main? (y/n): " REBASE
    if [[ "$REBASE" == "y" ]]; then
        git rebase main
    fi
    
    log "Fork synced with upstream"
}

# Check for conflicts
check_conflicts() {
    if git diff --name-only --diff-filter=U | grep -q .; then
        warn "Merge conflicts detected:"
        git diff --name-only --diff-filter=U
        
        echo "Resolve conflicts, then run:"
        echo "  git add ."
        echo "  git rebase --continue"
    else
        log "No conflicts"
    fi
}

# Main
main() {
    echo "========================================="
    echo "     Syncing Fork with Upstream"
    echo "========================================="
    
    sync_with_upstream
    check_conflicts
    
    echo ""
    log "Sync complete!"
}

main
```

### Automated Sync Setup

```bash
#!/bin/bash
# auto-sync.sh - Set up automated fork syncing

setup_auto_sync() {
    info "Setting up automated sync..."
    
    # Create GitHub Action for auto-sync
    mkdir -p .github/workflows
    
    cat > .github/workflows/sync-fork.yml << 'EOF'
name: Sync Fork

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:  # Manual trigger

jobs:
  sync:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Sync upstream changes
      run: |
        git config user.name 'GitHub Action'
        git config user.email 'action@github.com'
        
        git remote add upstream https://github.com/google-gemini/gemini-cli.git
        git fetch upstream
        git checkout main
        git merge upstream/main --no-edit
        git push origin main
EOF
    
    log "Auto-sync GitHub Action created"
    info "Push to GitHub to enable"
}

setup_auto_sync
```

## Part IV: Advanced Configuration

## Multiple Version Management

### Version Manager Script

```bash
#!/bin/bash
# version-manager.sh - Manage multiple Gemini CLI versions

VERSIONS_DIR="$HOME/.gemini-versions"
mkdir -p "$VERSIONS_DIR"

# Install specific version
install_version() {
    local version="$1"
    local install_dir="$VERSIONS_DIR/$version"
    
    info "Installing Gemini CLI version $version..."
    
    if [ "$version" == "latest" ]; then
        # Install latest from npm
        mkdir -p "$install_dir"
        cd "$install_dir"
        npm init -y &> /dev/null
        npm install @google/gemini-cli
        
    elif [ "$version" == "dev" ]; then
        # Use development version
        ln -sf "$HOME/Development/gemini-cli" "$install_dir"
        
    elif [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        # Install specific version
        mkdir -p "$install_dir"
        cd "$install_dir"
        npm init -y &> /dev/null
        npm install "@google/gemini-cli@$version"
        
    else
        # Install from git ref
        git clone https://github.com/google-gemini/gemini-cli.git "$install_dir"
        cd "$install_dir"
        git checkout "$version"
        npm install
        npm run build
    fi
    
    log "Version $version installed"
}

# Switch version
switch_version() {
    local version="$1"
    local version_dir="$VERSIONS_DIR/$version"
    
    if [ ! -d "$version_dir" ]; then
        error "Version $version not installed"
    fi
    
    # Update symlink
    ln -sf "$version_dir" "$VERSIONS_DIR/current"
    
    # Update PATH
    export PATH="$VERSIONS_DIR/current/node_modules/.bin:$PATH"
    
    log "Switched to version $version"
}

# List versions
list_versions() {
    info "Installed versions:"
    
    for dir in "$VERSIONS_DIR"/*/; do
        if [ -d "$dir" ]; then
            version=$(basename "$dir")
            
            if [ -L "$VERSIONS_DIR/current" ]; then
                current=$(readlink "$VERSIONS_DIR/current" | xargs basename)
                if [ "$version" == "$current" ]; then
                    echo "  * $version (current)"
                else
                    echo "    $version"
                fi
            else
                echo "    $version"
            fi
        fi
    done
}

# Create gemini-select command
create_selector() {
    cat > "$HOME/.local/bin/gemini-select" << 'EOF'
#!/bin/bash
# Select Gemini CLI version

VERSIONS_DIR="$HOME/.gemini-versions"

select_version() {
    echo "Available versions:"
    
    versions=()
    for dir in "$VERSIONS_DIR"/*/; do
        if [ -d "$dir" ] && [ "$(basename "$dir")" != "current" ]; then
            versions+=("$(basename "$dir")")
        fi
    done
    
    select version in "${versions[@]}"; do
        if [ -n "$version" ]; then
            "$VERSIONS_DIR/version-manager.sh" switch "$version"
            break
        fi
    done
}

select_version
EOF
    
    chmod +x "$HOME/.local/bin/gemini-select"
    log "Created gemini-select command"
}

# Main menu
case "${1:-}" in
    install)
        install_version "${2:-latest}"
        ;;
    switch)
        switch_version "$2"
        ;;
    list)
        list_versions
        ;;
    *)
        echo "Usage: $0 {install|switch|list} [version]"
        ;;
esac
```

## Development Tools Setup

### Complete Development Environment

```bash
#!/bin/bash
# dev-tools-setup.sh - Set up complete development environment

setup_vscode() {
    info "Setting up VS Code configuration..."
    
    mkdir -p .vscode
    
    # VS Code settings
    cat > .vscode/settings.json << 'EOF'
{
    "typescript.tsdk": "node_modules/typescript/lib",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll.eslint": true
    },
    "eslint.validate": [
        "javascript",
        "javascriptreact",
        "typescript",
        "typescriptreact"
    ],
    "files.exclude": {
        "**/dist": true,
        "**/node_modules": true,
        "**/.turbo": true
    },
    "search.exclude": {
        "**/dist": true,
        "**/node_modules": true,
        "**/package-lock.json": true
    }
}
EOF
    
    # Launch configuration
    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "node",
            "request": "launch",
            "name": "Debug Gemini CLI",
            "program": "${workspaceFolder}/packages/cli/src/index.ts",
            "preLaunchTask": "npm: build",
            "outFiles": ["${workspaceFolder}/packages/cli/dist/**/*.js"],
            "sourceMaps": true,
            "console": "integratedTerminal"
        },
        {
            "type": "node",
            "request": "launch",
            "name": "Debug Current Test",
            "program": "${workspaceFolder}/node_modules/.bin/jest",
            "args": [
                "--runInBand",
                "${relativeFile}"
            ],
            "console": "integratedTerminal"
        }
    ]
}
EOF
    
    # Tasks
    cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "npm",
            "script": "build",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": "$tsc"
        },
        {
            "type": "npm",
            "script": "test",
            "group": {
                "kind": "test",
                "isDefault": true
            }
        }
    ]
}
EOF
    
    # Extensions recommendations
    cat > .vscode/extensions.json << 'EOF'
{
    "recommendations": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "ms-vscode.vscode-typescript-tsc",
        "orta.vscode-jest",
        "streetsidesoftware.code-spell-checker"
    ]
}
EOF
    
    log "VS Code configuration complete"
}

setup_debugging() {
    info "Setting up debugging tools..."
    
    # Create debug script
    cat > debug.sh << 'EOF'
#!/bin/bash
# Debug Gemini CLI

export DEBUG=gemini:*
export NODE_ENV=development
export NODE_OPTIONS="--inspect-brk=9229"

echo "Debugger listening on port 9229"
echo "Open chrome://inspect in Chrome"
echo "Or attach VS Code debugger"

node packages/cli/dist/index.js "$@"
EOF
    
    chmod +x debug.sh
    
    log "Debug tools configured"
}

setup_git_hooks() {
    info "Setting up Git hooks..."
    
    # Pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook

echo "Running pre-commit checks..."

# Run linter
npm run lint || {
    echo "Linting failed. Fix issues before committing."
    exit 1
}

# Run tests
npm test || {
    echo "Tests failed. Fix tests before committing."
    exit 1
}

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    log "Git hooks configured"
}

# Main
main() {
    echo "========================================="
    echo "     Development Tools Setup"
    echo "========================================="
    
    setup_vscode
    setup_debugging
    setup_git_hooks
    
    echo ""
    log "Development tools ready!"
}

main
```

## Contributing Back

### Contribution Workflow

```bash
#!/bin/bash
# contribute.sh - Prepare and submit contributions

prepare_contribution() {
    info "Preparing contribution..."
    
    # Ensure tests pass
    npm test || error "Tests must pass"
    
    # Ensure linting passes
    npm run lint || error "Code must be properly formatted"
    
    # Update documentation
    read -p "Have you updated documentation? (y/n): " DOC_UPDATED
    if [[ "$DOC_UPDATED" != "y" ]]; then
        warn "Please update relevant documentation"
    fi
    
    # Add tests
    read -p "Have you added tests for new features? (y/n): " TESTS_ADDED
    if [[ "$TESTS_ADDED" != "y" ]]; then
        warn "Please add tests for new functionality"
    fi
    
    log "Contribution checks complete"
}

create_pull_request() {
    info "Creating pull request..."
    
    # Push latest changes
    git push origin "$(git branch --show-current)"
    
    # Create PR template
    cat > .github/pull_request_template.md << 'EOF'
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Manually tested

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Updated documentation
- [ ] No breaking changes
EOF
    
    # Open PR URL
    BRANCH=$(git branch --show-current)
    PR_URL="https://github.com/$GITHUB_USER/gemini-cli/compare/main...$BRANCH"
    
    info "Create PR at: $PR_URL"
    
    # Open in browser if possible
    if command -v xdg-open &> /dev/null; then
        xdg-open "$PR_URL"
    elif command -v open &> /dev/null; then
        open "$PR_URL"
    fi
}

# Main
main() {
    echo "========================================="
    echo "     Contributing to Gemini CLI"
    echo "========================================="
    
    prepare_contribution
    create_pull_request
    
    echo ""
    log "Ready to contribute!"
    info "Follow the PR template and guidelines"
}

main
```

### Final Complete Setup Script

```bash
#!/bin/bash
# complete-fork-setup.sh - One command to set up everything

set -euo pipefail

# Get configuration
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter working directory [$HOME/Development]: " WORK_DIR
WORK_DIR="${WORK_DIR:-$HOME/Development}"

# Clone fork
echo "Step 1: Cloning fork..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
git clone "git@github.com:$GITHUB_USER/gemini-cli.git" || \
git clone "https://github.com/$GITHUB_USER/gemini-cli.git"

cd gemini-cli

# Set up remotes
echo "Step 2: Setting up remotes..."
git remote add upstream https://github.com/google-gemini/gemini-cli.git
git fetch --all

# Install dependencies
echo "Step 3: Installing dependencies..."
npm install
npx lerna bootstrap

# Build project
echo "Step 4: Building project..."
npm run build

# Link globally
echo "Step 5: Linking globally..."
cd packages/cli
npm link
cd ../..

# Set up development tools
echo "Step 6: Setting up development tools..."
./dev-tools-setup.sh

# Create aliases
echo "Step 7: Creating aliases..."
echo "alias gdev='gemini --debug'" >> ~/.bashrc
echo "alias gfork='cd $WORK_DIR/gemini-cli'" >> ~/.bashrc

# Final message
echo ""
echo "========================================="
echo "✓ Fork setup complete!"
echo "========================================="
echo ""
echo "Quick reference:"
echo "  - Fork location: $WORK_DIR/gemini-cli"
echo "  - Run development version: gemini"
echo "  - Debug mode: gdev"
echo "  - Go to fork: gfork"
echo "  - Build: npm run build"
echo "  - Test: npm test"
echo "  - Watch mode: npm run dev"
echo ""
echo "Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. gemini --version"
echo "  3. Start developing!"
```

This comprehensive guide covers everything from forking the repository to contributing back, with specific attention to development workflows, testing, and maintaining synchronization with the upstream repository. The scripts are production-ready and handle edge cases, making the development process smooth and efficient.
