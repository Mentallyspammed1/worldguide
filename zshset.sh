#!/bin/bash

# Setup script for Oh My Zsh, Powerlevel10k, aichat, and plugins
# Last Updated: 2025-04-03

echo "Starting ZSH environment setup..."

# Install Zsh if not present
if ! command -v zsh &>/dev/null; then
  echo "Installing Zsh..."
  if [[ -f /etc/debian_version ]]; then
    sudo apt update && sudo apt install -y zsh
  elif [[ -f /etc/redhat-release ]]; then
    sudo dnf install -y zsh
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install zsh
  elif [[ -n "$TERMUX_VERSION" ]]; then
    pkg install zsh
  else
    echo "Unsupported system. Please install Zsh manually."
    exit 1
  fi
fi

# Install Oh My Zsh
if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
  echo "Installing Oh My Zsh..."
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

# Install Powerlevel10k
THEME_DIR="$HOME/.oh-my-zsh/custom/themes/powerlevel10k"
if [[ ! -d "$THEME_DIR" ]]; then
  echo "Installing Powerlevel10k..."
  git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$THEME_DIR"
fi

# Install Rust (for aichat)
if ! command -v cargo &>/dev/null; then
  echo "Installing Rust (required for aichat)..."
  if [[ -n "$TERMUX_VERSION" ]]; then
    pkg install rust
  else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
  fi
fi

# Install aichat
if ! command -v aichat &>/dev/null; then
  echo "Installing aichat..."
  cargo install aichat
fi

# Install Plugins
ZSH_CUSTOM="${ZSH:-$HOME/.oh-my-zsh}/custom/plugins"
mkdir -p "$ZSH_CUSTOM"

declare -A plugins=(
  ["zsh-autosuggestions"]="https://github.com/zsh-users/zsh-autosuggestions"
  ["zsh-completions"]="https://github.com/zsh-users/zsh-completions"
  ["fzf-tab"]="https://github.com/Aloxaf/fzf-tab"
  ["zsh-autopair"]="https://github.com/hlissner/zsh-autopair"
  ["alias-tips"]="https://github.com/djui/alias-tips"
  ["fast-syntax-highlighting"]="https://github.com/zdharma-continuum/fast-syntax-highlighting"
  ["aichat"]="https://github.com/sigoden/aichat"
)

for plugin in "${!plugins[@]}"; do
  if [[ ! -d "$ZSH_CUSTOM/$plugin" ]]; then
    echo "Installing $plugin..."
    git clone --depth=1 "${plugins[$plugin]}" "$ZSH_CUSTOM/$plugin"
  fi
done

# Install optional tools (bat, eza, fd)
if [[ -n "$TERMUX_VERSION" ]]; then
  pkg install -y bat eza fd
elif command -v apt &>/dev/null; then
  sudo apt install -y bat eza fd-find
  ln -sf /usr/bin/batcat "$HOME/bin/bat"  # Ubuntu workaround
fi

# Set Zsh as default shell
if [[ "$SHELL" != "$(which zsh)" ]]; then
  echo "Setting Zsh as default shell..."
  chsh -s "$(which zsh)"
fi

# Backup existing .zshrc and deploy new one
if [[ -f "$HOME/.zshrc" ]]; then
  mv "$HOME/.zshrc" "$HOME/.zshrc.bak.$(date +%Y%m%d)"
fi
echo "Deploying new .zshrc..."
curl -fsSL "https://gist.githubusercontent.com/<your-gist-id>/raw/zshrc" > "$HOME/.zshrc"  # Replace with your gist URL or local path

echo "Setup complete! Start Zsh with 'zsh' or restart your terminal."
