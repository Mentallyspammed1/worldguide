#!/data/data/com.termux/files/usr/bin/bash

# setup_zsh_wizard.sh - Setup script for Zsh Wizard Edition in Termux
# Run this in Termux to install and configure the enhanced Zsh environment.

# Exit on error
set -e

# Colors for output
RED='\e[31m'
GREEN='\e[32m'
BLUE='\e[36m'
YELLOW='\e[33m'
RESET='\e[0m'

echo -e "${BLUE}🧙‍♂️ Starting Zsh Wizard Edition setup for Termux... ✨${RESET}"

# Step 1: Update Termux and install core packages
echo -e "${YELLOW}Updating Termux and installing core dependencies...${RESET}"
pkg update -y && pkg upgrade -y
pkg install -y zsh git curl termux-api neovim nano tree fd bat eza gnupg openssh net-tools procps iputils traceroute whois bind-tools

# Step 2: Set up XDG directories
echo -e "${YELLOW}Configuring XDG base directories...${RESET}"
XDG_CONFIG_HOME="${HOME}/.config"
XDG_CACHE_HOME="${HOME}/.cache"
XDG_DATA_HOME="${HOME}/.local/share"
XDG_STATE_HOME="${HOME}/.local/state"
ZDOTDIR="${XDG_CONFIG_HOME}/zsh"

# Create directories if they don't exist
mkdir -p "$ZDOTDIR" "$XDG_CACHE_HOME/zsh" "$XDG_DATA_HOME/oh-my-zsh/custom/themes" "$XDG_STATE_HOME/zsh"

# Step 3: Set Zsh as the default shell
echo -e "${YELLOW}Setting Zsh as the default shell...${RESET}"
chsh -s zsh || echo -e "${RED}Failed to set Zsh as default shell. Run 'chsh -s zsh' manually if needed.${RESET}"

# Step 4: Install Powerlevel10k theme
echo -e "${YELLOW}Installing Powerlevel10k theme...${RESET}"
THEME_DIR="${XDG_DATA_HOME}/oh-my-zsh/custom/themes/powerlevel10k"
if [[ ! -d "$THEME_DIR" ]]; then
    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$THEME_DIR"
    echo -e "${GREEN}Powerlevel10k installed successfully.${RESET}"
else
    echo -e "${GREEN}Powerlevel10k already installed, skipping...${RESET}"
fi

# Step 5: Install Rust and aichat (optional AI features)
echo -e "${YELLOW}Installing Rust and aichat for AI-powered features (this may take a while)...${RESET}"
pkg install -y rust
cargo install aichat || echo -e "${RED}Failed to install aichat. AI features will be limited.${RESET}"

# Step 6: Backup existing .zshrc if it exists
if [[ -f "$HOME/.zshrc" ]]; then
    echo -e "${YELLOW}Backing up existing ~/.zshrc to ~/.zshrc.bak...${RESET}"
    mv "$HOME/.zshrc" "$HOME/.zshrc.bak"
fi

# Step 7: Move .zshrc to XDG-compliant location
echo -e "${YELLOW}Copying the Wizard Edition .zshrc to $ZDOTDIR/.zshrc...${RESET}"
# Assuming the .zshrc content provided is saved as 'zshrc_wizard' in the current directory
if [[ -f "zshrc_wizard" ]]; then
    cp "zshrc_wizard" "$ZDOTDIR/.zshrc"
else
    echo -e "${RED}Error: 'zshrc_wizard' file not found in current directory. Please provide the .zshrc content as 'zshrc_wizard'.${RESET}"
    exit 1
fi

# Step 8: Configure Termux to load .zshrc from ZDOTDIR
echo -e "${YELLOW}Configuring Termux to use $ZDOTDIR/.zshrc...${RESET}"
echo "source $ZDOTDIR/.zshrc" > "$HOME/.zshrc"

# Step 9: Set up storage access (optional)
echo -e "${YELLOW}Setting up Termux storage access (optional)...${RESET}"
termux-setup-storage || echo -e "${RED}Failed to set up storage access. Run 'termux-setup-storage' manually if needed.${RESET}"

# Step 10: Finalize and reload
echo -e "${YELLOW}Reloading Termux settings and starting Zsh...${RESET}"
termux-reload-settings

echo -e "${GREEN}🧙‍♂️ Zsh Wizard Edition setup complete! ✨${RESET}"
echo -e "${BLUE}Starting Zsh now... Run 'p10k configure' to customize your prompt.${RESET}"

# Launch Zsh
exec zsh
