#!/bin/bash

# Define the base directory for Termux
BASE_DIR="/data/data/com.termux/files/home"
CONFIG_DIR="$BASE_DIR/nano-config"

# Create the configuration directory
echo "Creating directory $CONFIG_DIR..."
mkdir -p "$CONFIG_DIR"
if [ $? -ne 0 ]; then
    echo "Failed to create $CONFIG_DIR. Check permissions."
    exit 1
fi

# Create .nanorc in home directory
echo "Creating .nanorc..."
touch "$BASE_DIR/.nanorc"
cat << 'EOF' > "$BASE_DIR/.nanorc"
# General settings
set mouse
set tabsize 4
set smooth
set linenumbers

# Include syntax highlighting files from nano-config directory
include "/data/data/com.termux/files/home/nano-config/nanorc.nano"
include "/data/data/com.termux/files/home/nano-config/python.nano"
include "/data/data/com.termux/files/home/nano-config/javascript.nano"
include "/data/data/com.termux/files/home/nano-config/json.nano"
EOF

# Create nanorc.nano for .nanorc and .file files
echo "Creating nanorc.nano..."
touch "$CONFIG_DIR/nanorc.nano"
cat << 'EOF' > "$CONFIG_DIR/nanorc.nano"
syntax "nanorc" "\.nanorc$" "\.file$"
color 9 "\b(set|color|include|unset|bind|key|meta|function|help|goto|whereis|execute|exit|writeout|save|discard|undo|redo|forward|backward|beginning|end|cut|paste|copy|trim|mark|goto|find|replace|spell|help|version|status|toggle|options|open|close|choose|select|print|suspend|interrupt|convert|transform|complete|expand|shrink|justify|center|fill|unfill|spell|wrap|unwrap|case|tolower|toupper|swapcase|comment|uncomment|indent|unindent|align|reformat|execute|shell|login|logout|history|timestamp)\b"
color 9 "\"[^\"]*\""
color 2 "^#.*$"
EOF

# Create python.nano for Python files
echo "Creating python.nano..."
touch "$CONFIG_DIR/python.nano"
cat << 'EOF' > "$CONFIG_DIR/python.nano"
syntax "python" "\.py$"
color 2 "^( |\t)*#"
color 9 "\"[^\"]*\""
color 9 "\b(and|as|assert|break|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|not|or|pass|print|raise|return|try|while|with|yield)\b"
color 3 "\b(True|False|None|NotImplemented|Ellipsis)\b"
color 11 "\b[0-9]+(\.[0-9]+)?\b"
color 5 "\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\("
EOF

# Create javascript.nano for JavaScript files
echo "Creating javascript.nano..."
touch "$CONFIG_DIR/javascript.nano"
cat << 'EOF' > "$CONFIG_DIR/javascript.nano"
syntax "javascript" "\.js$"
color 2 "//.*"
color 2 "/\*.*?\*/"
color 9 "\"[^\"]*\""
color 9 "\b(break|case|catch|continue|default|delete|do|else|finally|for|function|if|in|instanceof|new|return|switch|this|throw|try|typeof|var|void|while|with)\b"
color 5 "\b(true|false|null|undefined)\b"
color 11 "\b[0-9]+(\.[0-9]+)?\b"
color 5 "\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)\("
EOF

# Create json.nano for JSON files
echo "Creating json.nano..."
touch "$CONFIG_DIR/json.nano"
cat << 'EOF' > "$CONFIG_DIR/json.nano"
syntax "json" "\.json$"
color 9 "\"[^\"]*\""
color 11 "\b[0-9]+(\.[0-9]+)?\b"
color 9 "\b(true|false|null)\b"
EOF

echo "Setup complete! Use 'nano filename' to edit files with custom syntax highlighting."

#!/bin/bash

# Define the base directory for Termux
BASE_DIR="/data/data/com.termux/files/home"

# Create .nanorc in home directory
echo "Creating .nanorc..."
touch "$BASE_DIR/.nanorc"
cat << 'EOF' > "$BASE_DIR/.nanorc"
# Basic usability enhancements
set linenumbers
set mouse
set autoindent
set tabsize 4
set tabstospaces
set backup
set softwrap
set historylog

# Python-specific syntax highlighting
syntax "python" "\.py$"
color 9 "\b(and|as|assert|break|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|not|or|pass|print|raise|return|try|while|with|yield)\b"
color 9 "\"[^\"]*\""
color 2 "^( |\t)*#.*$"
color 3 "\b(True|False|None|NotImplemented|Ellipsis)\b"
color 11 "\b[0-9]+(\.[0-9]+)?\b"
color 5 "\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\("
color 10 "\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)\("
color 1 "^[ ]*import\s+"

# Keybindings for Python development tools
bind F9 execute "black %f" main
bind F10 execute "pylint %f" main
bind F11 execute "python3 -m py_compile %f" main
EOF

# Instructions for installing required tools
echo "Installing Black and Pylint (run these if not already installed)..."
echo "pkg install python"  # Ensure Python is installed in Termux
echo "pip install black pylint"

echo "Setup complete! Follow these steps to test:"
echo "1. Run 'pkg install python' if Python isn’t installed."
echo "2. Run 'pip install black pylint' to install formatting and linting tools."
echo "3. Create a test Python file (e.g., 'echo \"def hello(): print('Hi')\" > test.py')."
echo "4. Open it with 'nano test.py' and test keybindings:"
echo "   - F9: Format with Black"
echo "   - F10: Lint with Pylint"
echo "   - F11: Compile with Python3"
