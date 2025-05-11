#!/data/data/com.termux/files/usr/bin/bash
# Script to create a default ruff.toml in the current directory for Termux users

FILENAME="ruff.toml"

# Check if ruff.toml already exists
if [ -f "$FILENAME" ]; then
  echo ""
  echo "⚠️  $FILENAME already exists in the current directory."
  read -p "Do you want to overwrite it? (y/N): " choice
  echo "" # Newline for cleaner output
  case "$choice" in
    y|Y )
      echo "Overwriting $FILENAME..."
      ;;
    * )
      echo "Aborting. $FILENAME was not changed."
      exit 0
      ;;
  esac
fi

echo "Creating a default $FILENAME..."

# Use cat with a HEREDOC to write the config
cat > "$FILENAME" <<EOF
# ruff.toml - Default configuration for Ruff
# Generated for general Python projects.
# Customize as needed for your project's specific requirements.

# Specify the target Python version. Ruff uses this for version-specific rules and fixes.
# Examples: "py38", "py39", "py310", "py311", "py312", "py313"
# Check Ruff's documentation for the latest supported versions.
target-version = "py310"

# Set the line length.
line-length = 88
# indent-width = 4 (This is Ruff's default, usually no need to set)

# === Linting Configuration ===
[lint]
# Select the rules to enable.
# For a full list of rules, see: https://docs.astral.sh/ruff/rules/
#
# E: pycodestyle errors
# W: pycodestyle warnings
# F: Pyflakes (syntax errors, undefined names, etc.)
# I: isort (import sorting consistency)
# B: flake8-bugbear (finds potential bugs and design problems)
# C90: McCabe complexity (checks for overly complex code)
# UP: pyupgrade (helps upgrade syntax to newer Python versions)
# N: pep8-naming (naming convention checks)
# ANN: flake8-annotations (type annotation checks; can be strict, enable if ready)
# S: flake8-bandit (security checks; consider adding if applicable)
# RUF: Ruff-specific rules (often for performance or unique checks)
# PL: Pylint rules (Ruff implements a subset of Pylint rules)
select = [
    "E",  # pycodestyle errors
    "F",  # Pyflakes
    "W",  # pycodestyle warnings (Ruff's default rule set E,F doesn't include W)
    "I",  # isort
    "B",  # flake8-bugbear
    "C90",# McCabe complexity
    "UP", # pyupgrade
    "N",  # pep8-naming
    # "ANN", # flake8-annotations (Uncomment if your project heavily uses type hints and you want to enforce them)
    # "S",   # flake8-bandit (Uncomment for security linting; might need `pip install ruff[security]`)
    "RUF", # Ruff-specific rules
    # "PL",  # Consider adding specific Pylint rules you find valuable
]

# Ignore specific rules globally if they are too noisy for your project.
# Example: ignore = ["E501"] # To ignore 'line too long' if handled by formatter or intentionally exceeded.
# ignore = []

# Exclude files and directories from linting.
# Ruff has some built-in defaults (like .git, .venv, etc.).
# This list extends those defaults or makes them explicit.
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".git-rewrite",
    ".hg",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".pyenv",
    ".pytest_cache",
    ".pytype",
    ".ruff_cache", # Ruff's own cache directory
    ".svn",
    ".tox",
    ".venv",        # Common virtual environment folder names
    "venv",
    "env",
    ".env",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "*/migrations/*",   # Example: Django/Flask migrations
    "*/static/*",       # Example: Collected static files
    "*/media/*",        # Example: User uploaded media
    "*/vendor/*",       # Example: Vendored code / third-party libraries included directly
    "tests/fixtures/*", # Example: Test fixture files that might have unconventional code
    "docs/conf.py",     # Example: Sphinx configuration file
    "setup.py",         # Often has less strict linting needs
]

# Per-file ignores allow disabling specific rules for certain files or patterns.
# [lint.per-file-ignores]
# "__init__.py" = ["F401", "E402"] # Ignore 'unused import' and 'module level import not at top of file' in __init__.py files
# "tests/**/*.py" = [
#     "S101",  # Ignore 'assert' statements used in tests (flake8-bandit)
#     "ANN001",# Ignore missing type hint for function arguments that are not type annotated (e.g. self, cls)
#     "ANN201",# Ignore missing return type hint for public functions
#     "PLR2004", # Pylint: Magic value used in comparison, not a good fit for tests assertions.
# ]

# Some rules can be automatically fixed by Ruff.
# By default, all fixable rules are eligible. You can be more specific if needed.
# fixable = ["ALL"] # (default)
# unfixable = []    # (default)

# If you use type checking (e.g., Mypy) and import type checking related symbols:
# typing-imports = true (default)

# === Formatting Configuration ===
# This section configures `ruff format`. If you use another formatter like Black,
# you might skip this section or ensure settings are compatible.
[format]
# Style of quotes for strings ('single' or 'double').
# Black defaults to double quotes.
quote-style = "double"

# Style of indentation ('space' or 'tab').
indent-style = "space" # "tab" is also an option

# Add a trailing comma to multi-line expressions, parameters, and type parameters.
# Options: "all", "es5" (JavaScript-like), "none".
# "all" is generally good for cleaner version control diffs.
trailing-comma = "all"

# Whether to skip string normalization (e.g., convert all strings to use the 'quote-style').
# Default is false, meaning strings will be normalized.
# skip-string-normalization = false

# Line endings ('lf', 'crlf', 'native').
# 'lf' is standard for Linux/macOS (and Termux). 'crlf' for Windows.
# 'native' uses the OS default.
line-ending = "lf"

# Enable preview mode for the formatter.
# This enables newer, possibly less stable, formatting changes.
# preview = false (default)
EOF

echo ""
echo "✅ $FILENAME has been created in the current directory."
echo ""
echo "Next steps:"
echo "1. Review $FILENAME and customize it further for your project's needs."
echo "2. Install Ruff in your Termux environment if you haven't already:"
echo "   pip install ruff"
echo "3. Run Ruff to check your code (from your repo root):"
echo "   ruff check ."
echo "4. Run Ruff to format your code (from your repo root):"
echo "   ruff format ."
echo "5. To automatically fix linting issues (where possible):"
echo "   ruff check --fix ."
echo ""
echo "Happy linting! ✨"

exit 0
