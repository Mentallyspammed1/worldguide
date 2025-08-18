#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Neon Color Scheme for Enhanced Output (Termux-friendly) ---
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
WHITE='\033[1;37m'
ORANGE='\033[38;5;208m'
NC='\033[0m'

# --- Global Variables ---
REQUIRED_NODE_VERSION=18
REQUIRED_NPM_VERSION=8

# --- Logging Functions with Neon Colors ---
log_info() {
  echo -e "${CYAN}[INFO] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_success() {
  echo -e "${GREEN}[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_warning() {
  echo -e "${YELLOW}[WARNING] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}"
}

log_error() {
  echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S')${NC} ${1}" >&2
  exit 1
}

# --- Function to check if a command exists ---
command_exists() {
  type "$1" &> /dev/null
}

# --- Main Setup Functions ---

setup_environment() {
  log_info "Initializing Gbotx Bybit trading bot environment setup. Current time in St. Louis: $(TZ='America/Chicago' date '+%I:%M %p %Z, %B %d, %Y')."

  check_dependencies
  validate_node_npm
  clean_npm_cache_prune
  install_dependencies
  configure_vitest
  setup_backtest_data
  setup_logger
  run_tests_and_build
  debug_setup
  log_success "Environment setup complete! Your bot is ready for advanced backtesting! 😻"
}

check_dependencies() {
  log_info "Verifying essential system tools..."

  if ! command_exists node; then
    log_error "Node.js is not installed. Please install it to proceed."
  fi

  if ! command_exists npm; then
    log_error "npm is not installed. Please install it (usually with Node.js)."
  fi

  if ! command_exists tsc; then
    log_warning "TypeScript compiler (tsc) not found. Installing globally..."
    npm install -g typescript || log_error "Failed to install typescript."
  fi

  if ! command_exists vite; then
    log_warning "Vite not found. Installing as dev dependency..."
    npm install --save-dev vite || log_error "Failed to install Vite."
  fi

  if ! command_exists vitest; then
    log_warning "Vitest not found. Installing as dev dependency..."
    npm install --save-dev vitest || log_error "Failed to install Vitest."
  fi

  log_success "Essential tools checked."
}

validate_node_npm() {
  log_info "Validating Node.js and npm versions..."

  NODE_FULL_VERSION=$(node -v 2>/dev/null)
  NODE_MAJOR_VERSION=$(echo "$NODE_FULL_VERSION" | cut -d'v' -f2 | cut -d'.' -f1)
  NPM_FULL_VERSION=$(npm -v 2>/dev/null)
  NPM_MAJOR_VERSION=$(echo "$NPM_FULL_VERSION" | cut -d'.' -f1)

  if [ -z "$NODE_MAJOR_VERSION" ]; then
    log_error "Node.js version could not be determined."
  fi

  if [ "$NODE_MAJOR_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
    log_error "Node.js v${RED}$NODE_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  if [ -z "$NPM_MAJOR_VERSION" ]; then
    log_error "npm version could not be determined."
  fi

  if [ "$NPM_MAJOR_VERSION" -lt "$REQUIRED_NPM_VERSION" ]; then
    log_error "npm v${RED}$NPM_FULL_VERSION${NC} is too old. Requires >= v${REQUIRED_NODE_VERSION}."
  fi

  log_success "Node.js ${GREEN}$NODE_FULL_VERSION${NC} and npm ${GREEN}$NPM_FULL_VERSION${NC} validated."
}

clean_npm_cache_prune() {
  log_info "Clearing npm cache and pruning dependencies..."
  npm cache clean --force 2>/dev/null || log_warning "Failed to clear npm cache."
  npm prune 2>/dev/null || log_warning "Failed to prune dependencies."
  log_success "npm cache cleared and dependencies pruned."
}

install_dependencies() {
  log_info "Installing project dependencies..."
  if ! npm install; then
    log_error "npm install failed. Check package.json or network."
  fi
  log_success "Dependencies installed."
}

configure_vitest() {
  log_info "Configuring Vitest..."
  if [ ! -f vitest.config.ts ]; then
    log_info "Creating vitest.config.ts..."
    echo "/// <reference types=\"vitest\" />\nimport { defineConfig } from 'vite';\n\nexport default defineConfig({\n  test: {\n    include: ['src/**/*.{test,spec}.{js,ts,jsx,tsx}'],\n    exclude: ['node_modules', 'dist', '**/*.d.ts'],\n    environment: 'node',\n    tsconfig: 'tsconfig.json',\n    coverage: { reporter: ['text'], include: ['src/**/*.{ts,tsx}'] },\n    setupFiles: './test/setup.ts',\n    outputFile: './test.log'\n  }\n});" > vitest.config.ts
    log_success "vitest.config.ts created. Customize as needed."
  fi
  if [ ! -d test ]; then
    mkdir test
    log_info "Created test directory."
  fi
  if [ ! -f test/setup.ts ]; then
    echo "import 'dotenv/config';" > test/setup.ts
    log_success "Created test/setup.ts to load .env."
  fi
}

setup_backtest_data() {
  log_info "Setting up backtest data..."
  if [ ! -d data ]; then
    mkdir data
    log_info "Created data directory."
  fi
  if [ ! -f data/historical_prices.json ]; then
    log_info "Creating sample historical_prices.json..."
    echo '[\n      {"timestamp": "2025-07-20T08:00:00Z", "price": 50000},\n      {"timestamp": "2025-07-20T08:00:01Z", "price": 50010},\n      {"timestamp": "2025-07-20T08:00:02Z", "price": 50020},\n      {"timestamp": "2025-07-20T08:00:03Z", "price": 50015},\n      {"timestamp": "2025-07-20T08:00:04Z", "price": 50025}\n    ]' > data/historical_prices.json
    log_success "Sample historical_prices.json created. Replace with real data as needed."
  fi
}

setup_logger() {
  log_info "Setting up logging with winston..."
  if ! npm list winston > /dev/null 2>&1; then
    log_info "Installing winston..."
    npm install winston || log_error "Failed to install winston."
  fi
  if [ ! -f logger.ts ]; then
    log_info "Creating logger.ts..."
    echo "import winston from 'winston';\n\nconst logger = winston.createLogger({\n  level: 'info',\n  format: winston.format.combine(\n    winston.format.timestamp(),\n    winston.format.json()\n  ),\n  transports: [\n    new winston.transports.File({ filename: 'bot.log', level: 'info' }),\n    new winston.transports.Console({ format: winston.format.simple() })\n  ]\n});\n\nexport default logger;" > logger.ts
    log_success "logger.ts created. Import and use in your code."
  fi
  log_success "Logging setup complete."
}

run_tests_and_build() {
  log_info "Running tests and compiling TypeScript..."

  if ! npx vitest run; then
    log_warning "Tests failed! Review test.log for details."
  else
    log_success "Tests completed successfully."
  fi

  if command_exists tsc; then
    log_info "Compiling TypeScript files..."
    if ! tsc; then
      log_error "TypeScript compilation failed. Check tsconfig.json or code."
    else
      log_success "TypeScript compiled successfully."
    fi
  else
    log_warning "tsc not found. Skipping compilation."
  fi
}

debug_setup() {
  log_info "Running debugging checks..."

  # Check cli.tsx and add debug log
  if [ -f cli.tsx ]; then
    log_info "Adding debug log to cli.tsx..."
    sed -i '1i import logger from \"./logger\"; logger.info(\"cli.tsx running at\", new Date().toISOString());' cli.tsx
    log_success "Debug log added."
  else
    log_error "cli.tsx not found."
  fi

  # Test cli.tsx with node
  log_info "Testing cli.tsx with node..."
  if ! node cli.tsx > cli.log 2>&1; then
    log_warning "node cli.tsx failed. Check cli.log and bot.log."
  else
    log_success "cli.tsx executed. Check cli.log and bot.log."
  fi

  # Check bot.log
  if [ ! -f bot.log ] || [ ! -s bot.log ]; then
    log_warning "bot.log is empty or missing. Ensure logger is used."
    touch bot.log
    log_success "bot.log created."
  else
    log_success "bot.log exists with data."
  fi

  # Run Vitest and check test.log
  log_info "Running Vitest for debugging..."
  if ! npx vitest run > test.log 2>&1; then
    log_warning "Vitest failed. Check test.log for errors."
  else
    log_success "Vitest executed. Check test.log."
  fi

  # Check tree.md
  if [ -f tree.md ]; then
    log_info "Reading tree.md..."
    cat tree.md
    log_success "tree.md read successfully."
  else
    log_warning "tree.md not found. Creating placeholder..."
    echo "# Project Structure\n- src/\n  - cli.tsx\n  - App.tsx\n  - strategies/\n    - BasicMarketMakingStrategy.ts\n- test/\n- data/\n  - historical_prices.json\n- state.json\n- .env\n- bot.log" > tree.md
    log_success "Placeholder tree.md created."
  fi
}

# --- Execute the main setup function ---
setup_environment