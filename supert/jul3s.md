#!/usr/bin/env bash

# Pyrmethus's Grand Jules VM Setup Incantation
# A self-contained ritual to configure the Jules VM for Worldguidex,
# imbued with enhanced durability and mystical Termux-optimized logging.

# --- Evoke the Strict Mode Spell ---
# Abort the ritual if any command fails.
set -e
# Forging a trap to catch errors and display a crimson-hued farewell.
trap 'EXIT_CODE=$? ; if [[ $EXIT_CODE -ne 0 ]]; then echo -e "\n${COLORS[RED]}✗ ERROR: Ritual terminated unexpectedly with exit code ${EXIT_CODE}. The flow of commands has been disrupted.${COLORS[NC]}" >&2; fi' EXIT

# --- Configuration Sigils ---
readonly PROJECT_DIR_NAME="jules_worldguidex_project"
readonly PROJECT_DIR="$(readlink -f "${HOME}/${PROJECT_DIR_NAME}")"
readonly APP_DIR="${PROJECT_DIR}/app"
readonly CONFIG_DIR="${PROJECT_DIR}/config"
readonly LOGS_DIR="${PROJECT_DIR}/logs"
readonly VENV_DIR="${PROJECT_DIR}/venv"

readonly PYTHON_PACKAGES=(
    "pandas"
    "pandas-ta"
    "numpy"
    "requests"
    "ccxt"
    "python-dotenv"
    "pytest"
    "pytest-cov"
    "matplotlib"
    "websocket-client"
    "colorama"
    "pybit"
)

readonly MIN_PYTHON_MAJOR=3
readonly MIN_PYTHON_MINOR=8
readonly MIN_DISK_GB=2
readonly DEBUG_MODE=${DEBUG_MODE:-false}

# --- ANSI Colors for Enhanced Output (A mystical array) ---
declare -A COLORS
COLORS[NC]='\033[0m'
COLORS[BOLD]='\033[1m'
COLORS[RED]='\033[1;31m'
COLORS[GREEN]='\033[1;32m'
COLORS[YELLOW]='\033[1;33m'
COLORS[BLUE]='\033[1;34m'
COLORS[MAGENTA]='\033[1;35m'
COLORS[CYAN]='\033[1;36m'
COLORS[WHITE]='\033[1;37m'
COLORS[UNDERLINE]='\033[4m'

# --- Logging Oracles ---
mkdir -p "${LOGS_DIR}" || { echo -e "${COLORS[RED]}ERROR: Failed to create log directory at ${LOGS_DIR}. Check your sacred permissions!${COLORS[NC]}"; exit 1; }
readonly LOG_FILE="${LOGS_DIR}/jules_setup_$(date +'%Y%m%d_%H%M%S').log"
exec 3>&1 # Create a copy of stdout

_log() {
    local level="$1"
    shift
    local message="$@"
    local timestamp="$(date +'%Y-%m-%d %H:%M:%S')"
    printf "${COLORS[CYAN]}%s${COLORS[NC]} ${COLORS[BOLD]}%s${COLORS[NC]} %b\n" "$timestamp" "$level" "$message" | tee -a "${LOG_FILE}" >&2
}

log_info() { _log "[INFO]" "${COLORS[WHITE]}$1${COLORS[NC]}"; }
log_success() { _log "[SUCCESS]" "${COLORS[GREEN]}${COLORS[BOLD]}✓${COLORS[NC]} ${COLORS[GREEN]}$1${COLORS[NC]}"; }
log_warn() { _log "[WARNING]" "${COLORS[YELLOW]}⚠️ $1${COLORS[NC]}"; }
log_error() { _log "[ERROR]" "${COLORS[RED]}${COLORS[BOLD]}✗ ${COLORS[NC]}${COLORS[RED]} $1${COLORS[NC]}"; exit 1; }

# --- Helper Incantations ---

check_command() {
    local cmd="$1"
    log_info "Divining the presence of the command: ${cmd}..."
    if ! command -v "${cmd}" &> /dev/null; then
        log_error "The command '${cmd}' is not in your mystical PATH. It is required for this ritual. Consider: pkg install ${cmd}."
    fi
    log_success "The command '${cmd}' has been divined."
}

check_python_version() {
    log_info "Verifying Python's version (required: >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})..."
    local python_version
    if ! command -v python3 &> /dev/null; then
        log_error "The Python 3 spirit is not present. Invoke 'pkg install python' before proceeding."
    fi
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    local python_major=$(echo "${python_version}" | cut -d'.' -f1)
    local python_minor=$(echo "${python_version}" | cut -d'.' -f2)
    if [[ "${python_major}" -lt "${MIN_PYTHON_MAJOR}" || ( "${python_major}" -eq "${MIN_PYTHON_MAJOR}" && "${python_minor}" -lt "${MIN_PYTHON_MINOR}" ) ]]; then
        log_error "The Python spirit is too ancient (${python_version}). A more modern spirit (>= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}) is required."
    else
        log_success "The Python spirit (${python_version}) is of a suitable vintage."
    fi
}

check_disk_space() {
    log_info "Scrying for available disk space in the cosmic realm of ${HOME} (required: >= ${MIN_DISK_GB} GB)..."
    local available_space_kb=$(df -k "${HOME}" | awk 'NR==2 {print $4}')
    local available_space_gb=$((available_space_kb / (1024 * 1024)))
    if [[ "${available_space_gb}" -lt "${MIN_DISK_GB}" ]]; then
        log_error "Only ${available_space_gb} GB of disk space is available. A minimum of ${MIN_DISK_GB} GB is recommended to prevent the ritual from failing."
    else
        log_success "A bountiful ${available_space_gb} GB of disk space is available."
    fi
}

# --- Core Ritual Stages ---

setup_directories() {
    log_info "Carving the project directories under the sacred path of ${PROJECT_DIR}..."
    local dirs=("${PROJECT_DIR}" "${APP_DIR}" "${CONFIG_DIR}" "${LOGS_DIR}")
    for dir in "${dirs[@]}"; do
        if [ ! -d "${dir}" ]; then
            log_info "Forging directory: ${dir}..."
            mkdir -p "${dir}" || log_error "Failed to forge directory: ${dir}. The permissions are misaligned."
        else
            log_warn "A directory already exists at ${dir}. The path is already forged."
        fi
    done
    log_success "All sacred project directories are in place."
}

generate_requirements_file() {
    log_info "Scribing the sacred requirements onto the scroll at ${APP_DIR}/requirements.txt..."
    local req_file="${APP_DIR}/requirements.txt"
    printf "%s\n" "${PYTHON_PACKAGES[@]}" > "${req_file}" || log_error "Failed to scribe the requirements.txt scroll."
    log_success "The requirements scroll has been inscribed with ${#PYTHON_PACKAGES[@]} packages."
}

setup_python_environment() {
    log_info "Summoning the Python virtual environment at ${VENV_DIR}..."

    if [ -d "${VENV_DIR}" ]; then
        if [ ! -f "${VENV_DIR}/bin/activate" ]; then
            log_warn "A vestige of a corrupted virtual environment was found at ${VENV_DIR}. It must be cleansed before a new one can be summoned."
            rm -rf "${VENV_DIR}" || log_error "Failed to cleanse the corrupted virtual environment at ${VENV_DIR}. Manual cleansing may be needed."
        else
            log_warn "A virtual environment already exists and appears valid at ${VENV_DIR}. Skipping the summoning ritual to preserve its essence."
        fi
    fi

    if [ ! -d "${VENV_DIR}" ]; then
        log_info "Casting the spell to create a new virtual environment..."
        python3 -m venv "${VENV_DIR}" || log_error "Failed to conjure the virtual environment. Ensure the 'python3-venv' spirit is installed with 'pkg install python-venv'."
        log_success "The virtual environment has been conjured."
    fi

    log_info "Chanting the activation spell and installing/upgrading Python dependencies..."
    (
        source "${VENV_DIR}/bin/activate" || log_error "The activation spell for the virtual environment failed. This is a critical disruption!"
        log_info "Upgrading the pip, setuptools, and wheel spirits within the venv..."
        python -m pip install --upgrade pip setuptools wheel 2>/dev/null || log_warn "The upgrade of pip spirits failed. This may not be a critical omen."
        log_info "Invoking the Python dependencies from the requirements.txt scroll..."
        python -m pip install --no-cache-dir -r "${APP_DIR}/requirements.txt" || log_error "The invocation of Python dependencies failed. Verify your internet connection or the names on the scroll."
        deactivate || log_warn "The deactivation spell failed. This should not disrupt the grand ritual."
    )
    log_success "The Python dependencies have been successfully invoked within the virtual environment."
}

check_termux_api() {
    log_info "Verifying the presence of the Termux:API runes..."
    if ! command -v termux-toast &> /dev/null; then
        log_warn "The Termux:API runes are not installed. Whispers from the device will be silenced. Install them with 'pkg install termux-api' and the Termux:API app from F-Droid."
    fi
}

# --- The Grand Incantation's Execution ---
main() {
    log_info "${COLORS[BOLD]}${COLORS[MAGENTA]}Starting Pyrmethus's Grand Jules VM Setup Incantation...${COLORS[NC]}"
    log_info "The full chronicle of this ritual is being inscribed at: ${LOG_FILE}"

    log_info "Divining the state of the system's cosmic alignment..."
    check_command "git"
    check_command "python3"
    check_python_version
    check_disk_space
    check_termux_api
    log_success "The pre-ritual divinations are complete. All is in harmony."

    setup_directories
    generate_requirements_file
    setup_python_environment

    log_success "${COLORS[BOLD]}${COLORS[GREEN]}Pyrmethus's Grand Jules VM Setup Incantation completed successfully!${COLORS[NC]}"
    log_info "The sacred project root is at: ${PROJECT_DIR}"
    log_info "The application code awaits you at: ${APP_DIR}"
    log_info "The virtual environment's lair is at: ${VENV_DIR}"
    log_info "To awaken the environment manually, speak this spell: ${COLORS[YELLOW]}source ${VENV_DIR}/bin/activate${COLORS[NC]}"

    if command -v termux-toast &> /dev/null; then
        log_info "${COLORS[MAGENTA]}(Casting a transient whisper from the device...)${COLORS[NC]}"
        termux-toast "Jules VM setup completed successfully."
    fi

    log_info "${COLORS[MAGENTA]}# May your digital journey be ever enlightened.${COLORS[NC]}"
}

# Awaken the main incantation
main "$@"
