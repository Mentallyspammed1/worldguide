#!/usr/bin/env python3

# Pyrmethus's Grand Gemini File Review Spell
# A mystical conduit for text transformation via the Gemini API, imbued with enhanced power.
# This spell can review single files, entire directories, or text channeled from the ether (stdin).
# Optimized for Termux and vibrant neon colorization themes! �

# --- Arcane Imports & Initialization ---
import argparse
import configparser
import difflib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests
from colorama import Fore, init, Style
from dotenv import load_dotenv  # For .env file support

# Summon the colors of the terminal for vibrant enchantment
init(autoreset=True)  # Autoreset is enabled to prevent color bleed

# --- Constants of the Cosmos ---
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_JOBS = 5
DEFAULT_CONNECT_TIMEOUT = 20
DEFAULT_READ_TIMEOUT = 180
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
API_RATE_LIMIT_WAIT = 61  # Seconds to wait on 429 errors
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# --- File & Directory Glyphs ---
SCRIPT_NAME = Path(__file__).name
CONFIG_DIR = Path.home() / ".config" / "pyrmethus"
CONFIG_FILE = CONFIG_DIR / "config.ini"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
CHUNK_PREFIX = "chunk_"
OUTPUT_CHUNK_PREFIX = "output_chunk_"
CHECKPOINT_SUFFIX = ".checkpoint"
BACKUP_SUFFIX = ".bak"

# --- Cost Estimation Sigils (Approximation for Gemini 1.5 Pro) ---
# Heed the official scrolls from Google for the latest rates.
# Input: ~$3.50 / 1M tokens | Output: ~$10.50 / 1M tokens
# A rough heuristic of 4 chars/token is used. A true tokenizer is more precise but heavier.
COST_PER_MILLION_INPUT_TOKENS = 3.50
COST_PER_MILLION_OUTPUT_TOKENS = 10.50
CHARS_PER_TOKEN = 4

# --- Global State Runes ---
temp_dir: Path | None = None
gemini_api_key: str = ""
raw_api_output_mode: bool = False  # Controls if raw API response is used or code block extracted

# Using standard logging for better control and integration
logger = logging.getLogger("gemini_review")
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
# Set default log level, can be overridden by CLI
logger.setLevel(logging.INFO)
# Remove any existing handlers to prevent duplicate output
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")  # We'll handle colorization in custom log functions
handler.setFormatter(formatter)
logger.addHandler(handler)
total_input_tokens: int = 0
total_output_tokens: int = 0

# --- Ethereal Logging Functions (Neon Themed) ---
def log_message(level: int, message: str, color: str = Fore.RESET, style: str = Style.RESET_ALL):
    """Logs a message with specified level and color."""
    level_name = logging.getLevelName(level)
    # Using LIGHT variants for a brighter, more "neon" feel
    if level == logging.INFO:
        prefix = f"{Fore.LIGHTCYAN_EX}[INFO]{Style.RESET_ALL}"
    elif level == logging.WARNING:
        prefix = f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}[WARNING]{Style.RESET_ALL}"
    elif level == logging.ERROR:
        prefix = f"{Fore.LIGHTRED_EX}{Style.BRIGHT}[ERROR]{Style.RESET_ALL}"
    elif level == logging.DEBUG:
        prefix = f"{Fore.LIGHTMAGENTA_EX}[DEBUG]{Style.RESET_ALL}"
    else:  # For CRITICAL and others, use a default bright white/magenta
        prefix = f"{Fore.LIGHTWHITE_EX}{Style.BRIGHT}[{level_name}]{Style.RESET_ALL}"

    logger.log(level, f"{prefix} {color}{message}{style}")
def log_info(message: str):
    """Logs an informational message."""
    log_message(logging.INFO, message, Fore.LIGHTBLUE_EX)  # Neon blue for info
def log_success(message: str):
    """Logs a success message with a verdant glow."""
    log_message(logging.INFO, message, Fore.LIGHTGREEN_EX, Style.BRIGHT)  # Neon green for success
def log_warning(message: str):
    """Logs a warning message with a cautionary amber hue."""
    log_message(logging.WARNING, message, Fore.LIGHTYELLOW_EX, Style.BRIGHT)  # Neon yellow for warning
def log_error(message: str):
    """Logs an error message with a fiery crimson light."""
    log_message(logging.ERROR, message, Fore.LIGHTRED_EX, Style.BRIGHT)  # Neon red for error
def log_debug(message: str):
    """Logs a debug message if verbose mode is enabled, revealing the inner workings."""
    log_message(logging.DEBUG, message, Fore.LIGHTMAGENTA_EX)  # Neon magenta for debug

# --- Core Ritual Logic ---
def load_config() -> configparser.ConfigParser:
    """Loads configuration from the Pyrmethus config scroll."""
    config = configparser.ConfigParser()
    if CONFIG_FILE.is_file():
        log_debug(f"Loading configuration from {CONFIG_FILE}")
        try:
            config.read(CONFIG_FILE)
        except configparser.Error as e:
            log_error(f"Error reading config scroll {CONFIG_FILE}: {e}")
            return configparser.ConfigParser()
    return config
def get_api_key(args_key: str, config: configparser.ConfigParser):
    """Summons the Gemini API key from arguments, .env, environment, or config."""
    global gemini_api_key

    # 1. Command-line argument (highest priority)
    if args_key:
        gemini_api_key = args_key
        log_debug("API key summoned from command-line arguments.")
        return

    # 2. .env file (loaded first by dotenv)
    load_dotenv()  # Load variables from .env file
    if "GEMINI_API_KEY" in os.environ:
        gemini_api_key = os.environ["GEMINI_API_KEY"]
        log_debug("API key summoned from .env file or environment's ether.")
        return

    # 3. Config file
    if config.has_section("Gemini") and "api_key" in config["Gemini"]:
        gemini_api_key = config["Gemini"]["api_key"]
        log_debug("API key summoned from config scroll.")
        return

    # 4. Interactive input (lowest priority)
    try:
        gemini_api_key = input(f"{Fore.LIGHTBLUE_EX}{Style.BRIGHT}Enter your Gemini API key: {Style.RESET_ALL}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        log_error("API key input cancelled. The ritual cannot proceed without the key.")
        sys.exit(1)

    if not gemini_api_key:
        log_error("The API key is the heart of this ritual. It cannot be absent!")
        sys.exit(1)

    try:
        save_key = input(f"{Fore.LIGHTYELLOW_EX}Inscribe this key to {CONFIG_FILE} for future rituals? (y/N): {Style.RESET_ALL}").lower()
        if save_key == "y":
            save_api_key_to_config(gemini_api_key, config)
    except (EOFError, KeyboardInterrupt):
        print()
        log_info("Skipping key inscription.")

    if len(gemini_api_key) < 30:  # Gemini keys are typically long, this is a heuristic check
        log_warning("API Key seems unusually short. Please verify its authenticity.")
    if not gemini_api_key:
        log_error("Gemini API key is missing. Please provide it via -k, GEMINI_API_KEY env var, or interactively.")
        sys.exit(1)
def save_api_key_to_config(key_to_save: str, current_config: configparser.ConfigParser):
    """Inscribes the Gemini API key to the config scroll."""
    if not CONFIG_DIR.exists():
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            log_debug(f"Created config directory: {CONFIG_DIR}")
        except OSError as e:
            log_error(f"Could not create config directory {CONFIG_DIR}: {e}")
            return

    if not current_config.has_section("Gemini"):
        current_config.add_section("Gemini")
    current_config["Gemini"]["api_key"] = key_to_save

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            current_config.write(f)
        log_success(f"API key inscribed in {CONFIG_FILE}")
    except OSError as e:
        log_error(f"Failed to inscribe API key to config scroll {CONFIG_FILE}: {e}")

def check_dependencies():
    """Ensures the presence of optional arcane tools for pre-flight checks."""
    log_info(f"{Fore.LIGHTMAGENTA_EX}Inspecting the arcane arsenal for optional tools...{Style.RESET_ALL}")
    deps = {"ruff": "Python linter & formatter", "flake8": "Python linter", "black": "Python formatter", "shellcheck": "Shell script linter"}
    missing = []
    for dep, desc in deps.items():
        if not shutil.which(dep):
            missing.append(dep)
            log_debug(f"Missing optional tool: {dep} ({desc})")
    if missing:
        log_warning(f"Optional tools not found: {', '.join(missing)}. Pre-check functionality will be limited.")
    else:
        log_success("All optional arcane tools found.")

def create_temp_dir(args: argparse.Namespace) -> bool:
    """Conjures a secure temporary sanctum for chunk sorcery."""
    global temp_dir
    try:
        base_tmp_dir = args.temp_dir
        if not base_tmp_dir:
            # Prioritize Termux's tmp directory if it exists and is writable
            termux_tmp_path = Path("/data/data/com.termux/files/usr/tmp")
            base_tmp_dir = termux_tmp_path if termux_tmp_path.is_dir() and os.access(termux_tmp_path, os.W_OK) else None

        if base_tmp_dir:
            if not Path(base_tmp_dir).is_dir():
                log_error(f"Custom temporary directory does not exist: {base_tmp_dir}")
                return False
            if not os.access(base_tmp_dir, os.W_OK):
                log_error(f"Custom temporary directory is not writable: {base_tmp_dir}")
                return False

        temp_dir = Path(tempfile.mkdtemp(prefix="gemini_review_", dir=base_tmp_dir))
        log_debug(f"Temporary sanctum created at: {temp_dir}")
        return True
    except Exception as e:
        log_error(f"Failed to conjure temporary sanctum: {e}")
        return False

def cleanup_temp_dir():
    """Dissolves the temporary sanctum, banishing its remnants."""
    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            log_debug(f"Sanctum '{temp_dir}' dissolved into the void.")
        except OSError as e:
            log_warning(f"Could not banish temporary sanctum '{temp_dir}': {e}")

def handle_exit(signum=None, frame=None):
    """Guides the script to a graceful exit, ensuring no trace is left behind."""
    print()  # Newline for clean exit
    log_info(f"{Fore.LIGHTMAGENTA_EX}A signal from the beyond... initiating graceful dissolution of the ritual.{Style.RESET_ALL}")
    cleanup_temp_dir()
    sys.exit(0)

def backup_output_file(output_path: Path, force: bool, dry_run: bool) -> bool:
    """Preserves the existing output file as a sacred relic before overwriting."""
    if dry_run:
        log_debug(f"Dry run enabled. Skipping backup for '{output_path.name}'.")
        return True

    if not output_path.exists():
        return True

    # If a checkpoint exists, it means we are resuming, so no explicit backup needed
    if output_path.with_suffix(CHECKPOINT_SUFFIX).exists():
        log_info(f"Checkpoint found for '{output_path.name}'. Skipping explicit backup; will resume.")
        return True

    if not force:
        try:
            response = input(f"{Fore.LIGHTYELLOW_EX}Output file '{output_path.name}' exists. Overwrite? (y/N): {Style.RESET_ALL}").lower().strip()
            if response != "y":
                log_info("Aborted by user.")
                return False
        except (EOFError, KeyboardInterrupt):
            print()
            log_info("Aborted by user.")
            return False

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    backup_path = output_path.with_name(f"{output_path.name}.{timestamp}{BACKUP_SUFFIX}")
    try:
        shutil.copy2(output_path, backup_path)
        log_info(f"Output file preserved as '{backup_path.name}'.")
    except OSError as e:
        log_error(f"Failed to preserve output file '{output_path.name}': {e}")
        return False
    return True

def get_file_type_from_extension(file_path: Path) -> str | None:
    """Infers a language hint from a file's extension."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".sh": "bash",
        ".bash": "bash", ".zsh": "bash", ".html": "html", ".css": "css", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".xml": "xml", ".md": "markdown", ".txt": "text",
        ".go": "go", ".c": "c", ".cpp": "cpp", ".java": "java", ".rs": "rust", ".rb": "ruby",
        ".php": "php", ".swift": "swift", ".kt": "kotlin", ".m": "objective-c", ".h": "c",
        ".hpp": "cpp", ".cs": "csharp", ".vue": "vue", ".jsx": "javascript", ".tsx": "typescript",
        ".ipynb": "json"  # Treat notebooks as JSON for parsing, content might be extracted
    }
    return ext_map.get(file_path.suffix.lower())

def split_file_into_chunks(input_path: Path, temp_dir_path: Path, lang_hint: str | None = None, max_chunk_tokens: int | None = None) -> list[Path]:
    """
    Divides the input tome into content-aware fragments.
    Uses language-specific heuristics for Python and Bash, and a paragraph-based
    approach for others. Also handles max_chunk_tokens by further subdividing.
    """
    log_info(f"Dividing '{input_path.name}' into fragments...")
    try:
        content = input_path.read_text(encoding="utf-8")
    except OSError as e:
        log_error(f"Could not read input tome '{input_path.name}': {e}")
        return []

    if not content.strip():
        log_warning(f"Input tome '{input_path.name}' is empty. Treating as a single void chunk.")
        chunk_path = temp_dir_path / f"{CHUNK_PREFIX}0000"
        try:
            chunk_path.write_text("", encoding="utf-8")
        except OSError as e:
            log_error(f"Could not write empty chunk file '{chunk_path.name}': {e}")
            return []
        return [chunk_path]

    chunks_raw = []
    lines = content.splitlines(keepends=True)  # Keepends=True preserves newlines for reassembly

    if lang_hint == "python":
        log_debug("Using Python-aware splitting strategy (zero-indentation blocks).")
        current_chunk_lines = []
        for line in lines:
            stripped_line = line.lstrip()
            # If line is not empty and not a comment, and starts at column 0 (top-level)
            if stripped_line and not stripped_line.startswith("#") and (len(line) - len(stripped_line)) == 0:
                # If we have accumulated lines, this marks the start of a new block
                if current_chunk_lines:
                    chunks_raw.append("".join(current_chunk_lines))
                    current_chunk_lines = [line]
                else:  # First line of the file or first top-level statement
                    current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)
        if current_chunk_lines:  # Add the last accumulated chunk
            chunks_raw.append("".join(current_chunk_lines))

        # Filter out any chunks that are purely whitespace or comments, unless it's the only chunk
        if len(chunks_raw) > 1:
            chunks_raw = [c for c in chunks_raw if c.strip()]
        if not chunks_raw and content.strip():  # If all chunks were filtered but content exists, keep original
            chunks_raw = [content]

    elif lang_hint in ["bash", "sh", "zsh"]:
        log_debug("Using shell-aware splitting strategy (functions, control blocks, line continuations).")
        current_chunk_lines = []
        in_multi_line_command = False

        for line in lines:
            stripped_line = line.strip()

            # Check for line continuation (backslash at end of line, not just whitespace)
            if line.rstrip().endswith("\\"):
                in_multi_line_command = True
                current_chunk_lines.append(line)
                continue

            # If we were in a multi-line command, continue adding lines until it ends
            if in_multi_line_command:
                current_chunk_lines.append(line)
                in_multi_line_command = False  # Assume it ends unless another backslash is found
                continue

            # Check for new logical blocks (functions, if/for/while/case, or significant commands)
            # This regex looks for:
            # - function definitions: `func_name() {` or `function func_name {`
            # - control flow: `if`, `for`, `while`, `case` at line start
            # - comments that might delineate sections (e.g., `### SECTION ###`)
            # - a new command starting at the beginning of the line (not indented, not blank, not comment)
            is_new_block_start = re.match(
                r"^\s*(?:function\s+\w+\s*\{|\w+\s*\(\s*\)\s*\{|"  # function definitions
                r"if\s|for\s|while\s|case\s|"  # control flow
                r"#+\s*\S.*|"  # significant comments
                r"[a-zA-Z_][a-zA-Z0-9_]*\s+.*?"  # simple command start (e.g., `echo "hi"`)
                r")", stripped_line
            )

            # If a new block starts and we have accumulated lines, save the current chunk
            if is_new_block_start and current_chunk_lines:
                chunks_raw.append("".join(current_chunk_lines))
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

        if current_chunk_lines:  # Add the last accumulated chunk
            chunks_raw.append("".join(current_chunk_lines))

        # Filter out any chunks that are purely whitespace or comments, unless it's the only chunk
        if len(chunks_raw) > 1:
            chunks_raw = [c for c in chunks_raw if c.strip()]
        if not chunks_raw and content.strip():  # If all chunks were filtered but content exists, keep original
            chunks_raw = [content]

    else:
        log_debug("Using default paragraph-like splitting strategy (2+ newlines).")
        # Split by two or more newlines, keeping the newlines as delimiters to ensure they're not lost.
        # This creates chunks like: [text, \n\n, text, \n\n, ...]
        chunks_raw = re.split(r"(\n{2,})", content)
        # Recombine text and their following newlines
        temp_recombined_chunks = []
        i = 0
        while i < len(chunks_raw):
            current_piece = chunks_raw[i]
            if current_piece.strip():  # If it's actual content
                if i + 1 < len(chunks_raw) and re.match(r"\n{2,}", chunks_raw[i + 1]):
                    temp_recombined_chunks.append(current_piece + chunks_raw[i + 1])
                    i += 2
                else:
                    temp_recombined_chunks.append(current_piece)
                    i += 1
            else:  # If it's just newlines at the start or consecutive newlines
                i += 1
        chunks_raw = [c for c in temp_recombined_chunks if c.strip()]  # Filter out any remaining empty strings

    if not chunks_raw:
        log_warning("No logical fragments created. Using entire file as one chunk.")
        chunks_raw = [content]

    final_chunks_content = []
    for chunk_content in chunks_raw:
        if max_chunk_tokens:
            estimated_tokens = len(chunk_content) // CHARS_PER_TOKEN
            if estimated_tokens > max_chunk_tokens:
                log_debug(f"Fragment too large (~{estimated_tokens} tokens), subdividing by line.")
                lines = chunk_content.splitlines(keepends=True)
                current_sub_chunk_lines = []
                current_sub_chunk_tokens = 0
                for line in lines:
                    line_tokens = len(line) // CHARS_PER_TOKEN
                    if current_sub_chunk_tokens + line_tokens > max_chunk_tokens and current_sub_chunk_lines:
                        final_chunks_content.append("".join(current_sub_chunk_lines))
                        current_sub_chunk_lines = [line]
                        current_sub_chunk_tokens = line_tokens
                    else:
                        current_sub_chunk_lines.append(line)
                        current_sub_chunk_tokens += line_tokens
                if current_sub_chunk_lines:  # Add the last sub-chunk
                    final_chunks_content.append("".join(current_sub_chunk_lines))
                continue  # Move to next chunk_content
        final_chunks_content.append(chunk_content)

    chunk_files = []
    for i, chunk_content in enumerate(final_chunks_content):
        chunk_path = temp_dir_path / f"{CHUNK_PREFIX}{i:04d}"
        try:
            chunk_path.write_text(chunk_content, encoding="utf-8")
            chunk_files.append(chunk_path)
        except OSError as e:
            log_error(f"Could not write fragment file '{chunk_path.name}': {e}")
            return []

    log_success(f"Tome successfully divided into {len(chunk_files)} fragments.")
    return chunk_files

def extract_code_block(text: str) -> str:
    """Extracts the first code block from Gemini's response, or returns the text if no block is found."""
    global raw_api_output_mode
    if raw_api_output_mode:
        log_debug(f"Raw API response (raw_api_output_mode=True): \n---\n{text}\n---")
        return text.strip()

    # This regex is robust for various code fence formats (e.g., ```python, ```js, ```)
    match = re.search(r"```(?:\S*\n)?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    log_debug("No code fences found in response. Using the entire text as is.")
    return text.strip()

def run_pre_check(content: str, lang_hint: str) -> bool:
    """Runs language-specific linting/formatting checks. Returns True if check passes or is skipped."""
    log_info(f"Running pre-check for {lang_hint or 'unknown language'}...")
    if lang_hint == "python":
        has_ruff = shutil.which("ruff")
        if has_ruff:
            log_debug("Using Ruff for Python pre-check.")
            # Ruff combines linting and formatting. `ruff check --fix` can be used,
            # but for a pure check, we'll run `check` and `format --check`.
            # We consider it "passed" if Ruff would make no changes.

            # Check for formatting issues
            try:
                format_proc = subprocess.run(["ruff", "format", "--check", "-"], input=content.encode("utf-8"), capture_output=True, check=False, encoding="utf-8")
                if format_proc.returncode != 0:
                    log_warning(f"Ruff format check failed: {format_proc.stderr.strip()}")
                    return False
                log_debug("Ruff format check passed.")
            except FileNotFoundError:
                log_warning("Ruff executable not found.")
                return True # If ruff is missing, we can't run the check, so we assume pass

            # Check for linting issues
            try:
                lint_proc = subprocess.run(["ruff", "check", "-"], input=content.encode("utf-8"), capture_output=True, check=False, encoding="utf-8")
                if lint_proc.returncode != 0:
                    log_warning(f"Ruff lint check failed: {lint_proc.stderr.strip()}")
                    return False
                log_debug("Ruff lint check passed.")
            except FileNotFoundError:
                log_warning("Ruff executable not found.")
                return True # If ruff is missing, we can't run the check, so we assume pass

            log_info("Ruff pre-check passed.")
            return True

        # Fallback to black and flake8 if ruff is not available
        has_black = shutil.which("black")
        has_flake8 = shutil.which("flake8")
        if not has_black and not has_flake8:
            log_warning("Neither 'ruff', 'black', nor 'flake8' found. Skipping Python pre-check.")
            return True
        
        if has_black:
            log_debug("Running Black check...")
            try:
                proc = subprocess.run(["black", "--check", "--quiet", "-"], input=content.encode("utf-8"), capture_output=True, check=False, encoding="utf-8")
                if proc.returncode not in [0, 123]: # 123 means unmodified, which is a success for --check
                    log_warning(f"Black check failed: {proc.stderr.strip()}")
                    return False
                log_debug("Black check passed.")
            except FileNotFoundError:
                log_warning("Black executable not found.")
        
        if has_flake8:
            log_debug("Running Flake8 check...")
            try:
                proc = subprocess.run(["flake8", "--stdin-display-name", "stdin"], input=content.encode("utf-8"), capture_output=True, check=False, encoding="utf-8")
                if proc.returncode != 0:
                    log_warning(f"Flake8 check failed: {proc.stderr.strip()}")
                    return False
                log_debug("Flake8 check passed.")
            except FileNotFoundError:
                log_warning("Flake8 executable not found.")
        
        log_info("Python pre-check passed.")
        return True
    
    if lang_hint in ["bash", "sh", "zsh"]:
        if not shutil.which("shellcheck"):
            log_warning("shellcheck not found. Skipping shell pre-check.")
            return True
        log_debug("Running ShellCheck check...")
        try:
            proc = subprocess.run(["shellcheck", "-s", lang_hint, "-"], input=content.encode("utf-8"), capture_output=True, check=False, encoding="utf-8")
            if proc.returncode != 0:
                log_warning(f"ShellCheck failed: {proc.stderr.strip()}")
                return False
            log_info("Shell pre-check passed.")
            return True
        except FileNotFoundError:
            log_warning("ShellCheck executable not found.")
            return True # If shellcheck is missing, we can't run the check, so we assume pass

    log_debug(f"No pre-check defined for language '{lang_hint}'. Skipping.")
    return True

def process_chunk_with_api(original_chunk_path: Path, output_chunk_path: Path, model_url: str, args: argparse.Namespace, file_info: str) -> tuple[str | None, str, Path, int, int]:
    """Channels a single fragment through the Gemini API for enhancement."""
    global total_input_tokens, total_output_tokens
    chunk_name = original_chunk_path.name
    log_debug(f"Channeling fragment: {chunk_name}")

    try:
        original_content = original_chunk_path.read_text(encoding="utf-8")
    except OSError as e:
        log_error(f"Could not read fragment '{original_chunk_path.name}': {e}")
        return None, "", original_chunk_path, 0, 0  # Return None for processed_content if read fails

    if not original_content.strip():
        log_debug(f"Skipping empty or whitespace-only fragment: {chunk_name}. No changes made.")
        try:
            output_chunk_path.write_text("", encoding="utf-8")
        except OSError as e:
            log_error(f"Could not write empty output fragment '{output_chunk_path.name}': {e}")
        return "", original_content, original_chunk_path, 0, 0

    current_lang_hint = args.lang or get_file_type_from_extension(Path(file_info))  # Use Path() to ensure get_file_type works correctly

    # Pre-check *original* content to decide if API call is needed
    if args.pre_check and current_lang_hint and run_pre_check(original_content, current_lang_hint):
        log_info(f"Pre-check passed for fragment '{chunk_name}'. Skipping API call. Using original content.")
        try:
            output_chunk_path.write_text(original_content, encoding="utf-8")
        except OSError as e:
            log_error(f"Could not write original content to output fragment '{output_chunk_path.name}': {e}")
        # Return original_content and its token count
        return original_content, original_content, original_chunk_path, len(original_content) // CHARS_PER_TOKEN, 0

    # If no custom prompt, use the default instructions
    if args.custom_prompt_template:
        prompt_text = args.custom_prompt_template.replace("{original_code}", original_content).replace("{lang_hint}", current_lang_hint or "")
        log_debug("Using custom prompt template.")
    else:
        prompt_text = (
            f"You are an expert software engineer. Your task is to review and provide "
            f"only syntax corrections for the following code chunk from '{file_info}'.\n\n"
            "**CRITICAL INSTRUCTIONS:**\n"
            "1.  **If the code is syntactically correct, return it completely unchanged.**\n"
            "2.  **Correct ONLY syntax errors.** Do not make stylistic changes, rename anything, "
            "    alter logic, add/remove comments, or reformat unless strictly necessary to fix syntax.\n"
            "3.  **Respond ONLY with the complete, corrected code chunk.** Do not include any "
            "    text, explanations, or code fences outside of the code block.\n"
            f"4.  If the input contains a language hint (e.g., ```{current_lang_hint}`), maintain it in your output.\n"
            "5.  Ensure the output is a fully runnable and syntactically valid code chunk.\n\n"
            f"Original Code Chunk:\n```{current_lang_hint or ''}\n{original_content}\n```"
        )

    json_payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": args.temperature, "maxOutputTokens": 8192}
    }

    # Estimate input tokens - can be refined with a tokenizer if needed
    input_tokens = len(prompt_text) // CHARS_PER_TOKEN

    for attempt in range(1, args.retries + 1):
        try:
            log_debug(f"Sending fragment '{chunk_name}' (Attempt {attempt}/{args.retries})...")
            response = requests.post(
                url=model_url,
                headers={"Content-Type": "application/json"},
                json=json_payload,
                timeout=(args.connect_timeout, args.read_timeout)
            )
            response.raise_for_status()

            response_data = response.json()
            # Safely get the text content, handling potential missing keys and structures
            text_content = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")

            if not text_content or text_content.strip() == "null":
                log_warning(f"API returned empty or 'null' content for '{chunk_name}'. Retrying...")
                time.sleep(args.retry_delay)
                continue

            final_content = extract_code_block(text_content)
            # Estimate output tokens based on the extracted content. A real tokenizer would be more accurate.
            output_tokens = len(final_content) // CHARS_PER_TOKEN

            # Post-API check only if content changed, if pre-check is enabled, and if a language hint is available
            if args.pre_check and final_content.strip() != original_content.strip() and current_lang_hint:
                log_info(f"Running post-API check for '{chunk_name}'...")
                if not run_pre_check(final_content, current_lang_hint):
                    log_warning(f"Post-API check failed for '{chunk_name}'. Reverting to original content to prevent new errors.")
                    final_content = original_content
                    output_tokens = len(final_content) // CHARS_PER_TOKEN  # Recalculate if reverted
            elif args.pre_check and not current_lang_hint:
                log_debug(f"Skipping post-API check for '{chunk_name}' due to no language hint.")

            try:
                output_chunk_path.write_text(final_content, encoding="utf-8")
            except OSError as e:
                log_error(f"Could not write final content to output fragment '{output_chunk_path.name}': {e}")
                # Returning original_content so reassembly can use it if output write fails
                return original_content, original_content, original_chunk_path, input_tokens, len(original_content) // CHARS_PER_TOKEN

            # Return the processed content, original content, path, input tokens, and output tokens
            return final_content, original_content, original_chunk_path, input_tokens, output_tokens

        except requests.exceptions.Timeout:
            log_warning(f"Request timed out for '{chunk_name}'. Retrying in {args.retry_delay}s...")
            time.sleep(args.retry_delay)
        except requests.exceptions.HTTPError as e:
            error_details = e.response.text if e.response else "No response text"
            if e.response and e.response.status_code == 429:
                log_warning(f"Rate limit reached for '{chunk_name}'. Waiting {API_RATE_LIMIT_WAIT}s before next attempt...")
                time.sleep(API_RATE_LIMIT_WAIT)
            elif e.response and e.response.status_code in [400, 401, 403]:  # Added 403 for Forbidden
                log_error(f"API Error {e.response.status_code} for '{chunk_name}'. This is not retryable. Check prompt, API key, or permissions. Details: {error_details[:200]}")
                return None, original_content, original_chunk_path, input_tokens, 0  # Indicate failure by returning None for processed_content
            else:
                log_error(f"API failed for '{chunk_name}' with code {e.response.status_code if e.response else 'N/A'}. Retrying... Details: {error_details[:200]}")
                time.sleep(args.retry_delay)
        except (ValueError, json.JSONDecodeError, IndexError) as e:
            log_error(f"Error parsing API response for '{chunk_name}': {e}. Retrying...")
            time.sleep(args.retry_delay)
        except Exception as e:
            log_error(f"Unexpected error for '{chunk_name}': {e}. Retrying...")
            time.sleep(args.retry_delay)

    log_error(f"Failed to enhance fragment '{chunk_name}' after {args.retries} attempts. Using original content.")
    try:
        output_chunk_path.write_text(original_content, encoding="utf-8")
    except OSError as e:
        log_error(f"Could not write original content to output fragment '{output_chunk_path.name}' after failures: {e}")
    # Return original content and its token count, and None for processed_content to indicate failure
    return original_content, original_content, original_chunk_path, input_tokens, len(original_content) // CHARS_PER_TOKEN

def display_diff(original_content: str, new_content: str, file_path_for_display: str):
    """Displays a colorized diff between original and new content."""
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path_for_display}",
        tofile=f"b/{file_path_for_display}",
        lineterm=""  # Prevent adding extra newlines if lines already have them
    )
    diff_lines = list(diff)
    if not diff_lines:
        print(f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}No changes detected for this fragment.{Style.RESET_ALL}")
        return
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            print(f"{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{line.strip()}{Style.RESET_ALL}")  # Neon blue for file info
        elif line.startswith("+"):
            print(f"{Fore.LIGHTGREEN_EX}{line.strip()}{Style.RESET_ALL}")  # Neon green for additions
        elif line.startswith("-"):
            print(f"{Fore.LIGHTRED_EX}{line.strip()}{Style.RESET_ALL}")  # Neon red for deletions
        elif line.startswith("@@"):
            print(f"{Fore.LIGHTMAGENTA_EX}{line.strip()}{Style.RESET_ALL}")  # Neon magenta for chunk info
        else:
            print(line.strip())

def reassemble_output(all_original_chunks: list[Path], output_path: Path, temp_dir_path: Path, to_stdout: bool, interactive_review: bool, dry_run: bool):
    """Weaves processed fragments into the final enhanced tome, with optional interactive review."""
    log_info(f"Weaving enhanced fragments into '{'stdout' if to_stdout else output_path.name}'...")
    final_content_parts = []
    accept_all = False

    for original_chunk_file in all_original_chunks:
        chunk_basename = original_chunk_file.name
        original_chunk_number = chunk_basename.split("_")[-1]  # e.g., "0000" from "chunk_0000"
        output_chunk_file = temp_dir_path / f"{OUTPUT_CHUNK_PREFIX}{original_chunk_number}"

        try:
            original_content = original_chunk_file.read_text(encoding="utf-8")
            # If the output chunk file doesn't exist (e.g., due to an error during processing or it was skipped),
            # default to the original content to maintain integrity.
            new_content = output_chunk_file.read_text(encoding="utf-8") if output_chunk_file.exists() else original_content
        except OSError as e:
            log_error(f"Error reading fragment files for reassembly ({chunk_basename}): {e}. Skipping this chunk and using original content.")
            final_content_parts.append(original_content)  # Ensure original content is used if files can't be read
            continue

        # If the original chunk ended with a newline but the API response doesn't, add it back.
        # This prevents chunks from being merged onto a single line, causing syntax errors.
        # Only do this if the new content is not empty; otherwise, preserve the emptiness.
        if original_content.endswith("\n") and new_content and not new_content.endswith("\n"):
            new_content += "\n"

        # Interactive review only if there are actual changes and not in dry-run
        if interactive_review and not accept_all and original_content != new_content and not dry_run:
            print(f"\n{Style.BRIGHT}{Fore.LIGHTYELLOW_EX}--- Reviewing fragment: {original_chunk_file.name} for {output_path.name} ---{Style.RESET_ALL}")
            display_diff(original_content, new_content, original_chunk_file.name)  # Use chunk name for display

            while True:
                try:
                    response = input(f"{Fore.LIGHTGREEN_EX}Accept changes? {Style.BRIGHT}(y/n/e[dit]/a[ccept all]): {Style.RESET_ALL}").lower().strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    log_info("Interactive review cancelled. Rejecting remaining changes for this file.")
                    response = "n"  # Treat as 'n' for current and exit loop
                    break  # Exit inner while loop for this chunk

                if response == "y":
                    final_content_parts.append(new_content)
                    log_debug(f"Accepted changes for {chunk_basename}.")
                    break
                if response == "n":
                    final_content_parts.append(original_content)
                    log_info(f"Rejected changes for {chunk_basename}. Using original.")
                    break
                if response == "e":
                    editor = os.environ.get("EDITOR", "nano") # Default to nano if EDITOR not set
                    temp_edit_file = temp_dir_path / f"edit_{chunk_basename}"
                    try:
                        temp_edit_file.write_text(new_content, encoding="utf-8")
                    except OSError as e:
                        log_error(f"Could not write temporary file for editing {temp_edit_file.name}: {e}. Using current content.")
                        final_content_parts.append(new_content)  # Proceed with current new content
                        break
                    log_info(f"Opening '{temp_edit_file.name}' in {editor} for manual editing...")
                    try:
                        # Use 'subprocess.run' to wait for the editor to exit
                        subprocess.run([editor, str(temp_edit_file)], check=False)
                        edited_content = temp_edit_file.read_text(encoding="utf-8")
                        # Also ensure the manually edited content preserves the trailing newline if needed.
                        if original_content.endswith("\n") and edited_content and not edited_content.endswith("\n"):
                            edited_content += "\n"
                    except (OSError, subprocess.CalledProcessError) as e:
                        log_error(f"Error during manual edit or reading edited file {temp_edit_file.name}: {e}. Using pre-edit content.")
                        edited_content = new_content  # Fallback to previous new content
                    log_info("Manual edit complete. Using edited version.")
                    final_content_parts.append(edited_content)
                    break
                if response == "a":
                    log_info("Accepting this and all subsequent changes automatically.")
                    accept_all = True
                    final_content_parts.append(new_content)
                    break
                print(f"{Fore.LIGHTRED_EX}Invalid input. Please use 'y', 'n', 'e', or 'a'.{Style.RESET_ALL}")
        elif dry_run and original_content != new_content:
            print(f"\n{Style.BRIGHT}{Fore.LIGHTYELLOW_EX}--- Dry Run: Changes for fragment: {original_chunk_file.name} ---")
            display_diff(original_content, new_content, original_chunk_file.name)
            # In dry run, we conceptually accept all changes to show the final state,
            # but the original content will be written if interactive review isn't on.
            final_content_parts.append(new_content)
        else:
            # If not interactive review, or if changes were rejected, use new_content (which might be original_content)
            final_content_parts.append(new_content)

    final_output = "".join(final_content_parts)
    # Ensure the output ends with a newline if it's not empty and doesn't already have one
    if final_output and not final_output.endswith("\n"):
        final_output += "\n"

    if dry_run:
        log_info(f"{Fore.LIGHTCYAN_EX}Dry run complete. No files were written.{Style.RESET_ALL}")
        # If --stdout is specified, print the dry-run output
        if to_stdout:
            sys.stdout.write(final_output)
            sys.stdout.flush()
    elif to_stdout:
        sys.stdout.write(final_output)
        sys.stdout.flush()  # Ensure it's immediately written
    else:
        try:
            # Ensure parent directories exist for the output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_output)
            log_success(f"Tome woven successfully. Saved to '{output_path.name}'.")
        except OSError as e:
            log_error(f"Failed to write final tome to '{output_path.name}': {e}")

def display_final_summary(files_processed_count: int, dry_run: bool):
    """Displays a summary of the ritual's cost and completion."""
    print(f"\n{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")
    if dry_run:
        log_info(f"{Fore.LIGHTMAGENTA_EX}Dry Run Complete. No files were modified.{Style.RESET_ALL}")
    else:
        log_info(f"{Fore.LIGHTCYAN_EX}Ritual Complete. Final Summary:{Style.RESET_ALL}")

    log_info(f"  Files Processed: {files_processed_count}")

    input_cost = (total_input_tokens / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS
    output_cost = (total_output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT_TOKENS
    total_cost = input_cost + output_cost

    log_info(f"  Estimated Input Tokens:  ~{total_input_tokens:,}")
    log_info(f"  Estimated Output Tokens: ~{total_output_tokens:,}")
    log_success(f"  Estimated Total Cost:    ${total_cost:.4f}")
    print(f"{Fore.LIGHTBLUE_EX}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

def get_files_from_input_dir(input_dir: Path, output_dir: Path, force: bool, skip_extensions: list[str]) -> list[tuple[Path, Path]]:
    """Collects file paths for processing from an input directory."""
    files_to_process: list[tuple[Path, Path]] = []
    log_info(f"{Fore.LIGHTBLUE_EX}Scanning directory '{input_dir}' for files to process...{Style.RESET_ALL}")
    # Common binary/archive/image extensions to skip by default if not explicitly in skip_ext
    default_binary_skip = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
                           ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2", ".xz",
                           ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
                           ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mov", ".mkv",
                           ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
                           ".sqlite3", ".db", ".lock", ".log", ".tmp", ".bak", ".pyc",
                           ".o", ".a", ".so", ".dll"]

    # Combine user-provided skips with default binary skips, ensuring case-insensitivity
    all_skip_extensions = set(ext.lower() for ext in skip_extensions) | set(default_binary_skip)

    for root_str, _, files in os.walk(input_dir):
        root = Path(root_str)
        for file_name in files:
            file_path = root / file_name
            # Skip the script itself, backup files, checkpoint files, and explicitly skipped extensions
            if file_path.name == SCRIPT_NAME or \
               file_path.suffix.lower() in [BACKUP_SUFFIX, CHECKPOINT_SUFFIX] or \
               file_path.suffix.lower() in all_skip_extensions:
                log_debug(f"Skipping internal/backup/excluded file: {file_path.name}")
                continue

            relative_path = file_path.relative_to(input_dir)
            output_file_path = output_dir / relative_path

            try:
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log_error(f"Failed to create parent directory for '{output_file_path.name}': {e}. Skipping file.")
                continue
            
            # Check if the file is empty. If so, log a warning and skip.
            try:
                if file_path.stat().st_size == 0:
                    log_warning(f"Input file '{file_path.name}' is empty. Skipping.")
                    continue
            except FileNotFoundError:
                 log_warning(f"Input file '{file_path.name}' disappeared during scan. Skipping.")
                 continue

            if not force and output_file_path.exists():
                log_info(f"Output file '{output_file_path.name}' already exists. Skipping. Use --force to overwrite.")
                continue
            
            files_to_process.append((file_path, output_file_path))
            
    if not files_to_process and not any(f for _, f in files_to_process): # Check if anything was actually added
        log_warning(f"No processable files found in '{input_dir}'. Ensure files are not empty, not backups, and not excluded by --skip-ext.")

    return files_to_process
def process_single_file_workflow(original_file_path: Path, output_file_path: Path, api_url: str, args: argparse.Namespace):
    """Manages the workflow for processing a single file: chunking, concurrent API calls, and reassembly."""
    global total_input_tokens, total_output_tokens
    log_info(f"\n{Style.BRIGHT}{Fore.LIGHTMAGENTA_EX}--- Processing Tome: {original_file_path.name} ---{Style.RESET_ALL}")

    file_lang_hint = args.lang or get_file_type_from_extension(original_file_path)
    if not file_lang_hint:
        log_warning(f"Could not infer language for '{original_file_path.name}'. Consider specifying with -l.")

    checkpoint_file = None
    # Only create checkpoints if writing to a file (not stdout) and not in dry-run mode
    if str(output_file_path) != "-" and not args.dry_run:
        checkpoint_file = output_file_path.with_suffix(CHECKPOINT_SUFFIX)

    processed_chunks_basenames = set()
    if checkpoint_file and checkpoint_file.exists():
        log_info(f"Resuming from checkpoint '{checkpoint_file.name}'...")
        try:
            processed_chunks_basenames = {line.strip() for line in checkpoint_file.read_text(encoding="utf-8").splitlines() if line.strip()}
            log_debug(f"Loaded {len(processed_chunks_basenames)} processed chunk entries from checkpoint.")
        except OSError as e:
            log_error(f"Failed to read checkpoint file '{checkpoint_file.name}': {e}. Starting from scratch for this file.")
            processed_chunks_basenames.clear()

    all_chunks_raw_paths = split_file_into_chunks(original_file_path, temp_dir, file_lang_hint, args.max_chunk_tokens)
    if not all_chunks_raw_paths:
        log_error(f"Division of text for '{original_file_path.name}' failed. Skipping file.")
        return

    chunks_to_process_info = []  # List of (original_chunk_path, output_chunk_path) for processing
    for temp_chunk_path in all_chunks_raw_paths:
        chunk_basename = temp_chunk_path.name
        original_chunk_number_from_basename = chunk_basename.split("_")[-1]
        output_chunk_temp_path = temp_dir / f"{OUTPUT_CHUNK_PREFIX}{original_chunk_number_from_basename}"

        # In dry-run, always "process" to show potential changes. Otherwise, check if processing is needed.
        needs_processing = True
        if not args.dry_run:
            if chunk_basename in processed_chunks_basenames and output_chunk_temp_path.exists():
                needs_processing = False
                log_debug(f"Chunk '{chunk_basename}' already processed and output exists. Skipping API call.")
            else:
                log_debug(f"Chunk '{chunk_basename}' needs processing or output is missing. Adding to queue.")
        
        if needs_processing:
            chunks_to_process_info.append((temp_chunk_path, output_chunk_temp_path))

    total_chunks = len(all_chunks_raw_paths)
    completed_count = total_chunks - len(chunks_to_process_info)
    log_info(f"Total fragments for '{original_file_path.name}': {total_chunks}. To process: {len(chunks_to_process_info)}.")

    if chunks_to_process_info:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_chunk_info = {
                executor.submit(process_chunk_with_api, orig_c, out_c, api_url, args, str(original_file_path)): (orig_c, out_c)
                for (orig_c, out_c) in chunks_to_process_info
            }
            for future in as_completed(future_to_chunk_info):
                original_c_path, _ = future_to_chunk_info[future]
                chunk_basename = original_c_path.name
                try:
                    result_tuple = future.result()
                    if result_tuple:
                        # result_tuple is (processed_content, original_content, chunk_path, input_tokens, output_tokens)
                        processed_content, _, _, in_tokens, out_tokens = result_tuple
                        
                        # Accumulate tokens even if reverted or skipped due to pre-check for cost estimation
                        total_input_tokens += in_tokens
                        total_output_tokens += out_tokens
                        
                        # Only mark as completed if processing was actually done (not skipped or failed early, indicated by processed_content is not None)
                        if processed_content is not None: 
                            completed_count += 1

                        if checkpoint_file and processed_content is not None: # Only write to checkpoint if not in dry-run and processing was successful
                            try:
                                with open(checkpoint_file, "a", encoding="utf-8") as cf:
                                    cf.write(f"{chunk_basename}\n")
                            except OSError as e:
                                log_error(f"Failed to write to checkpoint file '{checkpoint_file.name}': {e}")
                    else:
                        log_error(f"Fragment '{chunk_basename}' processing yielded no valid result. Check logs.")

                except Exception as exc:
                    log_error(f"Fragment '{chunk_basename}' failed unexpectedly during execution: {exc}")

                # Update progress bar
                progress = int(100 * completed_count / total_chunks) if total_chunks > 0 else 100
                bar_length = 50
                filled_length = int(bar_length * progress / 100)
                bar = "█" * filled_length + "-" * (bar_length - filled_length)
                # Display current file name in progress bar for directory processing
                current_file_display = original_file_path.name if args.input_dir else ""
                # Use stderr for progress bar to avoid interfering with stdout output redirection
                sys.stderr.write(f"\r{Fore.LIGHTBLUE_EX}{Style.BRIGHT}[PROGRESS]{Style.RESET_ALL} [{bar}] {progress}% ({completed_count}/{total_chunks} chunks for {current_file_display}) ")
                sys.stderr.flush()
        sys.stderr.write("\n") # Newline after progress bar for clean output
    else:
        log_info(f"No new chunks needed processing for '{original_file_path.name}'.")

    # After all processing (or if no chunks needed processing), reassemble
    reassemble_output(all_chunks_raw_paths, output_file_path, temp_dir, str(output_file_path) == "-", args.interactive, args.dry_run)

    # Banish the checkpoint file only if the file processing was fully completed without critical errors
    # and it's not a dry run.
    if checkpoint_file and checkpoint_file.exists() and not args.dry_run:
        # Verify if all chunks were processed or skipped successfully. A simple check for existence is done earlier.
        # A more robust check would compare the number of entries in the checkpoint with total_chunks.
        try:
            checkpoint_file.unlink() # Banish the checkpoint upon successful completion of the file
            log_debug(f"Checkpoint '{checkpoint_file.name}' banished.")
        except OSError as e:
            log_warning(f"Could not banish checkpoint '{checkpoint_file.name}': {e}")
def main():
    """Orchestrates the grand ritual of text enhancement."""
    global total_input_tokens, total_output_tokens, raw_api_output_mode

    parser = argparse.ArgumentParser(
        description=f"{Fore.LIGHTCYAN_EX}{Style.BRIGHT}Pyrmethus's Grand Gemini File Review Spell{Style.RESET_ALL}",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f"""
{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Configuration Priority:{Style.RESET_ALL}
  Command-line arguments > .env file (in current directory) > Environment variables > Config File ({CONFIG_FILE}) > Defaults.
{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Examples:{Style.RESET_ALL}
  {SCRIPT_NAME} -i script.py -o enhanced.py --interactive
  cat script.sh | {SCRIPT_NAME} -l bash --pre-check -o enhanced.sh
  {SCRIPT_NAME} --input-dir ./project --output-dir ./project-fixed --lang python -j 8
  {SCRIPT_NAME} -i document.md --max-chunk-tokens 2000 --custom-prompt "Fix grammar: {{original_code}} For language: {{lang_hint}}"
  {SCRIPT_NAME} -i myscript.js -o myscript_fixed.js --log-level DEBUG # Enable detailed debugging
  {SCRIPT_NAME} --input-dir code --output-dir processed --skip-ext .log .txt # Skip specific file types
  {SCRIPT_NAME} -i my_code.py --dry-run # Preview changes without writing files
  {SCRIPT_NAME} -i empty_file.txt # Process an empty file (will result in an empty output)

{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}Dependencies:{Style.RESET_ALL}
  Install Python packages: `pip install colorama requests python-dotenv`
  Optional pre-check tools: `pip install ruff` (for Python), `apt install shellcheck` (for Termux/Debian)
"""
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-file", type=Path, help="Input file path or '-' for stdin.")
    input_group.add_argument("--input-dir", type=Path, help="Input directory to process all files within.")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-o", "--output-file", type=Path, help="Output file path. If omitted for a single input file, output goes to stdout.")
    output_group.add_argument("--output-dir", type=Path, help="Output directory for processed files. Required when using --input-dir.")
    parser.add_argument("--stdout", action="store_true", help="Print output to stdout (overrides -o and --output-dir for the main output).")

    parser.add_argument("-k", "--key", help="Gemini API Key. If not provided, it will be prompted or read from environment/config.")
    parser.add_argument("-m", "--model", help=f"Gemini model to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("-t", "--temperature", type=float, help=f"API temperature for generation (default: {DEFAULT_TEMPERATURE}). Controls randomness.")
    parser.add_argument("-l", "--lang", help="Language hint for the prompt (e.g., python, javascript, bash). Overrides auto-detection.")
    parser.add_argument("--custom-prompt-template", help="Custom prompt template. Use {original_code} for the chunk content and {lang_hint} for language (e.g., 'Review this {{lang_hint}} code: {{original_code}}').")
    parser.add_argument("-j", "--jobs", type=int, help=f"Maximum number of parallel API requests (default: {DEFAULT_MAX_JOBS}). Adjust based on your network and API rate limits.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite of existing output file(s) without prompting.")
    parser.add_argument("--raw", action="store_true", help="Output raw API response text directly, bypassing code block extraction. Useful for debugging prompt engineering.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (alias for --log-level DEBUG).")
    parser.add_argument("--log-level", choices=log_level_map.keys(), default="INFO",
                        help=f"Set the logging level (default: INFO). Choices: {', '.join(log_level_map.keys())}.")
    parser.add_argument("--interactive", action="store_true", help="Interactively review changes for each chunk before applying. Requires user input.")
    parser.add_argument("--pre-check", action="store_true", help="Perform pre-API lint/style checks (requires tools like black/flake8/shellcheck). If pre-check passes on the original code, API call is skipped for that chunk.")
    parser.add_argument("--max-chunk-tokens", type=int, help="Maximum estimated tokens per chunk. Larger files will be split into smaller fragments to stay within this limit.")
    parser.add_argument("--connect-timeout", type=int, help=f"Connection timeout for API requests in seconds (default: {DEFAULT_CONNECT_TIMEOUT}).")
    parser.add_argument("--read-timeout", type=int, help=f"Read timeout for API responses in seconds (default: {DEFAULT_READ_TIMEOUT}).")
    parser.add_argument("--retries", type=int, help=f"Number of API retry attempts on transient errors (default: {MAX_RETRIES}).")
    parser.add_argument("--retry-delay", type=int, help=f"Seconds to wait between API retries (default: {RETRY_DELAY_SECONDS}).")
    parser.add_argument("--skip-ext", nargs="*", default=[], help="List of file extensions to skip when processing directories (e.g., .log .txt .bak). Case-insensitive.")
    parser.add_argument("--dry-run", action="store_true", help="Perform all operations but do not write any files; shows what would change. Useful for previewing.")
    parser.add_argument("--temp-dir", type=Path, help="Specify a custom directory for temporary files (e.g., for chunk splitting).")

    args = parser.parse_args()

    # Set logger level based on verbose or log-level argument
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level_map.get(args.log_level.upper(), logging.INFO)) # Use .get for safety

    # --- Resolve Settings ---
    config = load_config()
    raw_api_output_mode = args.raw  # Set the global flag

    # Priority: CLI > Env (including .env) > Config > Default
    args.model = args.model if args.model is not None else os.getenv("GEMINI_MODEL") or config.get("Gemini", "model", fallback=DEFAULT_MODEL)
    args.temperature = args.temperature if args.temperature is not None else float(os.getenv("GEMINI_TEMPERATURE", config.get("Gemini", "temperature", fallback=str(DEFAULT_TEMPERATURE))))
    args.jobs = args.jobs if args.jobs is not None else int(os.getenv("PYRMETHUS_JOBS", config.get("Pyrmethus", "jobs", fallback=str(DEFAULT_MAX_JOBS))))
    args.connect_timeout = args.connect_timeout if args.connect_timeout is not None else int(os.getenv("PYRMETHUS_CONNECT_TIMEOUT", config.get("Pyrmethus", "connect_timeout", fallback=str(DEFAULT_CONNECT_TIMEOUT))))
    args.read_timeout = args.read_timeout if args.read_timeout is not None else int(os.getenv("PYRMETHUS_READ_TIMEOUT", config.get("Pyrmethus", "read_timeout", fallback=str(DEFAULT_READ_TIMEOUT))))
    args.retries = args.retries if args.retries is not None else int(os.getenv("PYRMETHUS_RETRIES", config.get("Pyrmethus", "retries", fallback=str(MAX_RETRIES))))
    args.retry_delay = args.retry_delay if args.retry_delay is not None else int(os.getenv("PYRMETHUS_RETRY_DELAY", config.get("Pyrmethus", "retry_delay", fallback=str(RETRY_DELAY_SECONDS))))
    args.skip_ext = [ext.lower() for ext in args.skip_ext]  # Ensure extensions are lowercase for consistent checking

    # If --stdout is used, it takes precedence for output destination.
    if args.stdout:
        args.output_file = Path("-")
        args.output_dir = None  # Clear output_dir if stdout is forced
        log_debug("Output redirected to stdout.")

    # --- Setup ---
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    check_dependencies()
    get_api_key(args.key, config)
    if not gemini_api_key:  # Re-check after attempting all methods to get key
        log_error("Gemini API key is required to proceed.")
        sys.exit(1)

    if not create_temp_dir(args):
        sys.exit(1)

    files_to_process: list[tuple[Path, Path]] = []  # List of (input_path, output_path) tuples

    if args.input_file:
        is_stdin = str(args.input_file) == "-"
        if is_stdin:
            if sys.stdin.isatty():
                log_error("Cannot read from a TTY. Please pipe input to the script (e.g., `cat file.txt | python script.py`) or use `-i <file_path>`.")
                sys.exit(1)
            log_info(f"{Fore.LIGHTBLUE_EX}Reading from stdin...{Style.RESET_ALL}")
            stdin_content = sys.stdin.read()
            if not stdin_content.strip():
                log_warning("Input from stdin is empty. Nothing to process.")
                sys.exit(0) # Exit gracefully if stdin is empty

            stdin_tmp_path = temp_dir / "stdin.tmp"
            try:
                stdin_tmp_path.write_text(stdin_content, encoding="utf-8")
            except OSError as e:
                log_error(f"Failed to write stdin content to temporary file: {e}")
                sys.exit(1)
            
            # Determine the output path for stdin processing
            output_path_for_stdin = Path("-") if args.output_file is None and args.output_dir is None and not args.stdout else args.output_file
            if args.stdout:
                output_path_for_stdin = Path("-")
            
            files_to_process.append((stdin_tmp_path, output_path_for_stdin))
        else:
            if not args.input_file.is_file():
                log_error(f"Input file '{args.input_file}' not found or is not a file.")
                sys.exit(1)

            output_path_for_file = args.output_file
            # If no output file or dir is specified, and stdout is not forced, default to stdout.
            if output_path_for_file is None and args.output_dir is None and not args.stdout:
                log_warning(f"No output file specified for '{args.input_file}'. Output will be printed to stdout by default. Use -o to specify a file.")
                output_path_for_file = Path("-") # Force stdout

            # If an output file is specified (even if it's stdout), check for backup if it exists and not a dry run
            if output_path_for_file and str(output_path_for_file) != "-":
                if not backup_output_file(output_path_for_file, args.force, args.dry_run):
                    sys.exit(1) # Aborted by user during backup prompt
            
            files_to_process.append((args.input_file, output_path_for_file if output_path_for_file else Path("-")))

        if not args.lang and not is_stdin:
            inferred_lang = get_file_type_from_extension(args.input_file)
            if inferred_lang:
                args.lang = inferred_lang
                log_info(f"Inferred language '{args.lang}' from file extension for '{args.input_file.name}'.")
            else:
                log_warning(f"Could not infer language for '{args.input_file.name}'. Specify with -l for better results.")

    elif args.input_dir:
        if not args.input_dir.is_dir():
            log_error(f"Input directory '{args.input_dir}' not found or is not a directory.")
            sys.exit(1)
        if not args.output_dir:
            log_error("--output-dir is required when using --input-dir.")
            sys.exit(1)
        if args.output_dir.resolve() == args.input_dir.resolve():
            log_error(f"{Fore.LIGHTRED_EX}{Style.BRIGHT}Input and output directories cannot be the same to prevent data loss!{Style.RESET_ALL}")
            sys.exit(1)

        if not args.dry_run:  # Only create output dir if not in dry-run mode
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                log_debug(f"Ensured output directory exists: {args.output_dir}")
            except OSError as e:
                log_error(f"Failed to create output directory '{args.output_dir}': {e}")
                sys.exit(1)
        
        files_to_process = get_files_from_input_dir(args.input_dir, args.output_dir, args.force, args.skip_ext)
        if not files_to_process:
            # Log a warning only if no files were found or all were skipped
            log_warning(f"No processable files found in '{args.input_dir}' that match the criteria.")
            sys.exit(0)

    api_url = f"{API_BASE_URL}/{args.model}:generateContent?key={gemini_api_key}"
    files_processed_count = 0

    try:
        for original_file_path, output_file_path in files_to_process:
            process_single_file_workflow(original_file_path, output_file_path, api_url, args)
            files_processed_count += 1

    finally:
        cleanup_temp_dir()
        # Display summary only if at least one file was attempted
        if files_processed_count > 0 or (args.input_file and str(args.input_file) == "-"):
            display_final_summary(files_processed_count, args.dry_run)
        elif args.input_dir and not files_to_process:
             # If input_dir was used but no files were found, the summary is implicitly handled by the warning above.
             pass 
        else:
            log_warning("No files were processed in this ritual.")
if __name__ == "__main__":
    main()

