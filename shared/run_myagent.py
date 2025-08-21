import argparse
import asyncio
import ast
import itertools
import json
import logging
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import zipfile
from asyncio import PriorityQueue
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Callable

import psutil
from colorama import Back, Fore, Style, init
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter, WordCompleter
from prompt_toolkit.history import FileHistory
from tqdm.asyncio import tqdm_asyncio

# --- Initialization ---
init(autoreset=True)  # Initialize colorama for cross-platform terminal colors

# --- Constants ---
# Define constants for task priorities and statuses for better readability and maintainability.
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TASK_TYPE_SHELL = "shell"
TASK_TYPE_CODE = "code"
TASK_TYPE_AGENT = "agent"
TASK_TYPE_SCRIPT = "script"

DEPENDENCY_MODE_STRICT = "strict"  # Task fails if dependency fails/is cancelled.
DEPENDENCY_MODE_SOFT = "soft"      # Task proceeds even if dependency fails/is cancelled, but waits for completion.

# --- Configuration Management ---

class ConfigManager:
    """Manages application configuration, loading from and saving to a JSON file,
    and allowing environment variable overrides.

    Attributes:
        DEFAULT_CONFIG (Dict[str, Any]): A dictionary containing the default configuration settings.
        COLORAMA_TO_PROMPT_TOOLKIT_STYLE (Dict[str, str]): Mapping from colorama styles to prompt_toolkit styles.
        config_file (str): The path to the configuration file.
        _config (Dict[str, Any]): The current configuration dictionary.
    """

    DEFAULT_CONFIG = {
        "default_agent": "pyrm",
        "history_file": os.path.join(os.path.expanduser("~"), ".pyrmethus_history.json"),
        "session_file": os.path.join(os.path.expanduser("~"), ".pyrmethus_session.json"),
        "prompt_history_file": os.path.join(os.path.expanduser("~"), ".pyrmethus_prompt_history"),
        "backup_dir": os.path.join(os.path.expanduser("~"), "pyrmethus_backups"),
        "log_file": "pyrmethus.log",
        "max_history_entries": 300,
        "task_timeout_seconds": 600,
        "retry_attempts": 3,
        "log_max_bytes": 5 * 1024 * 1024,  # 5 MB
        "log_backup_count": 5,
        "active_theme": "default",
        "auto_execute_agent_code": False,
        "default_priority": "medium",
        "default_dependency_mode": DEPENDENCY_MODE_STRICT,
        "custom_completions": [],
        "verbose_mode": True,  # Controls if output is printed to console (always logged).
        "enhance_files": False,
        "debug_files": False,
        "command_completions": [
            "ls", "cd", "pwd", "cat", "grep", "chmod", "touch", "mkdir", "rm", "cp",
            "mv", "git", "pytest", "black", "pylint", "pip", "pkg", "termux-info", "aichat",
            "status", "history", "exit", "help", "cancel", "queue", "settings", "export", "import",
            "enhance", "debug", "clear"
        ],
        "dangerous_commands": [
            'rm -rf /', 'mkfs', 'dd if=', ':(){ :|: & };:', 'sudo', 'format', 'wipe',
            'chown -R', 'chmod -R 777'
        ],
        "confirm_commands": ['rm', 'mv', 'dd', 'chown', 'chmod', 'chgrp', 'fdisk'],
        "languages": {
            "python": {"interpreter": [sys.executable], "extension": ".py", "linters": ["pylint"], "formatters": ["black"]},
            "javascript": {"interpreter": ["node"], "extension": ".js"},
            "bash": {"interpreter": ["bash"], "extension": ".sh"},
            "ruby": {"interpreter": ["ruby"], "extension": ".rb"},
            "php": {"interpreter": ["php"], "extension": ".php"}
        },
        "themes": {
            "default": {
                "success": Fore.GREEN, "error": Fore.RED, "info": Fore.CYAN,
                "prompt": Fore.BLUE, "warning": Fore.YELLOW, "progress": Fore.MAGENTA,
                "header": Fore.LIGHTYELLOW_EX, "muted": Fore.LIGHTBLACK_EX
            },
            "dark": {
                "success": Fore.LIGHTGREEN_EX, "error": Fore.RED, "info": Fore.LIGHTBLUE_EX,
                "prompt": Fore.MAGENTA, "warning": Fore.YELLOW, "progress": Fore.CYAN,
                "header": Fore.LIGHTCYAN_EX, "muted": Fore.WHITE
            },
            "hacker": {
                "success": Fore.GREEN, "error": Fore.RED, "info": Fore.GREEN,
                "prompt": Fore.GREEN, "warning": Fore.YELLOW, "progress": Fore.GREEN,
                "header": Fore.GREEN, "muted": Fore.LIGHTBLACK_EX
            }
        }
    }

    COLORAMA_TO_PROMPT_TOOLKIT_STYLE = {
        Fore.BLACK: 'ansiblack', Fore.RED: 'ansired', Fore.GREEN: 'ansigreen', Fore.YELLOW: 'ansiyellow',
        Fore.BLUE: 'ansiblue', Fore.MAGENTA: 'ansimagenta', Fore.CYAN: 'ansicyan', Fore.WHITE: 'ansiwhite',
        Fore.LIGHTBLACK_EX: 'ansigray', Fore.LIGHTRED_EX: 'ansired', Fore.LIGHTGREEN_EX: 'ansigreen',
        Fore.LIGHTYELLOW_EX: 'ansiyellow', Fore.LIGHTBLUE_EX: 'ansiblue', Fore.LIGHTMAGENTA_EX: 'ansimagenta',
        Fore.LIGHTCYAN_EX: 'ansicyan', Fore.LIGHTWHITE_EX: 'ansiwhite',
        Back.BLACK: 'bg:ansiblack', Back.RED: 'bg:ansired', Back.GREEN: 'bg:ansigreen', Back.YELLOW: 'bg:ansiyellow',
        Back.BLUE: 'bg:ansiblue', Back.MAGENTA: 'bg:ansimagenta', Back.CYAN: 'bg:ansicyan', Back.WHITE: 'bg:ansiwhite',
        Back.LIGHTBLACK_EX: 'bg:ansigray', Back.LIGHTRED_EX: 'bg:ansired', Back.LIGHTGREEN_EX: 'bg:ansigreen',
        Back.LIGHTYELLOW_EX: 'bg:ansiyellow', Back.LIGHTBLUE_EX: 'bg:ansiblue', Back.LIGHTMAGENTA_EX: 'bg:ansimagenta',
        Back.LIGHTCYAN_EX: 'bg:ansicyan', Back.LIGHTWHITE_EX: 'bg:ansiwhite',
    }

    def __init__(self, config_file: str = "config.json"):
        """Initializes ConfigManager, loading configuration from file or defaults."""
        self.config_file = os.path.join(os.path.expanduser("~"), config_file)
        self._config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Loads configuration, prioritizing user file, then environment variables, then defaults.
        Handles potential errors during file loading."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                self._config = {**self.DEFAULT_CONFIG, **user_config}
            else:
                self._config = self.DEFAULT_CONFIG.copy()
                self.save_config()  # Save default config if file doesn't exist
        except (json.JSONDecodeError, IOError) as e:
            print(f"{Fore.RED}Error loading config from {self.config_file}: {e}. Using default configuration.{Style.RESET_ALL}")
            self._config = self.DEFAULT_CONFIG.copy()

        # Override with environment variables (prefixed with PYRMETHUS_)
        for key, value in self._config.items():
            env_var_name = f"PYRMETHUS_{key.upper()}"
            if env_var_name in os.environ:
                env_value = os.environ[env_var_name]
                try:
                    # Attempt to parse as JSON for complex types (lists, dicts)
                    self._config[key] = json.loads(env_value)
                except json.JSONDecodeError:
                    self._config[key] = env_value  # Fallback to string if JSON parsing fails

    def save_config(self) -> None:
        """Saves the current configuration to the config file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4)
        except IOError as e:
            print(f"{Fore.RED}Failed to save configuration to {self.config_file}: {e}{Style.RESET_ALL}")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key, returning default if not found."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value and immediately saves the configuration."""
        self._config[key] = value
        self.save_config()

    def get_pt_style(self, colorama_color_obj) -> str:
        """Converts a colorama Fore/Back object to a prompt_toolkit style string."""
        return self.COLORAMA_TO_PROMPT_TOOLKIT_STYLE.get(colorama_color_obj, '')

# Instantiate the configuration manager globally.
CONFIG_MANAGER = ConfigManager()

# Load the active theme based on the configuration.
THEME = CONFIG_MANAGER.get("themes", {}).get(CONFIG_MANAGER.get("active_theme"), CONFIG_MANAGER.DEFAULT_CONFIG["themes"]["default"])

# --- Global State ---
task_queue: asyncio.PriorityQueue = PriorityQueue()
# Stores current status and metadata of all tasks.
task_status: Dict[str, Dict[str, Any]] = {}
# Stores dependencies for tasks: {task_id: [dependency_ids]}
task_dependencies: Dict[str, List[str]] = {}
# Simplified list of (id, type, priority) for quick queue viewing.
queued_tasks_list: List[Tuple[str, str, str]] = []
prompt_session: Optional[PromptSession] = None
# Context variable to track the current task being processed for logging.
current_task_id = contextvars.ContextVar('current_task_id', default=None)
# Event to signal graceful shutdown across all tasks.
shutdown_event = asyncio.Event()

# --- Logging Setup ---
def setup_logging() -> logging.Logger:
    """Configures the application logger with console and file handlers,
    including color formatting for console output.

    Returns:
        logging.Logger: The configured application logger instance.
    """
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Prevent default logging configuration from adding handlers if called multiple times.
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(logging.INFO) # Set initial level, can be overridden by config later.
    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    class ColorFormatter(logging.Formatter):
        """Custom formatter to add colors to log messages and include task ID context."""
        def format(self, record):
            task_id = current_task_id.get()
            if task_id:
                # Prepend task ID to the message for context.
                record.msg = f"[Task {task_id}] {record.msg}"

            log_message = super().format(record)

            # Apply color based on log level using the global THEME.
            level_colors = {
                logging.DEBUG: THEME.get('muted', Fore.WHITE),
                logging.INFO: THEME.get('info', Fore.CYAN),
                logging.WARNING: THEME.get('warning', Fore.YELLOW),
                logging.ERROR: THEME.get('error', Fore.RED),
                logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
            }
            color = level_colors.get(record.levelno, Fore.WHITE)

            return f"{color}{log_message}{Style.RESET_ALL}" if color else log_message

    # Console Handler: Uses ColorFormatter for colored output.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    app_logger.addHandler(console_handler)

    # File Handler: Uses a standard formatter and RotatingFileHandler for log rotation.
    log_file = CONFIG_MANAGER.get("log_file")
    max_bytes = CONFIG_MANAGER.get("log_max_bytes")
    backup_count = CONFIG_MANAGER.get("log_backup_count")
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(log_formatter)  # File logs don't need color.
        app_logger.addHandler(file_handler)
    except IOError as e:
        # Fallback if log file cannot be set up.
        print(f"{THEME.get('error')}Could not set up log file '{log_file}': {e}. Logging only to console.{Style.RESET_ALL}")
        app_logger.warning(f"Could not set up log file '{log_file}'. Logging only to console.")

    return app_logger
logger = setup_logging()

# --- Utility Functions ---
def get_pt_style(colorama_color_obj) -> str:
    """Converts a colorama Fore/Back object to a prompt_toolkit style string."""
    return CONFIG_MANAGER.get_pt_style(colorama_color_obj)
def load_json_file(file_path: str, default: Any, description: str) -> Any:
    """Loads JSON data from a file, handling file not found or corruption gracefully.

    Args:
        file_path (str): The path to the JSON file.
        default (Any): The default value to return if the file is not found or corrupted.
        description (str): A description of the file for logging purposes (e.g., "config", "history").

    Returns:
        Any: The loaded JSON data or the default value.
    """
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Corrupted or unreadable {description} file at {file_path}: {e}. Attempting backup and returning default.")
        # Attempt to backup corrupted file before returning default.
        try:
            backup_path = f"{file_path}.corrupted_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backed up corrupted {description} file to {backup_path}")
        except IOError as backup_e:
            logger.warning(f"Failed to backup corrupted {description} file: {backup_e}")
        return default
def save_json_file(file_path: str, data: Any, description: str) -> None:
    """Saves data to a JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logger.error(f"Failed to save {description} to {file_path}: {e}")
def save_history_entry(entry: Dict) -> None:
    """Appends a new entry to the history file, truncating if necessary to respect max_history_entries."""
    history_file = CONFIG_MANAGER.get("history_file")
    history = load_json_file(history_file, [], "history")
    history.append(entry)
    # Keep only the latest entries based on config.
    max_entries = CONFIG_MANAGER.get("max_history_entries")
    if len(history) > max_entries:
        history = history[-max_entries:]
    save_json_file(history_file, history, "history")
def backup_file(file_path: str) -> None:
    """Creates a timestamped backup of a given file if it exists."""
    if not os.path.exists(file_path):
        return
    backup_dir = CONFIG_MANAGER.get("backup_dir")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak")
    try:
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")
    except IOError as e:
        logger.error(f"Failed to backup {file_path}: {e}")
def create_task_id() -> str:
    """Generates a unique task ID using a timestamp and random hex characters."""
    return f"T-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(3).hex().upper()}"
async def stream_process_output(process: asyncio.subprocess.Process, task_id: str, verbose_mode: bool) -> str:
    """Asynchronously streams stdout and stderr from a subprocess, logging and optionally printing.

    Args:
        process (asyncio.subprocess.Process): The subprocess object.
        task_id (str): The ID of the task being processed.
        verbose_mode (bool): Whether to print output to the console.

    Returns:
        str: The combined stdout and stderr output as a single string.
    """
    output_buffer = []
    spinner_chars = itertools.cycle(['-', '\\', '|', '/'])

    async def _read_stream(stream: Optional[asyncio.StreamReader], stream_type: str):
        """Reads lines from a stream, logs them, and appends to the buffer."""
        if stream is None: return
        while True:
            try:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors='ignore').strip()
                if line:
                    log_color = THEME['info'] if stream_type == "stdout" else THEME['warning']
                    logger.info(f"[{task_id} {stream_type.upper()}] {line}")
                    if verbose_mode:
                        print(f"{log_color}[{stream_type.upper()}] {line}{Style.RESET_ALL}")
                    output_buffer.append(line)
            except asyncio.CancelledError:
                break # Allow cancellation.
            except Exception as e:
                logger.error(f"Error reading {stream_type} for task {task_id}: {e}")
                break # Stop reading on error.

    # Task to display a spinner while the process is running, if verbose mode is enabled.
    spinner_task = None
    if verbose_mode and sys.stdout.isatty():
        async def show_spinner():
            while True:
                print(f"\r{THEME['progress']}[Task {task_id}] Running... {next(spinner_chars)}{Style.RESET_ALL}", end="", flush=True)
                await asyncio.sleep(0.1)
        spinner_task = asyncio.create_task(show_spinner())

    try:
        # Gather output from stdout and stderr concurrently.
        await asyncio.gather(
            _read_stream(process.stdout, "stdout"),
            _read_stream(process.stderr, "stderr")
        )
    finally:
        if spinner_task:
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass
            # Clear the spinner line.
            print("\r" + " " * 60 + "\r", end="", flush=True)

    return "\n".join(output_buffer)
async def _run_code_enhancements(task_id: str, code: Optional[str], file_path: Optional[str], language: str, verbose_mode: bool, env: Dict, mode: str = "") -> None:
    """
    Helper function to run code enhancements (linting, formatting) or debugging.
    Mode can be "enhance", "debug", or empty (for general task execution).

    Args:
        task_id (str): The ID of the current task.
        code (Optional[str]): The code snippet if available.
        file_path (Optional[str]): The path to the file if available.
        language (str): The programming language of the code/file.
        verbose_mode (bool): Whether to print detailed output.
        env (Dict): The environment variables for subprocesses.
        mode (str): The mode of operation ('enhance' or 'debug').
    """
    lang_config = CONFIG_MANAGER.get('languages', {}).get(language)
    if not lang_config:
        logger.warning(f"No language configuration found for '{language}'. Skipping enhancements/debugging.")
        return

    # --- Debugging Output ---
    if mode == "debug" or CONFIG_MANAGER.get("debug_files"):
        print(f"\n{THEME['info']}--- Debugging Info for Task {task_id} ({language}) ---")
        if file_path:
            print(f"{THEME['muted']}File Path: {file_path}")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        print(f"{THEME['muted']}File Content:\n{f.read()}")
                except Exception as e:
                    print(f"{THEME['error']}Could not read file content: {e}")
            else:
                print(f"{THEME['error']}File does not exist.")
        elif code:
            print(f"{THEME['muted']}Code Content:\n{code}")

        if language == "bash" and code:
            logger.info(f"Prepending 'set -x' for bash debugging for Task {task_id}.")
            # Note: Modifying code strings here is tricky. For actual execution,
            # the caller should prepend 'set -x' if debug_files is enabled.
            # This function only reports the intent.

        print(f"{THEME['info']}--- End Debugging Info ---")

    # --- Enhancement Checks ---
    if mode == "enhance" or CONFIG_MANAGER.get("enhance_files"):
        print(f"\n{THEME['info']}--- Code Enhancement Checks for Task {task_id} ({language}) ---")

        target_file = file_path
        temp_file_created = False
        if not target_file and code:
            # Create a temporary file for linting/formatting if only code is provided.
            temp_file_created = True
            extension = lang_config.get('extension', '.tmp')
            target_file = f"temp_enhance_{task_id}{extension}"
            try:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(code)
            except IOError as e:
                logger.error(f"Failed to create temp file for enhancement: {e}")
                print(f"{THEME['error']}Failed to create temp file for enhancement: {e}")
                return

        if target_file and os.path.exists(target_file):
            try:
                if language == "python":
                    formatters = lang_config.get('formatters', [])
                    linters = lang_config.get('linters', [])

                    for formatter in formatters:
                        if shutil.which(formatter):
                            print(f"{THEME['progress']}Running {formatter} on {target_file}...{Style.RESET_ALL}")
                            # Use --check and --diff for reporting changes without modifying.
                            process = await asyncio.create_subprocess_exec(
                                formatter, "--check", "--diff", target_file,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=env
                            )
                            stdout, stderr = await process.communicate()
                            if stdout:
                                print(f"{THEME['warning']}{formatter} Suggestions:\n{stdout.decode(errors='ignore')}")
                            if stderr:
                                print(f"{THEME['error']}{formatter} Error:\n{stderr.decode(errors='ignore')}")
                            if process.returncode != 0 and stdout:
                                print(f"{THEME['warning']}{formatter} found formatting issues. Consider running '{formatter} {target_file}'.")
                            else:
                                print(f"{THEME['success']}{formatter} found no formatting issues.")
                        else:
                            print(f"{THEME['muted']}{formatter} not found. Skipping formatting check.")

                    for linter in linters:
                        if shutil.which(linter):
                            print(f"{THEME['progress']}Running {linter} on {target_file}...{Style.RESET_ALL}")
                            process = await asyncio.create_subprocess_exec(
                                linter, target_file,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=env
                            )
                            stdout, stderr = await process.communicate()
                            if stdout:
                                print(f"{THEME['warning']}{linter} Report:\n{stdout.decode(errors='ignore')}")
                            if stderr:
                                print(f"{THEME['error']}{linter} Error:\n{stderr.decode(errors='ignore')}")
                            if process.returncode != 0:
                                print(f"{THEME['warning']}{linter} found linting issues.")
                            else:
                                print(f"{THEME['success']}{linter} found no linting issues.")
                        else:
                            print(f"{THEME['muted']}{linter} not found. Skipping linting check.")
                else:
                    print(f"{THEME['muted']}No specific enhancement tools configured for '{language}'.")

            finally:
                # Clean up the temporary file if it was created.
                if temp_file_created and target_file and os.path.exists(target_file):
                    try:
                        os.remove(target_file)
                    except OSError as e:
                        logger.warning(f"Failed to remove temporary enhancement file {target_file}: {e}")
        else:
            print(f"{THEME['error']}Target file for enhancement '{target_file}' not found.")

        print(f"{THEME['info']}--- End Code Enhancement Checks ---")
async def execute_task(task_id: str, task_type: str, args: Tuple, verbose_mode: bool, retries: int, timeout: int, env_vars: Dict) -> None:
    """Executes a given task (shell, code, agent, script) with retries and timeout.

    Manages task status updates, context variables for logging, and exception handling.
    """
    token = current_task_id.set(task_id)  # Set context variable for logging.
    try:
        task_status[task_id]["status"] = STATUS_RUNNING
        task_status[task_id]["start_time"] = datetime.now().isoformat()
        logger.info(f"Executing Task {task_id} ({task_type}) with args: {args}")

        # Merge environment variables: CLI args > Task args > System env.
        env = os.environ.copy()
        env.update(env_vars)

        # Task execution logic with retries and timeout.
        for attempt in range(retries + 1):
            try:
                if task_type == TASK_TYPE_SHELL:
                    await execute_shell_command(task_id, args[0], verbose_mode, env)
                elif task_type == TASK_TYPE_CODE:
                    await execute_code(task_id, args[0], args[1], verbose_mode, env)
                elif task_type == TASK_TYPE_AGENT:
                    agent_name, prompt = args[0], args[1]
                    tool_code = args[2] if len(args) > 2 else None
                    await summon_agent(task_id, agent_name, prompt, tool_code, verbose_mode, env)
                elif task_type == TASK_TYPE_SCRIPT:
                    await execute_script(task_id, args[0], args[1], verbose_mode, env)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")

                task_status[task_id]["status"] = STATUS_COMPLETED
                logger.info(f"Task {task_id} completed successfully.")
                break  # Exit retry loop on success.
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
                logger.error(f"Attempt {attempt + 1}/{retries + 1} for Task {task_id} failed: {e}")
                if attempt == retries:
                    task_status[task_id]["status"] = STATUS_FAILED
                    task_status[task_id]["comment"] = f"Task failed after {retries} retries: {e}"
                    logger.critical(f"Task {task_id} failed after {retries} retries: {e}")
                    raise  # Re-raise on final failure.
                # Exponential backoff for retries.
                await asyncio.sleep(min(2 ** attempt, 10)) # Cap sleep time.
            except asyncio.CancelledError:
                # Task was cancelled externally while running.
                task_status[task_id]["status"] = STATUS_CANCELLED
                task_status[task_id]["comment"] = "Task cancelled during execution."
                logger.warning(f"Task {task_id} was cancelled during execution.")
                raise # Re-raise to be caught by the outer handler.
            except asyncio.TimeoutError:
                task_status[task_id]["status"] = STATUS_FAILED
                task_status[task_id]["comment"] = f"Task timed out after {timeout} seconds."
                logger.error(f"Task {task_id} timed out after {timeout} seconds.")
                raise # Re-raise to be caught by the outer handler.
            except Exception as e:
                # Catch any other unexpected errors.
                task_status[task_id]["status"] = STATUS_FAILED
                task_status[task_id]["comment"] = f"Unhandled exception during execution: {e}"
                logger.critical(f"Unhandled error in Task {task_id} during execution: {e}", exc_info=True)
                raise # Re-raise to be caught by the outer handler.

    except asyncio.CancelledError:
        # Task was cancelled before or during the retry loop setup.
        if task_id in task_status and task_status[task_id]["status"] != STATUS_CANCELLED:
            task_status[task_id]["status"] = STATUS_CANCELLED
            task_status[task_id]["comment"] = "Task cancelled externally."
        logger.warning(f"Task {task_id} was cancelled.")
    except asyncio.TimeoutError:
        # Task timed out during execution.
        if task_id in task_status and task_status[task_id]["status"] != STATUS_FAILED:
            task_status[task_id]["status"] = STATUS_FAILED
            task_status[task_id]["comment"] = f"Task timed out after {timeout} seconds."
        logger.error(f"Task {task_id} timed out.")
    except Exception as e:
        # Task failed with an error.
        if task_id in task_status and task_status[task_id]["status"] != STATUS_FAILED:
            task_status[task_id]["status"] = STATUS_FAILED
            task_status[task_id]["comment"] = f"Task failed: {e}"
        logger.critical(f"Task {task_id} failed with an unhandled exception: {e}", exc_info=True)
    finally:
        # Update task completion metadata.
        end_time = datetime.now()
        start_time_iso = task_status[task_id].get("start_time")
        if start_time_iso:
            duration = (end_time - datetime.fromisoformat(start_time_iso)).total_seconds()
            task_status[task_id]["duration"] = f"{duration:.2f}s"
        task_status[task_id]["end_time"] = end_time.isoformat()

        # Termux notification if applicable.
        try:
            import termux
            if hasattr(termux, 'toast'):
                status_msg = task_status[task_id]['status'].capitalize()
                comment_snippet = (task_status[task_id].get('comment') or '').split('\n')[0][:50]
                toast_msg = f"Task {task_id} {status_msg}"
                if comment_snippet:
                    toast_msg += f": {comment_snippet}..." if len(comment_snippet) == 50 else f": {comment_snippet}"
                termux.toast(toast_msg)
            else:
                logger.warning("termux.toast not found. Skipping toast notification.")
        except (ImportError, FileNotFoundError):
            pass # Not on Termux or termux-api not installed.

        current_task_id.reset(token)  # Reset context variable.

async def execute_shell_command(task_id: str, command: str, verbose_mode: bool, env: Dict) -> None:
    """Executes a shell command with safety checks (dangerous commands, confirmations)."""
    normalized_command = command.strip().lower()

    # Dangerous command check.
    dangerous_commands = CONFIG_MANAGER.get('dangerous_commands', [])
    if any(dangerous in normalized_command for dangerous in dangerous_commands):
        logger.error(f"Execution of dangerous command '{command}' blocked for task {task_id}.")
        print(f"{THEME['error']}Dangerous command '{command}' blocked!{Style.RESET_ALL}")
        raise ValueError("Dangerous command blocked.")

    # Confirmation for sensitive commands.
    confirm_commands = CONFIG_MANAGER.get('confirm_commands', [])
    if any(cmd_check in normalized_command for cmd_check in confirm_commands):
        if prompt_session: # Only prompt if interactive.
            confirm = await get_user_input(
                f"{THEME['warning']}Are you sure you want to execute '{command}'? (yes/no): {Style.RESET_ALL}"
            )
            if confirm.lower() != 'yes':
                logger.info(f"Execution of command '{command}' cancelled by user for task {task_id}.")
                raise asyncio.CancelledError("User cancelled execution.")
        else:
            # In non-interactive mode, sensitive commands might be blocked or require explicit override.
            logger.warning(f"Sensitive command '{command}' encountered in non-interactive mode for task {task_id}. Proceeding without confirmation.")

    # Determine if shell=True is needed (for pipes, redirects, etc.).
    use_shell = bool(re.search(r'[|&;<>()`$*?!#~=]', command))
    if use_shell:
        logger.warning(f"Task {task_id}: Command contains shell-specific characters. Executing with shell=True. Command: '{command}'")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
    else:
        # Split command into arguments for create_subprocess_exec using shlex for safety.
        try:
            cmd_parts = shlex.split(command)
        except ValueError as e:
            logger.warning(f"Could not parse command with shlex.split, falling back to simple split: {e}")
            cmd_parts = command.split()

        if not cmd_parts:
            raise ValueError("Empty command after splitting.")

        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

    # Stream output and wait for process completion.
    stdout_output = await stream_process_output(process, task_id, verbose_mode=verbose_mode)
    await process.wait()

    # Log history entry.
    history_entry = {
        "task_id": task_id, "timestamp": datetime.now().isoformat(), "type": TASK_TYPE_SHELL,
        "command": command, "stdout": stdout_output, "return_code": process.returncode
    }
    save_history_entry(history_entry)

    # Raise error if command failed.
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command, stdout_output)

async def execute_code(task_id: str, code: str, language: str, verbose_mode: bool, env: Dict) -> None:
    """Executes a code snippet in the specified language."""
    lang_config = CONFIG_MANAGER.get('languages', {}).get(language)
    if not lang_config:
        raise ValueError(f"Unsupported language: '{language}'. Supported: {list(CONFIG_MANAGER.get('languages', {}).keys())}")

    interpreter = lang_config['interpreter']
    extension = lang_config['extension']
    # Create a unique filename per task to avoid conflicts.
    filename = f"temp_script_{task_id}{extension}"

    if not code.strip():
        logger.warning(f"Task {task_id}: Empty code block for language '{language}'. Skipping execution.")
        return

    # Run enhancements/debugging before execution.
    await _run_code_enhancements(task_id, code=code, file_path=None, language=language, verbose_mode=verbose_mode, env=env)

    # Prepend 'set -x' for bash if debugging is enabled.
    code_to_execute = code
    if language == "bash" and CONFIG_MANAGER.get("debug_files"):
        code_to_execute = "set -x\n" + code_to_execute

    try:
        # Backup existing temp file if it somehow exists (unlikely with unique IDs).
        backup_file(filename) if os.path.exists(filename) else None
        # Write code to the temporary file.
        with open(filename, "w", encoding='utf-8') as f:
            f.write(code_to_execute)

        # Ensure the script is executable if it's a bash/shell script.
        if language == "bash":
            os.chmod(filename, 0o755)

        # Execute the script using the appropriate interpreter.
        process = await asyncio.create_subprocess_exec(
            *interpreter, filename,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout_output = await stream_process_output(process, task_id, verbose_mode=verbose_mode)
        await process.wait()

        # Log history entry.
        history_entry = {
            "task_id": task_id, "timestamp": datetime.now().isoformat(), "type": TASK_TYPE_CODE,
            "language": language, "code": code, "stdout": stdout_output, "return_code": process.returncode
        }
        save_history_entry(history_entry)

        # Raise error if script failed.
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, " ".join(interpreter) + f" {filename}", stdout_output)

    finally:
        # Clean up the temporary file.
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError as e:
                logger.warning(f"Failed to remove temporary script file {filename}: {e}")

async def execute_script(task_id: str, script_path: str, language: str, verbose_mode: bool, env: Dict) -> None:
    """Executes an external script file."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script file not found: '{script_path}'")

    # Ensure script is executable.
    if not os.access(script_path, os.X_OK):
        if language in ["bash", "python"]: # Common script types that benefit from executable permission.
            logger.warning(f"Script '{script_path}' is not executable. Attempting to set executable permission.")
            try:
                os.chmod(script_path, 0o755)
            except OSError as e:
                logger.error(f"Failed to set executable permission on '{script_path}': {e}")
                raise ValueError("Script is not executable and permission could not be set.")
        else:
            # For other languages, we might rely on the interpreter directly.
            logger.warning(f"Script '{script_path}' is not executable. Execution may fail if interpreter requires it.")

    lang_config = CONFIG_MANAGER.get('languages', {}).get(language)
    if not lang_config:
        raise ValueError(f"Unsupported language: '{language}'. Supported: {list(CONFIG_MANAGER.get('languages', {}).keys())}")

    # Run enhancements/debugging before execution.
    await _run_code_enhancements(task_id, code=None, file_path=script_path, language=language, verbose_mode=verbose_mode, env=env)

    # Backup the script before execution.
    backup_file(script_path)
    interpreter = lang_config['interpreter']

    # Execute the script.
    process = await asyncio.create_subprocess_exec(
        *interpreter, script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    stdout_output = await stream_process_output(process, task_id, verbose_mode=verbose_mode)
    await process.wait()

    # Log history entry.
    history_entry = {
        "task_id": task_id, "timestamp": datetime.now().isoformat(), "type": TASK_TYPE_SCRIPT,
        "language": language, "script_path": script_path, "stdout": stdout_output, "return_code": process.returncode
    }
    save_history_entry(history_entry)

    # Raise error if script failed.
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, " ".join(interpreter) + f" {script_path}", stdout_output)

async def process_agent_code_blocks(parent_task_id: str, agent_output: str, verbose_mode: bool):
    """Parses code blocks from agent output and offers to execute them.

    Args:
        parent_task_id (str): The ID of the agent task that generated the output.
        agent_output (str): The raw output from the AI agent.
        verbose_mode (bool): Whether to display prompts and code blocks to the user.
    """
    # Regex to find code blocks: ```[language]\n[code]\n```
    code_blocks = re.findall(r"```(\w*)\s*\n(.*?)(?:```|$)", agent_output, re.DOTALL)

    if not code_blocks:
        logger.info(f"No code blocks found in agent response for task {parent_task_id}.")
        return

    logger.info(f"Agent response contains {len(code_blocks)} code block(s).")
    auto_execute = CONFIG_MANAGER.get('auto_execute_agent_code', False)

    for i, (lang, code) in enumerate(code_blocks):
        lang = lang.lower().strip() or "python"  # Default to python if language not specified.
        code = code.strip()

        if not code or lang not in CONFIG_MANAGER.get('languages'):
            logger.warning(f"Skipping invalid or unsupported code block (lang: '{lang}', code snippet: '{code[:50]}...') from agent for task {parent_task_id}.")
            continue

        confirm_action = 'yes' if auto_execute else ''

        if not auto_execute and prompt_session:
            print(f"\n{THEME['prompt']}--- Agent Suggested Code Block {i+1} ({lang}) ---")
            print(f"{THEME['muted']}{code}")
            print(f"{THEME['prompt']}----------------------------------")
            confirm_action = await get_user_input(
                f"{THEME['warning']}Execute this code block? (yes/no/edit): {Style.RESET_ALL}"
            )

        if confirm_action.lower() == 'yes':
            logger.info(f"Executing agent suggested code block (Task {parent_task_id}, Block {i+1}).")
            await add_task_to_queue(
                (TASK_TYPE_CODE, (code, lang)),
                priority_str="high",  # Agent-suggested code usually implies high importance.
                dependencies=[parent_task_id],
                verbose_mode=verbose_mode,
                comment=f"Agent suggested code from {parent_task_id}"
            )
        elif confirm_action.lower() == 'edit' and prompt_session:
            edited_code = await get_user_input(
                "Edit code:", default=code, multiline=True
            )
            if edited_code.strip():
                logger.info(f"Executing edited agent code block (Task {parent_task_id}, Block {i+1}).")
                await add_task_to_queue(
                    (TASK_TYPE_CODE, (edited_code, lang)),
                    priority_str="high",
                    dependencies=[parent_task_id],
                    verbose_mode=verbose_mode,
                    comment=f"Edited agent suggested code from {parent_task_id}"
                )
            else:
                logger.info(f"Edited code was empty. Skipping execution for block {i+1}.")
        else:
            logger.info(f"Skipping execution of agent suggested code block (Task {parent_task_id}, Block {i+1}).")

async def summon_agent(task_id: str, agent_name: str, prompt: str, tool_code: Optional[str], verbose_mode: bool, env: Dict) -> None:
    """Interacts with an AI agent (aichat) and processes its responses.

    Args:
        task_id (str): The ID of the task.
        agent_name (str): The name of the AI agent to use.
        prompt (str): The query prompt for the agent.
        tool_code (Optional[str]): Optional tool code to provide to the agent.
        verbose_mode (bool): Whether to display agent output to the console.
        env (Dict): Environment variables for the 'aichat' subprocess.
    """
    if not await check_network_connectivity():
        logger.error(f"Task {task_id}: No network connectivity detected for agent task.")
        raise ValueError("Network connectivity required for agent tasks.")

    command = ["aichat", "--agent", agent_name, prompt]
    if tool_code:
        command.extend(["--tool-code", tool_code])

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        full_output = await stream_process_output(process, task_id, verbose_mode=verbose_mode)
        await process.wait()

        # Log history entry.
        save_history_entry({
            "task_id": task_id, "timestamp": datetime.now().isoformat(), "type": TASK_TYPE_AGENT,
            "agent": agent_name, "prompt": prompt, "tool_code": tool_code,
            "stdout": full_output, "return_code": process.returncode
        })

        # Raise error if agent command failed.
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, " ".join(command), full_output)

        # Process any code blocks returned by the agent.
        await process_agent_code_blocks(task_id, full_output, verbose_mode)

    except FileNotFoundError:
        logger.error(f"Task {task_id}: 'aichat' command not found. Please ensure 'aichat' is installed and in your PATH.")
        raise ValueError("'aichat' command not found.")
    except Exception as e:
        logger.error(f"Error while summoning agent for task {task_id}: {e}", exc_info=True)
        raise

# --- Task Queue Management ---

class PrioritizedItem:
    """Wrapper class for items in asyncio.PriorityQueue to allow custom comparison."""
    def __init__(self, priority: int, item: Any):
        self.priority = priority
        self.item = item

    def __lt__(self, other: 'PrioritizedItem') -> bool:
        """Lower priority value means higher priority (e.g., 1 is higher than 2)."""
        return self.priority < other.priority

async def add_task_to_queue(
    task_info: Tuple[str, Tuple],
    priority_str: str = "medium",
    dependencies: Optional[List[str]] = None,
    verbose_mode: Optional[bool] = None,
    retries: Optional[int] = None,
    timeout: Optional[int] = None,
    comment: str = "",
    env_vars: Optional[Dict] = None,
    dependency_mode: str = ""
) -> str:
    """Adds a new task to the priority queue.

    Args:
        task_info (Tuple[str, Tuple]): A tuple containing the task type and its arguments.
        priority_str (str): The priority of the task ('high', 'medium', 'low').
        dependencies (Optional[List[str]]): List of task IDs this task depends on.
        verbose_mode (Optional[bool]): Whether to show output in the console for this task.
        retries (Optional[int]): Number of retries for the task.
        timeout (Optional[int]): Timeout in seconds for the task.
        comment (str): A descriptive comment for the task.
        env_vars (Optional[Dict]): Environment variables to set for the task.
        dependency_mode (str): The dependency mode ('strict' or 'soft').

    Returns:
        str: The generated unique ID for the newly added task.
    """
    global queued_tasks_list
    priority_map = {"high": PRIORITY_HIGH, "medium": PRIORITY_MEDIUM, "low": PRIORITY_LOW}
    priority = priority_map.get(priority_str.lower(), PRIORITY_MEDIUM)

    task_id = create_task_id()
    task_type, args = task_info

    # Use config defaults if specific parameters are not provided.
    retries = retries if retries is not None else CONFIG_MANAGER.get('retry_attempts')
    timeout = timeout if timeout is not None else CONFIG_MANAGER.get('task_timeout_seconds')
    verbose_mode = verbose_mode if verbose_mode is not None else CONFIG_MANAGER.get('verbose_mode')
    env_vars = env_vars or {}
    dependency_mode = dependency_mode if dependency_mode else CONFIG_MANAGER.get('default_dependency_mode')

    # Store task details in task_status dictionary.
    task_status[task_id] = {
        "status": STATUS_QUEUED, "priority": priority_str.lower(), "type": task_type, "args": args,
        "verbose_mode": verbose_mode, "retries": retries, "timeout": timeout,
        "comment": comment, "env_vars": env_vars, "queued_at": datetime.now().isoformat(),
        "dependency_mode": dependency_mode
    }
    if dependencies:
        # Filter out invalid dependencies (e.g., non-existent task IDs).
        valid_deps = [dep_id for dep_id in dependencies if dep_id in task_status]
        if len(valid_deps) != len(dependencies):
            logger.warning(f"Some specified dependencies for task {task_id} were invalid and ignored.")
        if valid_deps:
            task_dependencies[task_id] = valid_deps
            task_status[task_id]["dependencies"] = valid_deps

    # Add the task to the priority queue.
    await task_queue.put(PrioritizedItem(priority, (task_id, task_type, args, verbose_mode, retries, timeout, env_vars)))
    # Add to the simplified list for quick queue viewing.
    queued_tasks_list.append((task_id, task_type, priority_str.lower()))
    logger.info(f"Task {task_id} ({task_type}) added to queue with priority '{priority_str}'. Comment: '{comment}'")
    print(f"{THEME['info']}Task {task_id} ({task_type}) added to queue.{Style.RESET_ALL}")
    return task_id

async def task_processor_loop():
    """Continuously pulls and executes tasks from the priority queue.

    Handles task dependencies, retries, and ensures graceful shutdown.
    """
    while not shutdown_event.is_set():
        try:
            # Wait for a task with a timeout to allow checking shutdown_event.
            prioritized_item = await asyncio.wait_for(task_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue  # No task, check shutdown_event again.

        task_id, task_type, args, verbose_mode, retries, timeout, env_vars = prioritized_item.item

        # --- Dependency Check ---
        deps = task_dependencies.get(task_id, [])
        dependency_mode = task_status[task_id].get("dependency_mode", DEPENDENCY_MODE_STRICT)

        # Check if any dependencies are still pending.
        unmet_deps = [dep for dep in deps if task_status.get(dep, {}).get("status") not in [STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED]]

        if unmet_deps:
            logger.info(f"Task {task_id} re-queued, waiting for dependencies: {unmet_deps}")
            await task_queue.put(prioritized_item)  # Put it back to be re-evaluated later.
            task_queue.task_done() # Mark this instance as done for the join() call.
            await asyncio.sleep(2) # Wait a bit before checking dependencies again.
            continue

        # Strict dependency mode check: fail if any dependency failed or was cancelled.
        if dependency_mode == DEPENDENCY_MODE_STRICT:
            failed_or_cancelled_deps = [dep for dep in deps if task_status.get(dep, {}).get("status") in [STATUS_FAILED, STATUS_CANCELLED]]
            if failed_or_cancelled_deps:
                logger.info(f"Task {task_id} ({DEPENDENCY_MODE_STRICT} mode) skipped due to failed/cancelled dependencies: {failed_or_cancelled_deps}")
                task_status[task_id]["status"] = STATUS_CANCELLED
                task_status[task_id]["comment"] = f"Skipped due to strict dependency failure/cancellation: {failed_or_cancelled_deps}"
                task_queue.task_done()
                continue

        # Remove task from the queued_tasks_list as it's about to be processed.
        try:
            queued_tasks_list.remove((task_id, task_type, task_status[task_id]['priority']))
        except ValueError:
            pass # Task might have been cancelled or already removed.

        # Create and run the task execution coroutine.
        coro = execute_task(task_id, task_type, args, verbose_mode, retries, timeout, env_vars)
        # Store the running task object for potential cancellation.
        task_status[task_id]["async_task"] = asyncio.create_task(coro, name=f"task_{task_id}")

        try:
            # Wait for the task to complete (or be cancelled/fail).
            await task_status[task_id]["async_task"]
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} processor loop caught external cancellation.")
        except Exception as e:
            logger.error(f"Task {task_id} processor loop caught unhandled exception: {e}", exc_info=True)
        finally:
            # Mark the item as done in the queue regardless of outcome.
            task_queue.task_done()

# --- CLI and Interactive Loop Functions ---

async def get_user_input(prompt_text: str, default: str = "", multiline: bool = False, completer: Optional[WordCompleter] = None) -> str:
    """Helper for getting user input with prompt_toolkit, applying theme and optional completer."""
    if not prompt_session:
        raise RuntimeError("Prompt session not initialized.")
    return await prompt_session.prompt_async(
        f"{THEME['prompt']}{prompt_text}{Style.RESET_ALL}",
        default=default,
        multiline=multiline,
        completer=completer
    )

def validate_priority(priority_str: str) -> str:
    """Validates and returns normalized priority string, defaulting to 'medium'."""
    valid_priorities = ["high", "medium", "low"]
    normalized = priority_str.lower().strip()
    if normalized not in valid_priorities:
        logger.warning(f"Invalid priority '{priority_str}'. Defaulting to 'medium'.")
        return "medium"
    return normalized

def validate_dependency_mode(mode_str: str) -> str:
    """Validates and returns normalized dependency mode string, defaulting to 'strict'."""
    valid_modes = [DEPENDENCY_MODE_STRICT, DEPENDENCY_MODE_SOFT]
    normalized = mode_str.lower().strip()
    if normalized not in valid_modes:
        logger.warning(f"Invalid dependency mode '{mode_str}'. Defaulting to '{DEPENDENCY_MODE_STRICT}'.")
        return DEPENDENCY_MODE_STRICT
    return normalized

async def get_common_task_params() -> Dict[str, Any]:
    """Collects common parameters for task creation from the user interactively."""
    priority = validate_priority(await get_user_input(
        "Priority (high/medium/low):", default=CONFIG_MANAGER.get('default_priority')
    ))
    dependencies_str = await get_user_input("Dependencies (comma-separated Task IDs, optional):")
    dependencies = [dep.strip().upper() for dep in dependencies_str.split(',') if dep.strip()] if dependencies_str else []
    dependency_mode = validate_dependency_mode(await get_user_input(
        "Dependency Mode (strict/soft):", default=CONFIG_MANAGER.get('default_dependency_mode')
    ))
    comment = await get_user_input("Comment (optional):")
    env_vars_str = await get_user_input("Environment variables (KEY=VALUE,KEY2=VALUE2, optional):")
    env_vars = {}
    if env_vars_str:
        for item in env_vars_str.split(','):
            if '=' in item:
                key, value = item.split('=', 1)
                env_vars[key.strip()] = value.strip()
            else:
                logger.warning(f"Invalid environment variable format: '{item}'. Skipping.")

    # Allow overriding global verbose_mode for this specific task.
    current_global_verbose = CONFIG_MANAGER.get('verbose_mode')
    verbose_q = await get_user_input(
        f"Show output in console for this task? (yes/no) [current default: {'yes' if current_global_verbose else 'no'}]:",
        default='yes' if current_global_verbose else 'no'
    )
    task_verbose_mode = verbose_q.lower() == 'yes'

    return {
        "priority_str": priority,
        "dependencies": dependencies,
        "dependency_mode": dependency_mode,
        "comment": comment,
        "env_vars": env_vars,
        "verbose_mode": task_verbose_mode
    }

# --- Command Handlers ---
# Each handler is responsible for gathering input and calling add_task_to_queue.

async def _handle_agent_command() -> None:
    agent = CONFIG_MANAGER.get('default_agent')
    prompt = await get_user_input(f"Enter query for agent '{agent}':")
    if not prompt: return
    tool_code = await get_user_input("Enter tool code (optional):")
    params = await get_common_task_params()
    await add_task_to_queue((TASK_TYPE_AGENT, (agent, prompt, tool_code or None)), **params)
async def _handle_code_command() -> None:
    lang_completer = WordCompleter(list(CONFIG_MANAGER.get('languages').keys()))
    lang = await get_user_input("Language (python/javascript/bash/ruby/php):", default="python", completer=lang_completer)
    if lang not in CONFIG_MANAGER.get('languages'):
        supported_langs = ', '.join(CONFIG_MANAGER.get('languages').keys())
        print(f"{THEME.get('error']}Invalid language: '{lang}'. Supported: {supported_langs}")
        logger.error(f"Invalid language selected for code execution: '{lang}'")
        return
    code = await get_user_input(f"Enter {lang} code:", multiline=True)
    if not code.strip():
        print(f"{THEME.get('warning']}No code entered. Task cancelled.")
        return
    params = await get_common_task_params()
    await add_task_to_queue((TASK_TYPE_CODE, (code, lang)), **params)
async def _handle_shell_command() -> None:
    command = await get_user_input("Enter shell command:")
    if not command.strip():
        print(f"{THEME.get('warning']}No command entered. Task cancelled.")
        return
    params = await get_common_task_params()
    await add_task_to_queue((TASK_TYPE_SHELL, (command,)), **params)
async def _handle_script_command() -> None:
    script_path = await get_user_input("Enter script path:")
    if not script_path.strip():
        print(f"{THEME.get('warning']}No script path entered. Task cancelled.")
        return
    if not os.path.exists(script_path):
        print(f"{THEME.get('error']}Script file not found at: '{script_path}'")
        return

    lang_completer = WordCompleter(list(CONFIG_MANAGER.get('languages').keys()))
    # Try to infer language from extension for the default value.
    _, ext = os.path.splitext(script_path)
    inferred_lang = next((lang for lang, cfg in CONFIG_MANAGER.get('languages').items() if cfg.get('extension') == ext), "python")
    lang = await get_user_input(
        f"Language ({', '.join(CONFIG_MANAGER.get('languages').keys())}):",
        default=inferred_lang,
        completer=lang_completer
    )
    if lang not in CONFIG_MANAGER.get('languages'):
        print(f"{THEME.get('error']}Unsupported language: '{lang}'.")
        logger.error(f"Unsupported language selected for script execution: '{lang}'")
        return
    params = await get_common_task_params()
    await add_task_to_queue((TASK_TYPE_SCRIPT, (script_path, lang)), **params)
async def _handle_status_command() -> None:
    handle_status_view()
    action = await get_user_input("Enter Task ID for details, 'c [ID]' to cancel, or 'c all' to cancel all:")
    parts = action.strip().upper().split()

    if not parts: return # Empty input

    command = parts[0]
    if command == 'C':
        if len(parts) == 1 or parts[1] == 'ALL':
            await cancel_all_tasks()
        elif len(parts) == 2:
            await cancel_task(parts[1])
        else:
            print(f"{THEME['warning']}Invalid cancel command format. Use 'c [TASK_ID]' or 'c all'.")
    elif parts[0]: # Assume it's a task ID for details.
        handle_history_view(filter_id=parts[0])
async def _handle_queue_command() -> None:
    list_queued_tasks()
async def _handle_history_command() -> None:
    handle_history_view()
async def _handle_settings_command() -> None:
    await handle_settings()
async def _handle_export_command() -> None:
    await handle_export_session()
async def _handle_import_command() -> None:
    await handle_import_tasks()
async def _handle_enhance_command() -> None:
    file_path = await get_user_input("Enter file path to enhance:")
    if not file_path.strip():
        print(f"{THEME.get('warning']}No file path entered. Command cancelled.")
        return
    if not os.path.exists(file_path):
        print(f"{THEME.get('error']}File not found: '{file_path}'")
        return

    # Try to infer language from extension.
    _, ext = os.path.splitext(file_path)
    inferred_lang = next((lang for lang, cfg in CONFIG_MANAGER.get('languages').items() if cfg.get('extension') == ext), "python")
    lang_completer = WordCompleter(list(CONFIG_MANAGER.get('languages').keys()))
    lang = await get_user_input(f"Language ({', '.join(CONFIG_MANAGER.get('languages').keys())}):", default=inferred_lang, completer=lang_completer)

    if lang not in CONFIG_MANAGER.get('languages'):
        print(f"{THEME.get('error']}Unsupported language: '{lang}'.")
        logger.error(f"Unsupported language for enhancement: '{lang}'")
        return

    # Use a temporary task ID for logging purposes.
    temp_task_id = create_task_id()
    await _run_code_enhancements(temp_task_id, code=None, file_path=file_path, language=lang, verbose_mode=True, env=os.environ, mode="enhance")
async def _handle_debug_command() -> None:
    file_path = await get_user_input("Enter file path to debug:")
    if not file_path.strip():
        print(f"{THEME.get('warning']}No file path entered. Command cancelled.")
        return
    if not os.path.exists(file_path):
        print(f"{THEME.get('error']}File not found: '{file_path}'")
        return

    # Try to infer language from extension.
    _, ext = os.path.splitext(file_path)
    inferred_lang = next((lang for lang, cfg in CONFIG_MANAGER.get('languages').items() if cfg.get('extension') == ext), "python")
    lang_completer = WordCompleter(list(CONFIG_MANAGER.get('languages').keys()))
    lang = await get_user_input(f"Language ({', '.join(CONFIG_MANAGER.get('languages').keys())}):", default=inferred_lang, completer=lang_completer)

    if lang not in CONFIG_MANAGER.get('languages'):
        print(f"{THEME.get('error']}Unsupported language: '{lang}'.")
        logger.error(f"Unsupported language for debugging: '{lang}'")
        return

    # Use a temporary task ID for logging purposes.
    temp_task_id = create_task_id()
    await _run_code_enhancements(temp_task_id, code=None, file_path=file_path, language=lang, verbose_mode=True, env=os.environ, mode="debug")

async def _handle_clear_queue_command() -> None:
    """Clears all tasks currently in the queued_tasks_list."""
    global queued_tasks_list
    if not queued_tasks_list:
        print(f"{THEME['info']}No tasks in the queue to clear.")
        return

    confirm = await get_user_input(f"{THEME['warning']}Are you sure you want to clear all {len(queued_tasks_list)} queued tasks? (yes/no): {Style.RESET_ALL}")
    if confirm.lower() == 'yes':
        cleared_count = 0
        tasks_to_remove_from_list = []
        # Iterate over a copy of the list to safely modify the original.
        for task_item in list(queued_tasks_list):
            task_id, _, _ = task_item
            if task_status.get(task_id, {}).get("status") == STATUS_QUEUED:
                task_status[task_id]["status"] = STATUS_CANCELLED
                task_status[task_id]["comment"] = "Cancelled by clear_queue command."
                tasks_to_remove_from_list.append(task_item)
                cleared_count += 1

        # Remove cancelled tasks from the global list.
        for task_item in tasks_to_remove_from_list:
            try:
                queued_tasks_list.remove(task_item)
            except ValueError:
                pass # Task might have been processed already.

        print(f"{THEME['success']}Cleared {cleared_count} queued tasks.")
        logger.info(f"Cleared {cleared_count} queued tasks.")
    else:
        print(f"{THEME['info']}Queue clear cancelled.")

async def handle_user_input(choice: str) -> Optional[str]:
    """Processes user input from the main interactive loop and dispatches to handlers.

    Args:
        choice (str): The user's input command.

    Returns:
        Optional[str]: Returns 'exit' if the user chooses to quit, otherwise None.
    """
    logger.info(f"User input received: '{choice}'")
    choice = choice.lower().strip()

    command_map = {
        '1': (_handle_agent_command, "Summon Agent"),
        'agent': (_handle_agent_command, "Summon Agent"),
        '2': (_handle_code_command, "Execute Code"),
        'code': (_handle_code_command, "Execute Code"),
        '3': (_handle_shell_command, "Execute Shell"),
        'shell': (_handle_shell_command, "Execute Shell"),
        '4': (_handle_script_command, "Execute Script"),
        'script': (_handle_script_command, "Execute Script"),
        '5': (_handle_status_command, "View Status"),
        'status': (_handle_status_command, "View Status"),
        '6': (_handle_queue_command, "List Queued Tasks"),
        'queue': (_handle_queue_command, "List Queued Tasks"),
        '7': (_handle_history_command, "View History"),
        'history': (_handle_history_command, "View History"),
        '8': (_handle_settings_command, "Settings"),
        'settings': (_handle_settings_command, "Settings"),
        '9': (print_help, "Help"),
        'help': (print_help, "Help"),
        '10': (_handle_export_command, "Export Session"),
        'export': (_handle_export_command, "Export Session"),
        '11': (_handle_import_command, "Import Tasks"),
        'import': (_handle_import_command, "Import Tasks"),
        '12': (_handle_enhance_command, "Enhance File"),
        'enhance': (_handle_enhance_command, "Enhance File"),
        '13': (_handle_debug_command, "Debug File"),
        'debug': (_handle_debug_command, "Debug File"),
        '14': (_handle_clear_queue_command, "Clear Queue"),
        'clear': (_handle_clear_queue_command, "Clear Queue"),
        '0': (lambda: "exit", "Exit"), # Special case to signal exit.
        'exit': (lambda: "exit", "Exit")
    }

    handler, _ = command_map.get(choice, (None, None))

    if handler:
        result = handler()
        if asyncio.iscoroutine(result):
            return await result
        elif result == "exit":
            return "exit"
    else:
        print(f"{THEME.get('warning']}Unknown command: '{choice}'. Type 'help' or '9' for options.")
        logger.warning(f"Unknown command received from user: '{choice}'")
    return None

def handle_status_view():
    """Displays the current status of all tasks, including dependencies."""
    print(f"\n{THEME['header']}\n--- Task Status Dashboard ---")
    if not task_status:
        print("No tasks to display.")
        return

    # Sort tasks by queued time, then start time for consistent viewing.
    sorted_tasks = sorted(
        task_status.items(),
        key=lambda item: (item[1].get('queued_at', ''), item[1].get('start_time', ''))
    )

    print("\nDependency Graph:")
    if task_dependencies:
        for tid, deps in task_dependencies.items():
            print(f"  {THEME['muted']}{tid} -> {', '.join(deps) or 'None'}")
    else:
        print("  No explicit dependencies defined.")

    print("\nTasks:")
    status_color_map = {
        STATUS_QUEUED: THEME['info'],
        STATUS_RUNNING: THEME['progress'],
        STATUS_COMPLETED: THEME['success'],
        STATUS_FAILED: THEME['error'],
        STATUS_CANCELLED: THEME['warning']
    }

    for tid, info in sorted_tasks:
        status = info.get('status', 'N/A').capitalize()
        color = status_color_map.get(status.lower(), Fore.WHITE) # Default to white if status unknown.

        duration = info.get('duration', '...')
        priority = info.get('priority', 'N/A').capitalize()
        task_type = info.get('type', 'N/A').capitalize()
        comment = info.get('comment', '')
        deps_str = ", ".join(info.get('dependencies', [])) or "None"
        dependency_mode = info.get('dependency_mode', 'N/A').capitalize()

        print(f"{color}{tid}: {status:<15} ({task_type}, P: {priority}, Deps: {deps_str}, Mode: {dependency_mode}, Time: {duration})")
        if comment:
            print(f"  {THEME['muted']}Comment: {comment}")
        print(Style.RESET_ALL) # Ensure reset after each line.

def list_queued_tasks():
    """Lists tasks currently in the queue, sorted by priority."""
    print(f"\n{THEME['header']}\n--- QUEUED TASKS ---")
    if not queued_tasks_list:
        print("No tasks currently in queue.")
        return

    # Sort by priority (high=1, medium=2, low=3) then by task ID.
    priority_order = {"high": 1, "medium": 2, "low": 3}
    sorted_queue = sorted(queued_tasks_list, key=lambda x: (priority_order.get(x[2], 4), x[0]))

    for task_id, task_type, priority in sorted_queue:
        comment = task_status.get(task_id, {}).get("comment", "")
        print(f"{THEME['info']}{task_id}: {task_type.capitalize()} (Priority: {priority.capitalize()}) - {comment}")

def handle_history_view(filter_id: Optional[str] = None):
    """Displays task history, with optional filtering and search."""
    history_file = CONFIG_MANAGER.get("history_file")
    history = load_json_file(history_file, [], "history")
    if not history:
        print("No history found.")
        return

    search_term = ""
    if not filter_id: # Only prompt for search if not filtering by specific ID.
        search_term = prompt_session.prompt(f"{THEME['prompt']}Search term (optional): {Style.RESET_ALL}") if prompt_session else ""

    print(f"\n{THEME['header']}\n--- Task History ---")
    found_entry = False
    # Iterate in reverse to show most recent first.
    for entry in reversed(history):
        tid = entry.get('task_id', 'N/A')

        # Apply filters.
        if filter_id and tid != filter_id:
            continue
        if search_term and search_term.lower() not in str(entry).lower():
            continue

        found_entry = True
        print(f"\n{THEME['warning']}Task ID: {tid} | Type: {entry.get('type', 'N/A').capitalize()} | Time: {entry.get('timestamp', 'N/A')}")

        # Display specific task details based on type.
        task_type = entry.get('type')
        if task_type == TASK_TYPE_SHELL:
            print(f"  Command: {entry.get('command', 'N/A')}")
        elif task_type == TASK_TYPE_AGENT:
            print(f"  Agent: {entry.get('agent', 'N/A')} | Prompt: {entry.get('prompt', 'N/A')[:100]}...")
            if 'tool_code' in entry and entry['tool_code']:
                print(f"  Tool Code: {entry['tool_code'][:100]}...")
        elif task_type == TASK_TYPE_CODE:
            print(f"  Language: {entry.get('language', 'N/A')} | Code Snippet:\n{entry.get('code', 'N/A')[:200]}...")
        elif task_type == TASK_TYPE_SCRIPT:
            print(f"  Language: {entry.get('language', 'N/A')} | Script Path: {entry.get('script_path', 'N/A')}")

        # Display common details.
        print(f"  Return Code: {entry.get('return_code', 'N/A')}")
        stdout = entry.get('stdout', '')
        if stdout:
            if len(stdout) > 500: # Show snippet and offer to view full output.
                print(f"  Output Snippet:\n{stdout[:500]}...")
                if prompt_session:
                    more = prompt_session.prompt(f"{THEME['prompt']}Show full output for {tid}? (yes/no): {Style.RESET_ALL}")
                    if more.lower() == 'yes':
                        print(f"  Full Output:\n{stdout}")
            else:
                print(f"  Output:\n{stdout}")

        print(f"{THEME['muted']}" + "-" * 40 + f"{Style.RESET_ALL}")
        if filter_id: # If filtered by ID, show only the first match.
            break

    if not found_entry:
        print(f"No history found matching your criteria.")

async def handle_settings():
    """Allows users to modify application settings interactively."""
    global THEME # Allow modification of the global THEME variable.

    print(f"\n{THEME['header']}\n--- Settings ---")

    # Define settings that can be modified.
    editable_settings = [
        "default_agent", "active_theme", "auto_execute_agent_code",
        "verbose_mode", "enhance_files", "debug_files", "max_history_entries",
        "task_timeout_seconds", "retry_attempts", "default_priority",
        "default_dependency_mode", "log_file", "log_max_bytes", "log_backup_count"
    ]
    setting_completer = WordCompleter(editable_settings)

    setting_key = await get_user_input("Setting to change? (e.g., 'active_theme', 'verbose_mode'):", completer=setting_completer)
    setting_key = setting_key.lower().strip()

    if setting_key not in editable_settings:
        print(f"{THEME.get('error']}Unknown setting: '{setting_key}'.")
        logger.error(f"User attempted to change unknown setting: '{setting_key}'")
        return

    current_value = CONFIG_MANAGER.get(setting_key)
    print(f"{THEME['info']}Current value for '{setting_key}': {current_value}")

    try:
        new_value = None
        if setting_key == 'default_agent':
            new_value = await get_user_input(f"New default agent name:", default=str(current_value))
        elif setting_key == 'active_theme':
            themes = list(CONFIG_MANAGER.get('themes', {}).keys())
            theme_completer = WordCompleter(themes)
            new_theme = await get_user_input(f"New theme ({', '.join(themes)}):", default=str(current_value), completer=theme_completer)
            if new_theme in themes:
                CONFIG_MANAGER.set('active_theme', new_theme)
                # Update global THEME variable immediately.
                THEME = CONFIG_MANAGER.get("themes").get(new_theme, CONFIG_MANAGER.DEFAULT_CONFIG["themes"]["default"])
                print(f"{THEME['success']}Theme updated successfully.")
                logger.info("Theme reloaded.")
            else:
                print(f"{THEME['error']}Invalid theme name. Available themes: {', '.join(themes)}")
                return # Do not proceed to save if theme is invalid.
        elif setting_key in ['auto_execute_agent_code', 'verbose_mode', 'enhance_files', 'debug_files']:
            new_val_str = await get_user_input(f"Set to 'yes' or 'no':", default='yes' if current_value else 'no')
            new_value = new_val_str.lower() == 'yes'
        elif setting_key in ['max_history_entries', 'task_timeout_seconds', 'retry_attempts', 'log_max_bytes', 'log_backup_count']:
            new_value_str = await get_user_input(f"New value (integer):", default=str(current_value))
            new_value = int(new_value_str)
            if new_value < 0:
                raise ValueError("Value cannot be negative.")
        elif setting_key == 'default_priority':
            new_value = validate_priority(await get_user_input(f"New default priority (high/medium/low):", default=str(current_value)))
        elif setting_key == 'default_dependency_mode':
            new_value = validate_dependency_mode(await get_user_input(f"New default dependency mode (strict/soft):", default=str(current_value)))
        elif setting_key == 'log_file':
            new_value = await get_user_input(f"New log file path:", default=str(current_value))
        else:
            # For other string settings, just prompt for new value.
            new_value = await get_user_input(f"New value:", default=str(current_value))

        # If a new value was determined, update and save config.
        if new_value is not None:
            CONFIG_MANAGER.set(setting_key, new_value)
            print(f"{THEME['success']}Setting '{setting_key}' updated to '{CONFIG_MANAGER.get(setting_key)}'.")
            logger.info(f"Setting '{setting_key}' updated to '{CONFIG_MANAGER.get(setting_key)}'.")

    except ValueError as e:
        print(f"{THEME['error']}Invalid input: {e}. Please enter a valid value.")
        logger.error(f"Invalid input for setting '{setting_key}': {e}")
    except Exception as e:
        print(f"{THEME['error']}Error updating setting: {e}")
        logger.error(f"Error updating setting '{setting_key}': {e}", exc_info=True)

async def cancel_task(task_id: str):
    """Cancels a specific running or queued task."""
    task_id = task_id.upper()
    if task_id not in task_status:
        print(f"{THEME['error']}Task '{task_id}' not found.")
        logger.error(f"Attempted to cancel non-existent task: '{task_id}'")
        return

    info = task_status.get(task_id)
    status = info.get('status')

    if status == STATUS_RUNNING:
        async_task = info.get('async_task')
        if async_task and not async_task.done():
            async_task.cancel()
            logger.info(f"Cancellation requested for running task '{task_id}'.")
            print(f"{THEME['warning']}Cancellation requested for running task '{task_id}'. It may take a moment to stop.")
        else:
            logger.warning(f"Task '{task_id}' is running but its async task is not available or already done.")
    elif status == STATUS_QUEUED:
        # Remove from the internal task_status and queued_tasks_list.
        info['status'] = STATUS_CANCELLED
        info['comment'] = "Cancelled externally."
        try:
            queued_tasks_list.remove((task_id, info['type'], info['priority']))
        except ValueError:
            pass # Task might have been processed already.
        logger.info(f"Queued task '{task_id}' marked for cancellation.")
        print(f"{THEME['warning']}Queued task '{task_id}' has been cancelled.")
    else:
        print(f"{THEME['info']}Task '{task_id}' cannot be cancelled (current status: '{status}'.).")
        logger.warning(f"Attempted to cancel task '{task_id}' in uncancelable state: '{status}'.")

async def cancel_all_tasks():
    """Cancels all running and queued tasks."""
    print(f"{THEME['warning']}Attempting to cancel all active and queued tasks...{Style.RESET_ALL}")
    tasks_to_cancel = [
        tid for tid, info in list(task_status.items())
        if info.get('status') in [STATUS_QUEUED, STATUS_RUNNING]
    ]
    if not tasks_to_cancel:
        print(f"{THEME['info']}No active or queued tasks to cancel.")
        return

    for task_id in tasks_to_cancel:
        await cancel_task(task_id)
    print(f"{THEME['success']}All eligible tasks cancellation initiated.")
    logger.info("All eligible tasks cancellation initiated.")

async def handle_export_session():
    """Exports current session state (tasks, dependencies, history, config) to a zip file."""
    export_dir = CONFIG_MANAGER.get("backup_dir")
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    zip_filename = os.path.join(export_dir, f"pyrmethus_session_export_{timestamp}.zip")

    # Use a temporary directory for staging export files.
    temp_dir = os.path.join(export_dir, f"temp_export_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Prepare session data, excluding non-serializable 'async_task' objects.
        serializable_tasks = {
            tid: {k: v for k, v in info.items() if k != 'async_task'}
            for tid, info in task_status.items()
        }

        session_data = {
            "tasks": serializable_tasks,
            "dependencies": task_dependencies,
            "queued_tasks": queued_tasks_list, # Store simplified queued list.
            "config": CONFIG_MANAGER._config # Export current config state.
        }
        session_file_temp = os.path.join(temp_dir, "session_state.json")
        save_json_file(session_file_temp, session_data, "temporary session state")

        # Copy history file if it exists.
        history_file_path = CONFIG_MANAGER.get("history_file")
        if os.path.exists(history_file_path):
            shutil.copy2(history_file_path, os.path.join(temp_dir, os.path.basename(history_file_path)))

        # Copy config file as well for completeness.
        config_file_path = CONFIG_MANAGER.config_file
        if os.path.exists(config_file_path):
            shutil.copy2(config_file_path, os.path.join(temp_dir, os.path.basename(config_file_path)))

        # Create zip archive from the temporary directory.
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir) # Path within the zip archive.
                    zf.write(file_path, arcname)

        print(f"{THEME['success']}Session exported to '{zip_filename}'")
        logger.info(f"Session exported to '{zip_filename}'")
    except Exception as e:
        print(f"{THEME['error']}Failed to export session: {e}")
        logger.error(f"Failed to export session: {e}", exc_info=True)
    finally:
        # Clean up the temporary directory.
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                logger.warning(f"Failed to remove temporary export directory '{temp_dir}': {e}")

async def handle_import_tasks():
    """Imports tasks from a JSON file into the current queue."""
    import_path = await get_user_input("Enter path to the task import file (JSON):")
    if not os.path.exists(import_path):
        print(f"{THEME['error']}File not found: '{import_path}'")
        return

    imported_data = load_json_file(import_path, {}, "tasks import")

    if not imported_data:
        print(f"{THEME['warning']}No valid tasks data found in '{import_path}'.")
        return

    # Handle different possible formats of imported data.
    tasks_to_import = []
    if isinstance(imported_data, list):
        # Older format: just a list of task dictionaries.
        tasks_to_import = imported_data
    elif isinstance(imported_data, dict):
        # Newer format: dictionary with 'tasks' key.
        tasks_to_import = imported_data.get('tasks', [])
    else:
        print(f"{THEME['error']}Invalid data format in import file. Expected a list or a dictionary with a 'tasks' key.")
        return

    if not tasks_to_import:
        print(f"{THEME['warning']}No tasks found in the import file.")
        return

    imported_count = 0
    for task_data in tasks_to_import:
        try:
            task_type = task_data.get('type')
            args: Tuple = ()
            # Reconstruct arguments based on task type.
            if task_type == TASK_TYPE_SHELL:
                args = (task_data.get('command'),)
            elif task_type == TASK_TYPE_CODE:
                args = (task_data.get('code'), task_data.get('language'))
            elif task_type == TASK_TYPE_AGENT:
                args = (task_data.get('agent'), task_data.get('prompt'), task_data.get('tool_code'))
            elif task_type == TASK_TYPE_SCRIPT:
                args = (task_data.get('script_path'), task_data.get('language'))
            else:
                logger.warning(f"Skipping task import: Unknown task type '{task_type}'.")
                continue

            # Ensure all required arguments for the task type are present.
            if not all(arg is not None for arg in args):
                logger.warning(f"Skipping task import for '{task_data.get('task_id', 'unknown')}': Missing required arguments for type '{task_type}'.")
                continue

            # Retrieve task parameters, using defaults if not present in import data.
            priority = task_data.get('priority', CONFIG_MANAGER.get('default_priority'))
            dependencies = task_data.get('dependencies', [])
            dependency_mode = task_data.get('dependency_mode', CONFIG_MANAGER.get('default_dependency_mode'))
            comment = task_data.get('comment', '')
            retries = task_data.get('retries') # Can be None if not specified.
            timeout = task_data.get('timeout') # Can be None if not specified.
            verbose_mode = task_data.get('verbose_mode') # Can be None if not specified.
            env_vars = task_data.get('env_vars', {})

            # Add the task to the queue.
            await add_task_to_queue(
                (task_type, args),
                priority_str=priority, comment=comment,
                dependencies=dependencies,
                retries=retries, timeout=timeout,
                verbose_mode=verbose_mode, env_vars=env_vars,
                dependency_mode=dependency_mode
            )
            imported_count += 1
        except Exception as e:
            logger.error(f"Error importing task from file ('{task_data.get('task_id', 'unknown')}' in '{import_path}'): {e}", exc_info=True)
            print(f"{THEME['error']}Error importing a task from file: {e}. See log for details.")

    print(f"{THEME['success']}Successfully imported {imported_count} tasks from '{import_path}'.")
    logger.info(f"Successfully imported {imported_count} tasks from '{import_path}'.")

async def check_network_connectivity() -> bool:
    """Checks for network connectivity by pinging a reliable host."""
    try:
        # Use a non-privileged, common command to check network reachability.
        # Ping is generally available on most systems.
        host_to_ping = "8.8.8.8" # Google DNS server.
        command = f"ping -c 1 {host_to_ping}" # -c 1 for one packet.

        # Use subprocess.run for simplicity if not needing async stream processing,
        # or stick with asyncio.create_subprocess_shell for consistency.
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        # A return code of 0 generally indicates success.
        is_connected = process.returncode == 0
        if not is_connected:
            logger.warning(f"Network check failed: ping to {host_to_ping} returned code {process.returncode}. stderr: {stderr.decode(errors='ignore')}")
        return is_connected

    except FileNotFoundError:
        logger.warning("'ping' command not found. Cannot reliably check network connectivity.")
        return True # Assume connected if ping is not available.
    except Exception as e:
        logger.error(f"Error checking network connectivity: {e}", exc_info=True)
        return False

def print_help():
    """Displays the help menu with command descriptions and usage."""
    help_text = f"""
    {THEME['header']}{Style.BRIGHT}\n--- Pyrmethus: The Enhanced Asynchronous Coding Wizard Help ---

    {THEME['prompt']}1. Summon Agent (agent):{THEME['muted']} Interact with an AI agent (requires 'aichat').
    {THEME['prompt']}2. Execute Code (code):{THEME['muted']} Run Python, JavaScript, Bash, Ruby, or PHP code snippets.
    {THEME['prompt']}3. Execute Shell (shell):{THEME['muted']} Run shell commands with safety checks and confirmations.
    {THEME['prompt']}4. Execute Script (script):{THEME['muted']} Run a script file from disk.
    {THEME['prompt']}5. View Status (status):{THEME['muted']} See real-time task statuses, dependencies, and manage tasks ('c [ID]' to cancel, 'c all' to cancel all).
    {THEME['prompt']}6. List Queued Tasks (queue):{THEME['muted']} Display pending tasks in the execution queue.
    {THEME['prompt']}7. View History (history):{THEME['muted']} Review past commands and their outcomes. Supports searching.
    {THEME['prompt']}8. Settings (settings):{THEME['muted']} Configure application behavior (themes, defaults, logging, etc.).
    {THEME['prompt']}9. Help (help):{THEME['muted']} Display this help menu.
    {THEME['prompt']}10. Export Session (export):{THEME['muted']} Save current session state (tasks, history, config) to a zip archive.
    {THEME['prompt']}11. Import Tasks (import):{THEME['muted']} Load tasks from a JSON file into the execution queue.
    {THEME['prompt']}12. Enhance File (enhance):{THEME['muted']} Run code linters/formatters (e.g., Black, Pylint) on a specified file.
    {THEME['prompt']}13. Debug File (debug):{THEME['muted']} Print file content and enable verbose debugging output for execution.
    {THEME['prompt']}14. Clear Queue (clear):{THEME['muted']} Cancel all currently queued tasks.
    {THEME['prompt']}0. Exit (exit):{THEME['muted']} Save session and exit the application gracefully.

    {THEME['info']}\nKey Features:{Style.BRIGHT}
        - Priorities:{THEME['muted']} Tasks can be assigned 'high', 'medium', or 'low' priority.
        - Dependencies:{THEME['muted']} Tasks can wait for others. 'Strict' mode fails if a dependency fails; 'Soft' mode waits but proceeds.
        - Cancellation:{THEME['muted']} Cancel running or queued tasks via the status view.
        - Persistence:{THEME['muted']} Session state (tasks, history) is automatically saved on exit and restored on startup.
        - Customization:{THEME['muted']} Themes, logging levels, and default behaviors are configurable via settings.
        - Environment Variables:{THEME['muted']} Pass custom environment variables to tasks via command line or settings.
        - Output Control:{THEME['muted']} Control console verbosity ('verbose_mode'); all output is always logged to file.
    """
    print(help_text)

async def create_toolbar() -> List[Tuple[str, str]]:
    """Creates a dynamic bottom toolbar for prompt_toolkit, showing task counts."""
    num_queued = len(queued_tasks_list)
    num_running = sum(1 for ts in task_status.values() if ts.get('status') == STATUS_RUNNING)

    toolbar_parts = []
    if num_queued > 0:
        toolbar_parts.append((get_pt_style(THEME['info']), f"Queued: {num_queued}"))
    if num_running > 0:
        toolbar_parts.append((get_pt_style(THEME['progress']), f"Running: {num_running}"))

    return toolbar_parts if toolbar_parts else [(get_pt_style(THEME['muted']), "No active tasks.")]
async def interactive_loop():
    """The main interactive loop for the Pyrmethus application."""
    global prompt_session

    # --- Setup Prompt Toolkit ---
    # Attempt to gather dynamic completions from installed packages (Termux specific).
    dynamic_completions = []
    if shutil.which("pkg"): # Check if 'pkg' command is available.
        try:
            installed_packages_output = await asyncio.create_subprocess_shell(
                "pkg list-installed", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await installed_packages_output.communicate()
            # Filter for package names, excluding library prefixes.
            installed_packages = stdout.decode(errors='ignore').splitlines()
            dynamic_completions = [pkg.split('/')[0] for pkg in installed_packages if '/' in pkg and not pkg.startswith('lib')]
            logger.info(f"Found {len(dynamic_completions)} dynamic completions from 'pkg list-installed'.")
        except Exception as e:
            logger.warning(f"Could not retrieve dynamic Termux package completions: {e}")
    else:
        logger.info("'pkg' command not found, skipping dynamic completions.")

    # Combine static and dynamic completions, ensure uniqueness and sort.
    all_completions = list(set(CONFIG_MANAGER.get('command_completions') + CONFIG_MANAGER.get('custom_completions') + dynamic_completions))
    all_completions.sort()

    prompt_session = PromptSession(
        history=FileHistory(CONFIG_MANAGER.get('prompt_history_file')),
        completer=FuzzyWordCompleter(all_completions), # Use Fuzzy completer for better matching.
        bottom_toolbar=create_toolbar # Attach the dynamic toolbar.
    )

    # --- Welcome Message ---
    print(f"{THEME['header']}{Style.BRIGHT}\nPyrmethus: The Asynchronous Coding Wizard Awakens!{Style.RESET_ALL}")
    print(f"{THEME['muted']}Type 'help' or '9' for commands. Ctrl+C or '0'/'exit' to quit.")

    # Offer optional tutorial.
    tutorial = await get_user_input("Run interactive tutorial? (yes/no):", default="no")
    if tutorial.lower() == 'yes':
        print(f"{THEME['info']}Tutorial: Try '1' (agent) to summon an agent, '2' (code) to run code, or '9' for help.")
        print(f"{THEME['info']}Also, try 'status' or 'queue' to see how tasks are managed!{Style.RESET_ALL}")

    # --- Main Input Loop ---
    while True:
        try:
            choice = await prompt_session.prompt_async(f"\n{THEME['prompt']}Pyrmethus> {Style.RESET_ALL}")
            if await handle_user_input(choice) == "exit":
                break
        except (KeyboardInterrupt, EOFError):
            logger.info("KeyboardInterrupt/EOFError detected. Initiating graceful shutdown.")
            break # Exit loop for graceful shutdown.
        except Exception as e:
            logger.critical(f"Main interactive loop encountered an error: {e}", exc_info=True)
            print(f"{THEME['error']}An unexpected error occurred in the main loop: {e}")

# --- Application Startup and Shutdown ---

async def shutdown(loop: asyncio.AbstractEventLoop):
    """Performs a graceful shutdown of the application."""
    logger.info("Initiating graceful shutdown...")
    print(f"\n{THEME['warning']}Shutting down Pyrmethus. Please wait...{Style.RESET_ALL}")

    # Signal the processor loop and other tasks to stop.
    shutdown_event.set()

    # Save current session state before exiting.
    session_data = {
        "tasks": {tid: {k: v for k, v in info.items() if k != 'async_task'} for tid, info in task_status.items()},
        "dependencies": task_dependencies,
        "queued_tasks": queued_tasks_list, # Save the simplified list of queued tasks.
        "config": CONFIG_MANAGER._config # Save the current configuration.
    }
    save_json_file(CONFIG_MANAGER.get("session_file"), session_data, "session")
    logger.info("Session state saved.")

    # Cancel all running asyncio tasks except the current shutdown task and the processor loop.
    current_task = asyncio.current_task()
    tasks_to_cancel = [
        t for t in asyncio.all_tasks(loop=loop)
        if t is not current_task and t.get_name() != "task_processor_loop"
    ]
    if tasks_to_cancel:
        logger.info(f"Cancelling {len(tasks_to_cancel)} running tasks...")
        # Use gather to cancel tasks concurrently and handle potential exceptions.
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    # Ensure the task processor loop is also stopped and awaited.
    processor_task = next((t for t in asyncio.all_tasks(loop=loop) if t.get_name() == "task_processor_loop"), None)
    if processor_task and not processor_task.done():
        try:
            # Give the processor a short timeout to finish its current operation and exit.
            await asyncio.wait_for(processor_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Task processor loop did not stop gracefully within timeout.")
        except asyncio.CancelledError:
            pass # Expected if it was cancelled by shutdown_event.
        except Exception as e:
            logger.error(f"Error awaiting processor task during shutdown: {e}", exc_info=True)

    logger.info("Shutdown complete. The portal closes.")
    print(f"{THEME['success']}Shutdown complete. Goodbye!{Style.RESET_ALL}")
async def main():
    """Main function to parse arguments, initialize, and start the application."""
    parser = argparse.ArgumentParser(
        description="Pyrmethus: The Asynchronous Coding Wizard. Execute tasks, manage agents, and automate workflows.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve formatting in help text.
    )
    # --- CLI Arguments for Non-Interactive Mode ---
    parser.add_argument("--agent", type=str, help="Summon an AI agent (e.g., 'pyrm').")
    parser.add_argument("--prompt", type=str, help="Query prompt for the agent.")
    parser.add_argument("--tool-code", type=str, help="Tool code to provide to the agent.")
    parser.add_argument("--shell", type=str, help="Execute a shell command.")
    parser.add_argument("--codefile", type=str, help="Path to a code file to execute.")
    parser.add_argument("--lang", type=str, help="Language of the code file/script (e.g., python, bash, javascript).")
    parser.add_argument("--script", type=str, help="Path to a script file to execute.")
    parser.add_argument("--priority", type=str, default=CONFIG_MANAGER.get('default_priority'),
                        choices=['high', 'medium', 'low'], help="Task priority (default: medium).")
    parser.add_argument("--comment", type=str, default="", help="A descriptive comment for the task.")
    parser.add_argument("--no-verbose-output", action='store_true',
                        help="Suppress task output to console (only log). Overrides global verbose_mode.")
    parser.add_argument("--enhance-files", action='store_true',
                        help="Globally enable file enhancement checks (linting/formatting). Overrides config.")
    parser.add_argument("--debug-files", action='store_true',
                        help="Globally enable detailed file debugging output. Overrides config.")
    parser.add_argument("--env", type=str,
                        help="Comma-separated environment variables for the task (KEY=VALUE,KEY2=VALUE2).")
    parser.add_argument("--dependency-mode", type=str, default=CONFIG_MANAGER.get('default_dependency_mode'),
                        choices=[DEPENDENCY_MODE_STRICT, DEPENDENCY_MODE_SOFT],
                        help="Default dependency mode for tasks (strict/soft).")

    args = parser.parse_args()

    # --- Initial Setup ---
    print(f"{THEME.get('warning']}WARNING: This script executes arbitrary code and commands. USE WITH CAUTION!{Style.RESET_ALL}")
    logger.info("Pyrmethus starting up.")

    loop = asyncio.get_running_loop()
    # Register signal handlers for graceful shutdown (SIGINT: Ctrl+C, SIGTERM: termination signal).
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(loop)))

    # Apply global CLI flags that override configuration.
    if args.enhance_files: CONFIG_MANAGER.set('enhance_files', True)
    if args.debug_files: CONFIG_MANAGER.set('debug_files', True)
    if args.dependency_mode: CONFIG_MANAGER.set('default_dependency_mode', args.dependency_mode)

    # Start the background task processor loop.
    processor_task = asyncio.create_task(task_processor_loop(), name="task_processor_loop")

    # Determine if running in non-interactive mode (commands provided via CLI args).
    is_non_interactive = args.agent or args.shell or args.codefile or args.script

    # Parse environment variables provided via CLI argument.
    env_vars_from_cli = {}
    if args.env:
        for item in args.env.split(','):
            if '=' in item:
                key, value = item.split('=', 1)
                env_vars_from_cli[key.strip()] = value.strip()
            else:
                logger.warning(f"Invalid environment variable format in CLI argument: '{item}'. Skipping.")

    if is_non_interactive:
        logger.info("Running in non-interactive mode.")
        # Determine verbose_mode: CLI flag --no-verbose-output takes precedence.
        verbose_mode_for_task = not args.no_verbose_output if args.no_verbose_output else CONFIG_MANAGER.get('verbose_mode')

        # --- Add tasks based on CLI arguments ---
        if args.agent:
            if not args.prompt: parser.error("--prompt is required when --agent is used.")
            await add_task_to_queue(
                (TASK_TYPE_AGENT, (args.agent, args.prompt, args.tool_code)),
                priority_str=args.priority, comment=args.comment,
                verbose_mode=verbose_mode_for_task, env_vars=env_vars_from_cli,
                dependency_mode=args.dependency_mode
            )

        if args.shell:
            await add_task_to_queue(
                (TASK_TYPE_SHELL, (args.shell,)),
                priority_str=args.priority, comment=args.comment,
                verbose_mode=verbose_mode_for_task, env_vars=env_vars_from_cli,
                dependency_mode=args.dependency_mode
            )

        if args.codefile:
            if not args.lang: parser.error("--lang is required when --codefile is used.")
            try:
                with open(args.codefile, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                await add_task_to_queue(
                    (TASK_TYPE_CODE, (code_content, args.lang)),
                    priority_str=args.priority, comment=args.comment,
                    verbose_mode=verbose_mode_for_task, env_vars=env_vars_from_cli,
                    dependency_mode=args.dependency_mode
                )
            except FileNotFoundError:
                logger.error(f"Code file not found: '{args.codefile}'")
                sys.exit(1) # Exit if critical file is missing.
            except IOError as e:
                logger.error(f"Could not read code file '{args.codefile}': {e}")
                sys.exit(1)

        if args.script:
            if not args.lang: parser.error("--lang is required when --script is used.")
            await add_task_to_queue(
                (TASK_TYPE_SCRIPT, (args.script, args.lang)),
                priority_str=args.priority, comment=args.comment,
                verbose_mode=verbose_mode_for_task, env_vars=env_vars_from_cli,
                dependency_mode=args.dependency_mode
            )

        # Wait for all submitted non-interactive tasks to complete.
        await task_queue.join()
        logger.info("All non-interactive tasks completed or failed. Initiating shutdown.")

        # Cancel the processor task and perform final shutdown.
        processor_task.cancel()
        try:
            await processor_task # Await its completion after cancellation.
        except asyncio.CancelledError:
            pass
        await shutdown(loop)

    else:
        # --- Interactive Mode ---
        # Load previous session state if available.
        session_file = CONFIG_MANAGER.get("session_file")
        if os.path.exists(session_file):
            session_data = load_json_file(session_file, {}, "session")
            if session_data:
                # Restore task status and dependencies.
                restored_tasks = session_data.get('tasks', {})
                for tid, info in restored_tasks.items():
                    task_status[tid] = info
                    if 'dependencies' in info:
                        task_dependencies[tid] = info['dependencies']

                # Re-add queued tasks to the live priority queue.
                restored_queued = session_data.get('queued_tasks', [])
                for tid, task_type, priority_str in restored_queued:
                    if tid in restored_tasks: # Ensure task details exist.
                        info = restored_tasks[tid]
                        priority_map = {"high": PRIORITY_HIGH, "medium": PRIORITY_MEDIUM, "low": PRIORITY_LOW}
                        priority_val = priority_map.get(priority_str.lower(), PRIORITY_MEDIUM)

                        await task_queue.put(PrioritizedItem(priority_val, (
                            tid,
                            info.get('type'),
                            info.get('args'),
                            info.get('verbose_mode', CONFIG_MANAGER.get('verbose_mode')),
                            info.get('retries', CONFIG_MANAGER.get('retry_attempts')),
                            info.get('timeout', CONFIG_MANAGER.get('task_timeout_seconds')),
                            info.get('env_vars', {})
                        )))
                        queued_tasks_list.append((tid, task_type, priority_str.lower()))
                        logger.info(f"Restored queued task '{tid}' ({task_type}) with priority '{priority_str}'.")
                    else:
                        logger.warning(f"Could not restore queued task '{tid}' as its full details were missing from session data.")
                print(f"{THEME['info']}Session loaded: {len(restored_queued)} queued tasks and {len(task_status)} total tasks restored.")

        # Start the interactive command loop.
        await interactive_loop()

        # After interactive loop exits, perform shutdown.
        await shutdown(loop)
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Graceful shutdown is handled by signal handlers, so just catch and pass here.
        pass
    except Exception as e:
        # Catch any unexpected critical errors outside the main loop's error handling.
        logger.critical(f"A critical error occurred outside the main event loop: {e}", exc_info=True)
        sys.exit(1)
