#!/data/data/com.termux/files/usr/bin/env python
from colorama import init, Fore, Style
import requests
import os
import sys
import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm
import diff_match_patch as dmp_module

# Initialize Colorama for vibrant terminal output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('enhancement.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# NVIDIA NIM configuration
NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
DEFAULT_NIM_MODEL = 'nvidia/nemotron-4-340b-instruct'


class RateLimiter:
    """Token-bucket rate limiter for API calls."""
    def __init__(self, calls_per_minute):
        self.rate = calls_per_minute / 60.0  # Tokens per second
        self.capacity = calls_per_minute
        self.tokens = self.capacity
        self.last_refill = time.time()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.rate
            logger.info(Fore.YELLOW + f"# Pausing for {wait_time:.2f}s to respect rate limit..." + Style.RESET_ALL)
            await asyncio.sleep(wait_time)
            self.tokens = min(self.capacity, self.tokens + wait_time * self.rate)
            self.last_refill = time.time()

        self.tokens -= 1


def configure_api(api_key):
    """Validate the NVIDIA NIM API key."""
    if not api_key:
        logger.error(Fore.RED + "Error: NVIDIA_API_KEY is not set." + Style.RESET_ALL)
        sys.exit(1)
    logger.info(Fore.CYAN + "# NVIDIA NIM API key configured successfully." + Style.RESET_ALL)


def load_config(config_path):
    """Load configuration from a JSON file."""
    default_config = {
        "model": DEFAULT_NIM_MODEL,
        "max_calls_per_minute": 59,
        "enhancement_mode": "comprehensive",
        "backup_dir": ".enhancement_backups"
    }
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            default_config.update(config)
            logger.info(Fore.CYAN + f"# Loaded configuration from {config_path}." + Style.RESET_ALL)
        except Exception as e:
            logger.warning(Fore.YELLOW + f"# Failed to load config: {e}. Using defaults." + Style.RESET_ALL)
    return default_config


def create_backup(file_path, original_code, backup_dir):
    """Create a backup of the original file."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_code)
    logger.info(Fore.MAGENTA + f"# Backed up {file_path} to {backup_path}." + Style.RESET_ALL)
    return backup_path


def compute_diff(original, enhanced):
    """Compute a human-readable diff between original and enhanced code."""
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(original, enhanced)
    dmp.diff_cleanupSemantic(diffs)
    diff_text = []
    for op, data in diffs:
        if op == 0:
            diff_text.append(data)
        elif op == 1:
            diff_text.append(Fore.GREEN + f"+ {data}" + Style.RESET_ALL)
        elif op == -1:
            diff_text.append(Fore.RED + f"- {data}" + Style.RESET_ALL)
    return '\n'.join(diff_text)


async def enhance_code(file_path, model_name, mode, rate_limiter, api_key, dry_run=False):
    """Enhance a Python file using NVIDIA NIM."""
    try:
        logger.info(Fore.BLUE + f"# Summoning enhancements for {file_path} (Mode: {mode})..." + Style.RESET_ALL)
        if not os.path.exists(file_path):
            logger.error(Fore.RED + f"Error: File {file_path} does not exist." + Style.RESET_ALL)
            return False

        # Read the original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # Define enhancement prompts based on mode
        prompts = {
            "readability": """Fix syntax errors and enhance the Python code for readability and maintainability. Follow PEP 8, add clear comments, and improve variable names. Preserve functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "performance": """Fix syntax errors and optimize the Python code for performance. Reduce time complexity, minimize memory usage, and use efficient data structures. Preserve functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "type_hints": """Fix syntax errors and add type hints to the Python code to improve type safety, following PEP 484. Preserve functionality. Return only the enhanced code wrapped in ```python ... ```.""",
            "comprehensive": """Fix syntax errors and enhance the Python code comprehensively: improve readability (PEP 8, comments, naming), optimize performance, and add type hints (PEP 484). Preserve functionality. Return only the enhanced code wrapped in ```python ... ```.""",
        }
        prompt_template = prompts.get(mode, prompts["comprehensive"])
        prompt = f"{prompt_template}\n\n```python\n{original_code}\n```"

        # Rate limiting
        await rate_limiter.acquire()

        # Call NVIDIA NIM API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert Python code enhancer. Return only the enhanced code wrapped in ```python ... ``` blocks."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 8192,
        }

        try:
            resp = requests.post(NIM_BASE_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(Fore.RED + f"Error calling NVIDIA NIM API for {file_path}: {str(e)}" + Style.RESET_ALL)
            return False

        if not response_text:
            logger.error(Fore.RED + f"Error: No valid response from NVIDIA NIM for {file_path}." + Style.RESET_ALL)
            return False

        # Extract enhanced code
        enhanced_code = response_text.strip()
        if enhanced_code.startswith('```python') and enhanced_code.endswith('```'):
            enhanced_code = enhanced_code[10:-3].strip()
        elif enhanced_code.startswith('```') and enhanced_code.endswith('```'):
            enhanced_code = enhanced_code[3:-3].strip()
        else:
            logger.warning(Fore.YELLOW + f"# Response not properly formatted; assuming raw code." + Style.RESET_ALL)

        # Compute and log diff
        diff = compute_diff(original_code, enhanced_code)
        logger.info(Fore.CYAN + f"# Diff for {file_path}:\n{diff}" + Style.RESET_ALL)

        if dry_run:
            logger.info(Fore.YELLOW + f"# Dry run: Changes not applied to {file_path}." + Style.RESET_ALL)
            return True

        # Create backup
        backup_path = create_backup(file_path, original_code, ".enhancement_backups")

        # Write enhanced code
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_code)

        logger.info(Fore.GREEN + f"Successfully enhanced {file_path}." + Style.RESET_ALL)
        logger.info(Fore.YELLOW + f"# Original length: {len(original_code)} chars, Enhanced length: {len(enhanced_code)} chars" + Style.RESET_ALL)
        return True

    except Exception as e:
        logger.error(Fore.RED + f"Error enhancing {file_path}: {str(e)}" + Style.RESET_ALL)
        return False


async def main():
    """Main function to parse arguments and enhance files."""
    parser = argparse.ArgumentParser(description="Pyrmethus Advanced Code Enhancer (NVIDIA NIM)")
    parser.add_argument("file_path", help="Path to the Python file to enhance")
    parser.add_argument("--model", default=DEFAULT_NIM_MODEL, help="NVIDIA NIM model to use")
    parser.add_argument("--max-calls", type=int, default=59, help="Max API calls per minute (1-60)")
    parser.add_argument("--mode", choices=["readability", "performance", "type_hints", "comprehensive"], default="comprehensive", help="Enhancement mode")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying them")
    args = parser.parse_args()

    logger.info(Fore.MAGENTA + "# Pyrmethus Advanced Code Enhancer Initialized (NVIDIA NIM)" + Style.RESET_ALL)

    # Load configuration
    config = load_config(args.config)
    model_name = args.model or config["model"]
    max_calls = args.max_calls or config["max_calls_per_minute"]
    mode = args.mode or config["enhancement_mode"]

    # Validate max_calls
    if not 1 <= max_calls <= 60:
        logger.error(Fore.RED + f"Error: max-calls must be between 1 and 60, got {max_calls}." + Style.RESET_ALL)
        sys.exit(1)

    # Get NVIDIA API key
    api_key = os.getenv("NVIDIA_API_KEY")
    configure_api(api_key)

    # Initialize rate limiter
    rate_limiter = RateLimiter(max_calls)

    # Enhance file with progress feedback
    with tqdm(total=1, desc="Enhancing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        success = await enhance_code(args.file_path, model_name, mode, rate_limiter, api_key, args.dry_run)
        pbar.update(1)

    if success:
        logger.info(Fore.GREEN + "# Enhancement spell completed successfully." + Style.RESET_ALL)
        sys.exit(0)
    else:
        logger.error(Fore.RED + "# Enhancement spell failed." + Style.RESET_ALL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
