#!/usr/bin/env python3
import os
import sys
import time
import glob
import logging
import re
from datetime import datetime
from typing import Tuple, Optional, List
from pathlib import Path
import google.generativeai as genai
from colorama import init, Fore, Style

init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("enhancement_log.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def print_wizard_message(message: str, color: str = Fore.CYAN) -> None:
    """Print a styled message to console."""
    logger.info(f"✨ Pyrmethus whispers: {message}")
    print(color + Style.BRIGHT + "✨ Pyrmethus whispers: " + Style.RESET_ALL + color + message)

def print_error_message(message: str) -> None:
    """Print an error message to console and log."""
    logger.error(f"❌ Arcane Anomaly: {message}")
    print(Fore.RED + Style.BRIGHT + "❌ Arcane Anomaly: " + Style.RESET_ALL + Fore.RED + message)

def enhance_code_with_gemini(
    file_path: str, content: str, model: genai.GenerativeModel, max_api_calls: int, calls_made: int
) -> Tuple[str, str, int]:
    """Enhance a single Python file's content using Gemini API with rate limiting."""
    if calls_made >= max_api_calls:
        print_wizard_message(f"Rate limit of {max_api_calls} calls/min reached. Pausing...", Fore.YELLOW)
        time.sleep(60 - (time.time() % 60))
        calls_made = 0

    prompt = f"""
    You are an expert Python code enhancer, Pyrmethus's familiar. Analyze the following Python code and suggest improvements such as:
    - Adding docstrings and type hints
    - Optimizing performance
    - Improving readability
    - Fixing potential bugs
    Provide the enhanced code in a code block (```python\n...\n```) and a brief explanation of changes after the code block.

    File: {file_path}
    ```python
    {content}
    ```
    """
    print_wizard_message(f"Enhancing {file_path}...", Fore.BLUE)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Robust parsing for code block
        code_pattern = r"```python\n([\s\S]*?)\n```"
        match = re.search(code_pattern, response_text)
        enhanced_code = content
        explanation = "No explanation provided."
        
        if match:
            enhanced_code = match.group(1)
            explanation_start = match.end()
            explanation = response_text[explanation_start:].strip() or "No explanation provided."
        else:
            explanation = response_text

        logger.info(f"Enhanced {file_path}\nExplanation: {explanation}")
        return enhanced_code, explanation, calls_made + 1
    except Exception as e:
        import traceback
        error_message = f"Failed to enhance {file_path}: {str(e)}\n{traceback.format_exc()}"
        print_error_message(error_message)
        logger.error(error_message)
        return content, f"Error: {str(e)}", calls_made + 1

def main() -> None:
    """Main function to enhance Python (.py) files."""
    # Ensure log file is created at startup
    logger.info("Initializing enhancement log")
    
    print_wizard_message("Starting Code Enhancement Spell for Python Files...", Fore.MAGENTA)
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_error_message(f"Usage: {sys.argv[0]} <repository_path> [file_pattern]")
        sys.exit(2)

    repo_path = Path(sys.argv[1])
    file_pattern = sys.argv[2] if len(sys.argv) == 3 else os.getenv("FILE_PATTERN", "**/*.py")
    
    if not file_pattern.endswith(".py"):
        print_error_message("File pattern must target .py files (e.g., '*.py', 'src/*.py')")
        sys.exit(1)

    print_wizard_message(f"Repository path: {repo_path}", Fore.CYAN)
    print_wizard_message(f"File pattern: {file_pattern}", Fore.CYAN)
    logger.info(f"Repository path: {repo_path}")
    logger.info(f"File pattern: {file_pattern}")

    if not repo_path.is_dir():
        print_error_message(f"Invalid repository path: {repo_path}")
        sys.exit(1)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print_error_message("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    max_api_calls = int(os.getenv("MAX_API_CALLS", 59))
    if not 1 <= max_api_calls <= 60:
        print_error_message("MAX_API_CALLS must be between 1 and 60")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        print_wizard_message("Gemini Oracle configured successfully.", Fore.GREEN)
        logger.info("Gemini Oracle configured successfully.")
    except Exception as e:
        print_error_message(f"Failed to configure Gemini API: {e}")
        sys.exit(1)

    try:
        python_files = glob.glob(file_pattern, root_dir=repo_path, recursive=True)
        print_wizard_message(f"Found {len(python_files)} Python files matching pattern.", Fore.CYAN)
        logger.info(f"Found {len(python_files)} Python files matching pattern.")
    except Exception as e:
        print_error_message(f"Failed to find Python files with pattern {file_pattern}: {e}")
        sys.exit(1)

    if not python_files:
        print_wizard_message("No Python files found to enhance.", Fore.YELLOW)
        logger.warning("No Python files found to enhance.")
        with open("enhancement_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} [WARNING] No Python files found to enhance.\n")
        sys.exit(0)

    calls_made = 0
    start_time = time.time()
    for file_path in python_files:
        abs_file_path = repo_path / file_path
        try:
            with open(abs_file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print_error_message(f"Failed to read {file_path}: {e}")
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        enhanced_code, explanation, calls_made = enhance_code_with_gemini(
            file_path, content, model, max_api_calls, calls_made
        )

        if enhanced_code and enhanced_code != content:
            with open(abs_file_path, "w", encoding="utf-8") as f:
                f.write(enhanced_code)
            print_wizard_message(f"Enhanced {file_path}", Fore.GREEN)
            logger.info(f"Enhanced {file_path}")
        else:
            print_wizard_message(f"No enhancements needed for {file_path}", Fore.YELLOW)
            logger.info(f"No enhancements needed for {file_path}")

    elapsed = time.time() - start_time
    print_wizard_message(f"Enhancement complete. Processed {len(python_files)} Python files in {elapsed:.2f} seconds.", Fore.MAGENTA)
    logger.info(f"Enhancement complete. Processed {len(python_files)} Python files in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print_error_message(error_message)
        logger.error(error_message)
        sys.exit(1)
