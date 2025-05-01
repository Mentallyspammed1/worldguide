#!/usr/bin/env python3
import os
import sys
import time
import glob
import google.generativeai as genai
from colorama import init, Fore, Style
from pathlib import Path

init(autoreset=True)

def print_wizard_message(message, color=Fore.CYAN):
    print(color + Style.BRIGHT + "✨ Pyrmethus whispers: " + Style.RESET_ALL + color + message)

def print_error_message(message):
    print(Fore.RED + Style.BRIGHT + "❌ Arcane Anomaly: " + Style.RESET_ALL + Fore.RED + message)

def enhance_code_with_gemini(file_path, content, model, max_api_calls, calls_made):
    """Enhance a single file's content using Gemini API with rate limiting."""
    if calls_made >= max_api_calls:
        print_wizard_message(f"Rate limit of {max_api_calls} calls/min reached. Pausing...", Fore.YELLOW)
        time.sleep(60 - (time.time() % 60))
        calls_made = 0

    prompt = f"""
    You are an expert code enhancer, Pyrmethus's familiar. Analyze the following Python code and suggest improvements such as:
    - Adding docstrings and type hints
    - Optimizing performance
    - Improving readability
    - Fixing potential bugs
    Provide the enhanced code in a code block (```python\n...\n```) and a brief explanation of changes.

    File: {file_path}
    ```python
    {content}
    ```
    """
    print_wizard_message(f"Enhancing {file_path}...", Fore.BLUE)
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        enhanced_code = ""
        explanation = ""
        if response_text.startswith("```python"):
            parts = response_text.split("```python\n", 1)[1].split("```\n", 1)
            enhanced_code = parts[0]
            explanation = parts[1].strip() if len(parts) > 1 else "No explanation provided."
        else:
            explanation = response_text
        return enhanced_code, explanation, calls_made + 1
    except Exception as e:
        print_error_message(f"Failed to enhance {file_path}: {e}")
        return content, f"Error: {e}", calls_made + 1

def main():
    print_wizard_message("Starting Code Enhancement Spell...", Fore.MAGENTA)
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_error_message(f"Usage: {sys.argv[0]} <repository_path> [file_pattern]")
        sys.exit(2)

    repo_path = Path(sys.argv[1])
    file_pattern = sys.argv[2] if len(sys.argv) == 3 else os.getenv("FILE_PATTERN", "**/*.py")
    
    print_wizard_message(f"Repository path: {repo_path}", Fore.CYAN)
    print_wizard_message(f"File pattern: {file_pattern}", Fore.CYAN)

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
    except Exception as e:
        print_error_message(f"Failed to configure Gemini API: {e}")
        sys.exit(1)

    output_dir = repo_path / "enhanced"
    output_dir.mkdir(exist_ok=True)

    try:
        python_files = glob.glob(file_pattern, root_dir=repo_path, recursive=True)
        print_wizard_message(f"Found {len(python_files)} files matching pattern.", Fore.CYAN)
    except Exception as e:
        print_error_message(f"Failed to find files with pattern {file_pattern}: {e}")
        sys.exit(1)

    if not python_files:
        print_wizard_message("No files found to enhance.", Fore.YELLOW)
        with open(output_dir / "no_files.txt", "w") as f:
            f.write("No files matched the pattern.")
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
            continue

        enhanced_code, explanation, calls_made = enhance_code_with_gemini(
            file_path, content, model, max_api_calls, calls_made
        )

        if enhanced_code and enhanced_code != content:
            output_path = output_dir / file_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(enhanced_code)
            print_wizard_message(f"Enhanced {file_path} saved to {output_path}", Fore.GREEN)
            print_wizard_message(f"Explanation: {explanation}", Fore.CYAN)
        else:
            print_wizard_message(f"No enhancements needed for {file_path}", Fore.YELLOW)

    elapsed = time.time() - start_time
    print_wizard_message(f"Enhancement complete. Processed {len(python_files)} files in {elapsed:.2f} seconds.", Fore.MAGENTA)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error_message(f"Unexpected error: {e}")
        sys.exit(1)
