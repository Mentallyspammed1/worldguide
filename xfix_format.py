#!/usr/bin/env python3
import os
import sys
import glob
import time
import google.generativeai as genai
from datetime import datetime


def configure_gemini(api_key):
    """Configure the Gemini API with the provided API key."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-pro")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        sys.exit(1)


def format_and_fix_code(content, model, max_attempts=3):
    """Format Python code to PEP 8 standards and fix common issues."""
    prompt = (
        "Format the following Python code to comply with PEP 8 guidelines. "
        "Fix common issues such as syntax errors, deprecated functions, inefficient patterns, "
        "and ensure consistent style (e.g., quotes, spacing). "
        "Return only the formatted and fixed code without explanations or markdown formatting:\n\n"
        f"{content}"
    )
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            print(f"Empty response from Gemini API on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error formatting code on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)
    print("Failed to format code after maximum attempts")
    return content


def process_files(directory, file_pattern, api_key, max_calls):
    """Process Python files to format and fix code."""
    print(f"Starting code formatting and fixing at {datetime.now()}")
    print(f"Directory: {directory}")
    print(f"File pattern: {file_pattern}")
    print(f"Max API calls per minute: {max_calls}")

    try:
        os.chdir(directory)
    except Exception as e:
        print(f"Error changing to directory {directory}: {e}")
        sys.exit(1)

    model = configure_gemini(api_key)
    max_calls = int(max_calls)
    call_interval = 60.0 / max_calls
    modified_files = []
    processed_files = 0

    files = list(glob.glob(file_pattern, recursive=True))
    if not files:
        print(f"No files found matching pattern '{file_pattern}'")
        return modified_files

    print(f"Found {len(files)} file(s) to process")

    for filepath in files:
        print(f"\nProcessing {filepath}")
        processed_files += 1

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original_content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        if processed_files > 1:
            time.sleep(call_interval)

        formatted_content = format_and_fix_code(original_content, model)
        if formatted_content == original_content:
            print(f"No changes needed for {filepath}")
            continue

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(formatted_content)
            print(f"Successfully formatted and fixed {filepath}")
            modified_files.append(filepath)
        except Exception as e:
            print(f"Error writing to {filepath}: {e}")
            continue

    print(f"\nProcessing complete at {datetime.now()}")
    print(f"Processed {processed_files} file(s), modified {len(modified_files)} file(s)")
    if modified_files:
        print("Modified files:")
        for filepath in modified_files:
            print(f"  - {filepath}")
    return modified_files


def main():
    """Main function to process command-line arguments and run the formatter."""
    if len(sys.argv) < 3:
        print("Usage: xfix_format.py <directory> <file_pattern>")
        sys.exit(1)

    directory = sys.argv[1]
    file_pattern = sys.argv[2]
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        sys.exit(1)

    max_calls = os.getenv("MAX_API_CALLS", "59")
    try:
        max_calls = int(max_calls)
        if not 1 <= max_calls <= 60:
            raise ValueError
    except ValueError:
        print("Error: MAX_API_CALLS must be a number between 1 and 60")
        sys.exit(1)

    modified_files = process_files(directory, file_pattern, api_key, max_calls)
    if not modified_files:
        print("No files were modified")
        sys.exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
