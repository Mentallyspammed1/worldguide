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
        return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        sys.exit(1)

def enhance_code(content, model, max_attempts=3):
    """Enhance Python code using the Gemini API with retry logic."""
    prompt = (
        "Enhance the following Python code. Improve readability, performance, and maintainability. "
        "Add comments where necessary, optimize algorithms, and follow PEP 8 guidelines. "
        "Return only the enhanced code without any explanations or markdown formatting:\n\n"
        f"{content}"
    )
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            print(f"Empty response from Gemini API on attempt {attempt + 1}")
        except Exception as e:
            print(f"Error enhancing code on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2)  # Wait before retrying
    print("Failed to enhance code after maximum attempts")
    return content  # Return original content if enhancement fails

def process_files(directory, file_pattern, api_key, max_calls):
    """Process Python files in the directory matching the file pattern."""
    print(f"Starting code enhancement at {datetime.now()}")
    print(f"Directory: {directory}")
    print(f"File pattern: {file_pattern}")
    print(f"Max API calls per minute: {max_calls}")

    # Change to the specified directory
    try:
        os.chdir(directory)
    except Exception as e:
        print(f"Error changing to directory {directory}: {e}")
        sys.exit(1)

    # Configure Gemini API
    model = configure_gemini(api_key)
    max_calls = int(max_calls)
    call_interval = 60.0 / max_calls  # Seconds between API calls
    modified_files = []
    processed_files = 0

    # Find matching files
    files = list(glob.glob(file_pattern, recursive=True))
    if not files:
        print(f"No files found matching pattern '{file_pattern}'")
        return modified_files

    print(f"Found {len(files)} file(s) to process")

    for filepath in files:
        print(f"\nProcessing {filepath}")
        processed_files += 1

        # Read original content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

        # Rate limiting
        if processed_files > 1:
            time.sleep(call_interval)

        # Enhance code
        enhanced_content = enhance_code(original_content, model)
        if enhanced_content == original_content:
            print(f"No changes needed for {filepath}")
            continue

        # Write enhanced content back to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)
            print(f"Successfully enhanced {filepath}")
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
    """Main function to process command-line arguments and run the enhancement."""
    if len(sys.argv) < 3:
        print("Usage: xfix_files.py <directory> <file_pattern>")
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
        sys.exit(0)  # Exit successfully if no changes, but workflow will skip commit
    sys.exit(0)

if __name__ == "__main__":
    main()
