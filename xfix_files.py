#!/usr/bin/env python3
import os
import glob
import google.generativeai as genai
import time
import logging
from pathlib import Path
import sys
import re

<<<<<<< HEAD
def configure_gemini(api_key):
    """Configure the Gemini API with the provided API key."""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        sys.exit(1)
=======
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhancement_log.txt', mode='a')
    ]
)
logger = logging.getLogger(__name__)
>>>>>>> 67c3910 (Add GitHub Actions workflows for code enhancement and linting)

def configure_api():
    """Configure the Gemini API with the provided API key and safety settings."""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        sys.exit(1)
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25', safety_settings=safety_settings)

def get_python_files(base_dir, pattern):
    """Get list of Python files matching the pattern in the base directory."""
    try:
        base_path = Path(base_dir)
        files = [str(f) for f in base_path.glob(pattern) if f.is_file()]
        with open('matched_files.txt', 'w', encoding='utf-8') as f:
            f.write(f"Matched files for pattern '{pattern}' in directory '{base_dir}':\n")
            if files:
                f.write('\n'.join(files) + '\n')
            else:
                f.write("No Python files matched the pattern\n")
        logger.info(f"Found {len(files)} Python files matching pattern '{pattern}' in '{base_dir}'")
        return sorted(files)
    except Exception as e:
        logger.error(f"Invalid file pattern '{pattern}' in directory '{base_dir}': {str(e)}")
        with open('matched_files.txt', 'w', encoding='utf-8') as f:
            f.write(f"Error: Invalid file pattern '{pattern}' in directory '{base_dir}': {str(e)}\n")
        sys.exit(1)

def read_file(file_path):
    """Read content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Successfully read {file_path}")
            return content
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {str(e)}")
        return None

def write_file(file_path, content):
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Successfully wrote enhanced code to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {str(e)}")
        return False

def enhance_code(model, code, file_path):
    """Enhance Python code using Gemini API."""
    prompt = f"""You are an expert Python code reviewer. Analyze the following Python code and suggest improvements focusing on:
1. Code readability and documentation (e.g., docstrings, comments)
2. Performance optimization (e.g., efficient algorithms, reducing redundancy)
3. Error handling (e.g., try-except blocks, input validation)
4. Python best practices and PEP 8 compliance (e.g., naming conventions, code structure)
5. Type hints where appropriate (e.g., for function parameters and return types)

Provide the enhanced code wrapped in ```python
```

File: {file_path}
```python
{code}
```

Return only the improved code within ```python``` blocks, with comments explaining major changes."""
    
    try:
        response = model.generate_content(prompt)
        if not response.text:
            logger.warning(f"No enhancement suggestions for {file_path}")
            return None
        
        # Extract code from response
        code_match = re.search(r'```python\n(.*?)```', response.text, re.DOTALL)
        if not code_match:
            logger.error(f"No valid code block in response for {file_path}")
            return None
            
        enhanced_code = code_match.group(1).strip()
        if enhanced_code == code.strip():
            logger.info(f"No changes suggested for {file_path}")
            return None
            
        logger.info(f"Generated enhancements for {file_path}")
        return enhanced_code
    except Exception as e:
        logger.error(f"API call failed for {file_path}: {str(e)}")
        return None

def main(base_dir, file_pattern):
    """Main function to process and enhance Python files."""
    # Ensure enhancement log exists
    with open('enhancement_log.txt', 'w', encoding='utf-8') as f:
        f.write(f"Code enhancement log for directory '{base_dir}' and pattern '{file_pattern}'\n")
    
    max_api_calls = int(os.environ.get('MAX_API_CALLS', 59))
    logger.info(f"Starting enhancement process with max API calls: {max_api_calls}")
    model = configure_api()
    files = get_python_files(base_dir, file_pattern)
    
    if not files:
        logger.warning("No Python files found to enhance")
        with open('enhancement_log.txt', 'a', encoding='utf-8') as f:
            f.write("No Python files found to enhance\n")
        return
    
    # Calculate delay to respect API rate limit (calls per minute)
    delay = 60.0 / max_api_calls if max_api_calls > 0 else 0
    logger.info(f"API rate limit delay: {delay} seconds per call")
    
    modified_files = 0
    for file_path in files:
        logger.info(f"Processing {file_path}")
        
        original_code = read_file(file_path)
        if original_code is None:
            continue
            
        enhanced_code = enhance_code(model, original_code, file_path)
        if enhanced_code and write_file(file_path, enhanced_code):
            modified_files += 1
        
        # Respect rate limit
        if delay > 0:
            time.sleep(delay)
    
    with open('enhancement_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"Processed {len(files)} files, modified {modified_files} files\n")
    logger.info(f"Completed processing: {len(files)} files processed, {modified_files} files modified")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: ./xfix_files.py <base_directory> <file_pattern>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    file_pattern = sys.argv[2]
    
    if not os.path.isdir(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        sys.exit(1)
    
    main(base_dir, file_pattern)
