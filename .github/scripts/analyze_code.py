# .github/scripts/analyze_code.py
import os
import json
import requests
import google.generativeai as genai
from colorama import init, Fore, Style, Back

init(autoreset=True)

MAX_CHUNK_TOKENS = 3800
GEMINI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25'
GITHUB_API_BASE_URL = "https://api.github.com"

def print_wizard_message(message, color=Fore.CYAN):
    print(color + Style.BRIGHT + "✨ Pyrmethus whispers: " + Style.RESET_ALL + color + message + Style.RESET_ALL)

def print_error_message(message):
    print(Back.RED + Fore.WHITE + Style.BRIGHT + "⚠️ Arcane Anomaly: " + Style.RESET_ALL + Fore.RED + message + Style.RESET_ALL)

def chunk_diff(diff_text, max_tokens=MAX_CHUNK_TOKENS):
    print_wizard_message(f"Chunking the diff (max ~{max_tokens} tokens per chunk)...", Fore.MAGENTA)
    max_chars = max_tokens * 4
    lines = diff_text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_char_count = 0
    current_file = None

    for line in lines:
        if line.startswith('+++ b/'):
            current_file = line[6:]
        line_len = len(line) + 1
        if current_char_count + line_len > max_chars and current_chunk_lines:
            chunks.append(("\n".join(current_chunk_lines), current_file))
            current_chunk_lines = []
            current_char_count = 0
            if line.startswith('+++ b/'):
                 current_chunk_lines.append(line)
                 current_char_count += line_len
            elif not line.startswith('--- a/'):
                 current_chunk_lines.append(line)
                 current_char_count += line_len
        elif not (line.startswith('--- a/') and not current_chunk_lines):
             current_chunk_lines.append(line)
             current_char_count += line_len

    if current_chunk_lines:
        if not current_file:
             for line in reversed(current_chunk_lines):
                 if line.startswith('+++ b/'):
                     current_file = line[6:]
                     break
        chunks.append(("\n".join(current_chunk_lines), current_file if current_file else "unknown_file"))

    print_wizard_message(f"Diff split into {Fore.YELLOW}{len(chunks)}{Fore.MAGENTA} chunks.", Fore.MAGENTA)
    return chunks

def analyze_code_chunk_with_gemini(diff_chunk, model):
    prompt = f"""
    You are an expert code reviewer AI assistant, Pyrmethus's familiar.
    Analyze the following code diff strictly for potential bugs, security vulnerabilities, and significant logical errors within the changes presented.
    Ignore stylistic suggestions unless they indicate a potential bug (e.g., variable shadowing).
    Focus ONLY on the added or modified lines (usually starting with '+').

    For each distinct issue found, provide:
    1.  `file_path`: The full path of the file where the issue occurs (e.g., "src/utils/helpers.py"). Infer this from the '+++ b/...' line in the diff chunk.
    2.  `line_number`: The approximate line number *in the new file version* where the issue begins. Estimate based on the '@@ ... +start,count @@' hunk header and '+' lines.
    3.  `description`: A concise explanation of the potential bug or vulnerability.
    4.  `affected_code`: The specific line(s) from the diff (lines starting with '+') that contain the issue.
    5.  `suggested_fix`: A concrete code snippet demonstrating how to fix the issue. If the fix involves removing lines, indicate that clearly.

    If no significant issues are found in this chunk, respond with an empty JSON array: `[]`.

    Respond ONLY with a valid JSON array containing objects matching the structure described above.

    Diff Chunk:
    ```diff
    {diff_chunk}
