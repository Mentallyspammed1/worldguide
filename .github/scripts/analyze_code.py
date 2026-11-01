# .github/scripts/analyze_code.py
import os
import json
import requests
from colorama import init, Fore, Style, Back

init(autoreset=True)

MAX_CHUNK_TOKENS = 3800
NIM_MODEL_NAME = 'nvidia/nemotron-4-340b-instruct'
NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
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


def analyze_code_chunk_with_nim(diff_chunk, api_key):
    """Send a diff chunk to NVIDIA NIM for code review."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
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
    ```
    """
    payload = {
        "model": NIM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert code reviewer that responds only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    try:
        resp = requests.post(NIM_BASE_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print_error_message(f"NVIDIA NIM API error: {e}")
        return "[]"


def get_nim_api_key():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print_error_message("NVIDIA_API_KEY is not set. Please set it as an environment variable or repository secret.")
        return None
    return api_key
