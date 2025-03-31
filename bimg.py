import time

from bing_image_downloader import downloader
from colorama import Back, Fore, Style, init

# Initialize Colorama for Termux compatibility
init()

# --- Visually Enchanted Theme ---
# Colors chosen for mystical clarity and Termux vibrancy

TITLE_COLOR = Fore.BLUE + Style.BRIGHT
STATUS_COLOR = Fore.CYAN + Style.BRIGHT
INFO_COLOR = Fore.WHITE + Style.DIM
DATA_COLOR = Fore.YELLOW + Style.BRIGHT
SUCCESS_TITLE_COLOR = Fore.GREEN + Style.BRIGHT + Back.BLACK
SUCCESS_TEXT_COLOR = Fore.GREEN + Style.BRIGHT
ERROR_TITLE_COLOR = Fore.RED + Style.BRIGHT + Back.BLACK
ERROR_TEXT_COLOR = Fore.RED + Style.BRIGHT
CODE_TITLE_COLOR = Fore.MAGENTA + Style.BRIGHT
CODE_COLOR = Fore.WHITE
COMMENT_COLOR = Fore.CYAN + Style.BRIGHT + Style.DIM
RULE_COLOR = Fore.BLUE + Style.DIM
RESET_ALL = Style.RESET_ALL

# --- Configuration ---
query_string = "cyberpunk cityscape high resolution"  # A visually rich query
limit = 7  # Fetching a few more
output_dir = "visual_summons"  # A fitting directory name
safe_search_off = True  # The requested parameter

# --- The Summoning Ritual ---
print(RULE_COLOR + "━" * 50 + RESET_ALL)
print(TITLE_COLOR + "✨ Initiating Visual Image Summoning ✨" + RESET_ALL)
print(RULE_COLOR + "━" * 50 + RESET_ALL)

print(STATUS_COLOR + "\nConfiguration:" + RESET_ALL)
print(
    INFO_COLOR + "  Query: " + RESET_ALL + DATA_COLOR + f"'{query_string}'" + RESET_ALL
)
print(INFO_COLOR + "  Limit: " + RESET_ALL + DATA_COLOR + str(limit) + RESET_ALL)
print(
    INFO_COLOR + "  Output: " + RESET_ALL + DATA_COLOR + f"'{output_dir}'" + RESET_ALL
)
safe_search_status = "OFF" if safe_search_off else "ON"
safe_search_color = ERROR_TEXT_COLOR if safe_search_off else SUCCESS_TEXT_COLOR
print(
    INFO_COLOR
    + "  Safe Search: "
    + RESET_ALL
    + safe_search_color
    + safe_search_status
    + RESET_ALL
)
print("\n")

try:

    def update_status(text: str):
        print(STATUS_COLOR + "⏳ " + text + RESET_ALL + "...")

    update_status(f"Preparing to summon images of '{query_string}'")

    # --- The Core Download Call ---
    def progress_callback(count, total):
        if total:
            update_status(f"Summoning image {count}/{total}")
        else:
            update_status(f"Processing image {count}")

    downloader.download(
        query=query_string,
        limit=limit,
        output_dir=output_dir,
        adult_filter_off=safe_search_off,
        force_replace=False,
        timeout=60,
        verbose=False,  # Turn off default verbose, we'll show our own messages
        callback_fn=progress_callback,
    )
    # --- End Download Call ---

    time.sleep(0.5)
    update_status("Finalizing summoning")
    time.sleep(0.5)

    # --- Success Message ---
    print(RULE_COLOR + "━" * 50 + RESET_ALL)
    print(SUCCESS_TITLE_COLOR + "✨ Summoning Complete! ✨" + RESET_ALL)
    print(RULE_COLOR + "━" * 50 + RESET_ALL)
    success_message = f"Magnificent! {limit} images matching '{query_string}' summoned to '{output_dir}'."
    print(SUCCESS_TEXT_COLOR + success_message + RESET_ALL + "\n")


except Exception as e:
    # --- Error Message ---
    print(RULE_COLOR + "━" * 50 + RESET_ALL)
    print(ERROR_TITLE_COLOR + "❌ Summoning Error! ❌" + RESET_ALL)
    print(RULE_COLOR + "━" * 50 + RESET_ALL)
    print(ERROR_TEXT_COLOR + "The visual summoning ritual failed!" + RESET_ALL)
    print(
        ERROR_TEXT_COLOR
        + "Error: "
        + RESET_ALL
        + INFO_COLOR
        + str(e)
        + RESET_ALL
        + "\n"
    )
    print(ERROR_TEXT_COLOR + "\nDetailed Traceback:" + RESET_ALL)
    import traceback

    traceback.print_exc()  # Keeping it simple for Termux output

# --- Code Display Section ---
code_to_display = f"""\
from bing_image_downloader import downloader

# Configuration - Whispers of the Aether
query_string = "{query_string}" # The vision we seek
limit = {limit} # Number of images to materialize
output_dir = "{output_dir}" # Realm for the summoned images
safe_search_off = {safe_search_off} # Safe search OFF - Unleashed visions!

# Initiate download - Invoke the spell
downloader.download(
    query=query_string,
    limit=limit,
    output_dir=output_dir,
    adult_filter_off=safe_search_off,
    force_replace=False,
    timeout=60,
    verbose=False # Silent incantation, our status updates guide us
)

print(f"Successfully downloaded images to {{output_dir}}") # Echo of success
"""

print(RULE_COLOR + "━" * 50 + RESET_ALL)
print(CODE_TITLE_COLOR + "📜 The Python Spellbook Page 📜" + RESET_ALL)
print(RULE_COLOR + "━" * 50 + RESET_ALL)

code_lines = code_to_display.strip().split("\n")
for line in code_lines:
    if line.strip().startswith("#"):
        print(COMMENT_COLOR + line + RESET_ALL)  # Channeling the ether...
    elif "downloader.download" in line:
        print(CODE_COLOR + Fore.GREEN + line + RESET_ALL)  # Highlight the core spell
    elif (
        "query_string" in line
        or "limit" in line
        or "output_dir" in line
        or "safe_search_off" in line
    ):
        print(
            CODE_COLOR + Fore.YELLOW + line + RESET_ALL
        )  # Highlight configuration variables
    else:
        print(CODE_COLOR + line + RESET_ALL)  # Standard incantation text

print(RULE_COLOR + "━" * 50 + RESET_ALL)
print(TITLE_COLOR + "✨ Visual Incantation Concluded ✨" + RESET_ALL)
print(RULE_COLOR + "━" * 50 + RESET_ALL)
