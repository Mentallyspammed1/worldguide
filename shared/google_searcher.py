import sys
import os
import re
import argparse
import time

try:
    from googlesearch import search
except ImportError:
    print("\033[91m# The required library 'google' is not woven into your environment.\033[0m")
    print("\033[93m# Please cast the spell: pip install google\033[0m")
    sys.exit(1)

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("\033[91m# The 'colorama' library, which gives life to my words, is missing.\033[0m")
    print("\033[93m# Please cast the spell: pip install colorama\033[0m")
    # Define dummy Fore and Style if colorama is not present
    class DummyStyle:
        def __getattr__(self, name):
            return ""
    Fore = DummyStyle()
    Style = DummyStyle()

def sanitize_filename(query):
    """Sanitizes a string to be used as a valid filename."""
    sanitized = re.sub(r'[^\w\s-]', '', query).strip()
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized

def main():
    """
    Performs a Google search for a given query and saves the links to a file.
    """
    parser = argparse.ArgumentParser(
        description=f"{Fore.MAGENTA}A script to scry the digital aether (Google) for knowledge.{Style.RESET_ALL}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("query", type=str, help="The arcane words to search for.")
    parser.add_argument("--num", type=int, default=10, help="Number of results to conjure per page. [default: 10]")
    parser.add_argument("--stop", type=int, default=50, help="The final result to retrieve before ceasing the scrying. [default: 50]")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause in seconds between requests to appease the Google spirits. [default: 2.0]")
    parser.add_argument("--lang", type=str, default="en", help="The language of the ancient texts to seek. [default: en]")
    parser.add_argument("--tld", type=str, default="com", help="The top-level domain of the realm to search within. [default: com]")
    parser.add_argument("-o", "--output-dir", type=str, default="searches", help="Directory to store the resulting scrolls. [default: searches]")
    parser.add_argument("--tbs", type=str, default=None, help="Time-based search filter (e.g., 'qdr:h', 'qdr:d', 'qdr:w', 'qdr:m', 'qdr:y').")

    args = parser.parse_args()

    print(Fore.CYAN + f"# Summoning the ether for: '{args.query}'...")
    time.sleep(1)

    # --- Create output directory ---
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(Fore.GREEN + f"# A new repository for scrolls has been forged: '{args.output_dir}'")
        except OSError as e:
            print(Fore.RED + f"# A shadow falls! Could not forge the directory '{args.output_dir}': {e}")
            sys.exit(1)

    # --- Generate a safe filename ---
    filename = f"{sanitize_filename(args.query)}_links.txt"
    filepath = os.path.join(args.output_dir, filename)

    try:
        print(Fore.YELLOW + f"# Scrying... This may take a moment as I consult the digital winds.")
        # --- Perform the search ---
        search_results = list(search(
            args.query,
            tld=args.tld,
            lang=args.lang,
            num=args.num,
            stop=args.stop,
            pause=args.pause,
            tbs=args.tbs
        ))

        if not search_results:
            print(Fore.YELLOW + "# The ether is silent. No results were found for your query.")
            return

        # --- Inscribe results to the scroll ---
        with open(filepath, 'w', encoding='utf-8') as f:
            for url in search_results:
                f.write(f"{url}\n")

        print(Fore.GREEN + Style.BRIGHT + f"\n# Success! {len(search_results)} mystical links have been inscribed upon the scroll: {filepath}")

    except Exception as e:
        print(Fore.RED + Style.BRIGHT + f"\n# A great disturbance in the ether! The scrying has failed: {e}")
        print(Fore.YELLOW + "# This may be a transient issue, or the Google spirits may be wary of too many requests.")
        print(Fore.YELLOW + "# Consider increasing the --pause duration if this persists.")

if __name__ == "__main__":
    main()