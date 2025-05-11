import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from bing_image_downloader import downloader
from colorama import Back, Fore, Style, init
from tqdm import tqdm

# Initialize Colorama
init(autoreset=True)


# Configure Colored Logging
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        message = super().format(record)
        return f"{color}{message}"


logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.handlers = [handler]
logger.setLevel(logging.INFO)


# Color Print Functions
def print_header(text):
    print(Fore.YELLOW + Style.BRIGHT + "\n" + "═" * 50)
    print(Fore.YELLOW + Style.BRIGHT + f"  {text}")
    print(Fore.YELLOW + Style.BRIGHT + "═" * 50 + "\n")


def print_success(text):
    print(Fore.GREEN + Style.BRIGHT + "✓ " + text)


def print_warning(text):
    print(Fore.YELLOW + Style.BRIGHT + "! " + text)


def print_error(text):
    print(Fore.RED + Style.BRIGHT + "✗ " + text)


def print_info(text):
    print(Fore.CYAN + Style.NORMAL + "➤ " + text)


# Utility Functions
def validate_date(date_str):
    """Validates date strings in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        print_error(f"Invalid date format: {date_str}. Please use YYYY-MM-DD.")
        return False


def format_date_for_bing(date_str):
    """Formats date for Bing search URL."""
    if not validate_date(date_str):
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")


def create_directory(path):
    """Creates directory with error handling."""
    try:
        os.makedirs(path, exist_ok=True)
        print_success(f"Directory created: {path}")
    except Exception as e:
        print_error(f"Failed to create directory {path}: {e}")
        sys.exit(1)


def rename_files(file_paths, query):
    """Renames files with progress tracking."""
    renamed_paths = []
    for idx, path in enumerate(
        tqdm(file_paths, desc=Fore.BLUE + "🔄 Renaming Files", unit="file"), start=1
    ):
        try:
            new_name = f"{query.replace(' ', '_')}_{idx}{os.path.splitext(path)[1]}"
            new_path = os.path.join(os.path.dirname(path), new_name)
            os.rename(path, new_path)
            renamed_paths.append(new_path)
        except Exception as e:
            print_error(f"Error renaming {path}: {e}")
    return renamed_paths


def apply_filters(**kwargs):
    """Generates Bing filter query parameters."""
    filters = []
    filter_map = {
        "size": "imagesize-{}",
        "layout": "photo-{}",
        "date_filter": None,
        "location": "location:{}",
        "site": "site:{}",
        "file_type": "photo-{}",
    }

    for key, value in kwargs.items():
        if value:
            if key == "date_filter":
                filters.append(value)
            elif template := filter_map.get(key):
                filters.append(template.format(value.lower()))

    return "&qft=" + "+".join(filters) if filters else ""


# Core Functions
def download_images(query, output_dir, limit, timeout, adult_filter_off, filters):
    """Handles image downloading with progress tracking."""
    try:
        print_info(f"Starting download for {Fore.YELLOW}{query}")
        return downloader.download(
            query,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=adult_filter_off,
            timeout=timeout,
            filter=filters,
        )
    except Exception as e:
        print_error(f"Download failed: {e}")
        return None


def extract_metadata(image_paths):
    """Extracts image metadata with parallel processing."""
    metadata_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_image_metadata, path) for path in image_paths]
        for i, future in enumerate(
            tqdm(futures, desc=Fore.BLUE + "📄 Extracting Metadata", unit="file")
        ):
            try:
                metadata_list.append({"path": image_paths[i], **future.result()})
            except Exception as e:
                print_error(f"Error processing {image_paths[i]}: {e}")
    return metadata_list


def get_image_metadata(url):
    """Fetches image metadata from headers."""
    try:
        response = requests.head(url, timeout=10)
        return {
            "content_length": response.headers.get("content-length", "N/A"),
            "content_type": response.headers.get("content-type", "N/A"),
        }
    except requests.RequestException as e:
        print_error(f"Metadata fetch failed for {url}: {e}")
        return {"content_length": "N/A", "content_type": "N/A"}


def save_metadata(metadata_list, output_dir):
    """Saves metadata to JSON file."""
    metadata_file = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata_list, f, indent=4)
        print_success(f"Metadata saved to {metadata_file}")
    except OSError as e:
        print_error(f"Failed to save metadata: {e}")


# Main Application
def main():
    print_header("🌟 Enhanced Bing Image Downloader 🌟")

    # Get user inputs with color prompts
    query = input(Fore.CYAN + "⌨  Search Query: " + Fore.WHITE).strip()
    output_dir = (
        input(
            Fore.CYAN + "📂 Output Directory (default: downloads): " + Fore.WHITE
        ).strip()
        or "downloads"
    )
    output_dir = os.path.join(output_dir, query.replace(" ", "_"))
    create_directory(output_dir)

    # Numerical inputs
    try:
        limit = int(input(Fore.CYAN + "🔢 Max Images to Download: " + Fore.WHITE))
        timeout = int(input(Fore.CYAN + "⏳ Timeout (seconds): " + Fore.WHITE))
    except ValueError:
        print_error("Invalid numerical input")
        sys.exit(1)

    adult_filter_off = (
        input(Fore.CYAN + "🔞 Disable adult filter? (y/n): " + Fore.WHITE).lower()
        == "y"
    )

    # Filter inputs
    print_header("🎨 Search Filters")
    filters = {
        "size": input(
            Fore.CYAN + "📏 Size (small/medium/large/wallpaper): " + Fore.WHITE
        ).strip(),
        "layout": input(
            Fore.CYAN + "🖼  Layout (square/wide/tall): " + Fore.WHITE
        ).strip(),
        "site": input(
            Fore.CYAN + "🌐 Site filter (example.com): " + Fore.WHITE
        ).strip(),
        "location": input(
            Fore.CYAN + "📍 Location (e.g., 'New York'): " + Fore.WHITE
        ).strip(),
        "file_type": input(
            Fore.CYAN + "📁 File Type (jpg/png/gif): " + Fore.WHITE
        ).strip(),
        "date_filter": input(
            Fore.CYAN + "📅 Date Range (YYYYMMDD..YYYYMMDD): " + Fore.WHITE
        ).strip(),
    }
    filter_query = apply_filters(**filters)

    # Download process
    try:
        with tqdm(
            total=limit, desc=Fore.BLUE + "📥 Downloading Images", unit="img"
        ) as pbar:
            image_paths = download_images(
                query, output_dir, limit, timeout, adult_filter_off, filter_query
            )
            if image_paths:
                pbar.update(len(image_paths))

        if not image_paths:
            print_error(
                "No images downloaded. Check filters or try different keywords."
            )
            return

        renamed_paths = rename_files(image_paths, query)
        metadata = extract_metadata(renamed_paths)

        print_header("📄 Download Results")
        for item in metadata:
            print(Fore.MAGENTA + f"📷 {os.path.basename(item['path'])}")
            print(Fore.CYAN + f"   Size: {item.get('content_length', 'N/A')} bytes")
            print(Fore.CYAN + f"   Type: {item.get('content_type', 'N/A')}\n")

        save_metadata(metadata, output_dir)
        print_success("Operation completed successfully!")

    except KeyboardInterrupt:
        print_error("Operation cancelled by user!")
        sys.exit(1)


if __name__ == "__main__":
    main()
