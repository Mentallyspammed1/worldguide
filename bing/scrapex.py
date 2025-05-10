#!/usr/bin/env python3

"""Enhanced Bing Image Downloader Script
-------------------------------------
Downloads images from Bing based on user queries and filters,
renames them sequentially, extracts local file metadata (size, dimensions),
and saves the metadata to a JSON file.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Third-party Libraries
from bing_image_downloader import downloader
from colorama import Back, Fore, Style, init
from tqdm import tqdm

# Attempt to import Pillow for image metadata; provide guidance if missing
try:
    from PIL import Image, UnidentifiedImageError
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    UnidentifiedImageError = None  # type: ignore

# --- Constants ---
DEFAULT_OUTPUT_DIR: str = "downloads"
MAX_FILENAME_LENGTH: int = 200  # Max length for base filename derived from query

# --- Initialize Colorama ---
init(autoreset=True)


# --- Configure Colored Logging ---
class ColoredFormatter(logging.Formatter):
    """Custom logging formatter with colors."""
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record with appropriate colors."""
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        message = super().format(record)
        # Apply color only to the message part if needed, or whole line
        # return f"{color}{message}{Style.RESET_ALL}" # Color whole line
        log_fmt = f"%(asctime)s - {color}%(levelname)s{Style.RESET_ALL} - %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


logger = logging.getLogger(__name__)  # Use __name__ for logger hierarchy
logger.propagate = False  # Prevent duplicate logging if root logger is configured
if not logger.handlers:  # Avoid adding handler multiple times if script is re-run/imported
    handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# --- Color Print Functions (Simplified Wrappers around Logging/Printing) ---
def print_header(text: str) -> None:
    """Prints a formatted header."""
    bar = "═" * 50
    print(Fore.YELLOW + Style.BRIGHT + f"\n{bar}")
    print(Fore.YELLOW + Style.BRIGHT + f"  {text}")
    print(Fore.YELLOW + Style.BRIGHT + f"{bar}\n")


def print_success(text: str) -> None:
    """Prints a success message."""
    logger.info(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text: str) -> None:
    """Prints a warning message."""
    logger.warning(f"{Fore.YELLOW}! {text}{Style.RESET_ALL}")


def print_error(text: str) -> None:
    """Prints an error message."""
    logger.error(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text: str) -> None:
    """Prints an informational message."""
    # Using logger.info already provides timestamp and level, so keep it simple
    # Or use a dedicated print for non-log info if preferred
    print(Fore.CYAN + Style.NORMAL + f"➤ {text}")


# --- Utility Functions ---
def sanitize_filename(name: str) -> str:
    """Removes or replaces characters invalid for filenames and truncates."""
    # Remove characters invalid in most file systems
    sanitized = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    sanitized = sanitized.replace(' ', '_')  # Replace spaces with underscores
    # Limit length
    return sanitized[:MAX_FILENAME_LENGTH]


def create_directory(path: str) -> bool:
    """Creates a directory if it doesn't exist. Returns True on success."""
    try:
        os.makedirs(path, exist_ok=True)
        # print_success(f"Ensured directory exists: {path}") # Less verbose
        return True
    except OSError as e:
        print_error(f"Failed to create directory {path}: {e}")
        return False


def rename_files(file_paths: list[str], base_query: str) -> list[str]:
    """Renames downloaded files sequentially with a sanitized query prefix."""
    renamed_paths: list[str] = []
    sanitized_query = sanitize_filename(base_query)
    if not file_paths:
        return []

    print_info("Renaming downloaded files...")
    for idx, old_path in enumerate(
        tqdm(file_paths, desc=Fore.BLUE + "🔄 Renaming Files", unit="file", ncols=100), start=1
    ):
        try:
            if not os.path.exists(old_path):
                print_warning(f"File not found for renaming: {old_path}")
                continue

            _, ext = os.path.splitext(old_path)
            dir_name = os.path.dirname(old_path)
            new_name = f"{sanitized_query}_{idx}{ext}"
            new_path = os.path.join(dir_name, new_name)

            # Handle potential filename collisions (though unlikely with sequential numbering)
            counter = 1
            while os.path.exists(new_path):
                new_name = f"{sanitized_query}_{idx}_{counter}{ext}"
                new_path = os.path.join(dir_name, new_name)
                counter += 1

            os.rename(old_path, new_path)
            renamed_paths.append(new_path)
        except OSError as e:
            print_error(f"Error renaming {os.path.basename(old_path)}: {e}")
        except Exception as e:
            print_error(f"Unexpected error renaming {os.path.basename(old_path)}: {e}")

    if renamed_paths:
        print_success(f"Renamed {len(renamed_paths)} files.")
    return renamed_paths


def apply_filters(**kwargs: str | None) -> str:
    """Generates Bing filter query parameters string (`+filterui:` syntax).
    Focuses on filters commonly supported via this method.
    Reference: Bing search advanced options often use 'filterui:' prefix.
    """
    filters: list[str] = []
    filter_map: dict[str, str] = {
        # Maps user input key to Bing filter syntax part
        "size": "imagesize:{}",
        "layout": "aspect:{}",  # Common aspect ratios
        "type": "photo:{}",  # file type filter
        "license": "license:{}",
        # Add more filters here if known and tested
    }
    # Site filter is usually handled differently (site:example.com in query)
    # but bing-image-downloader might have a dedicated param or handle it in query

    for key, value in kwargs.items():
        if value and value.strip():
            value = value.strip().lower()
            if template := filter_map.get(key):
                filters.append(template.format(value))

    return "+".join(filters) if filters else ""  # Bing uses '+' as separator in filterui


# --- Core Functions ---
def download_images_with_bing(
    query: str,
    output_dir: str,
    limit: int,
    timeout: int,
    adult_filter_off: bool,
    extra_filters: str,
    site_filter: str | None = None
) -> list[str]:
    """Handles image downloading using bing-image-downloader."""
    effective_query = query
    if site_filter:
        effective_query += f" site:{site_filter}"

    downloaded_files: list[str] = []
    try:
        print_info(f"Starting download for query: '{Fore.YELLOW}{effective_query}{Fore.CYAN}'")
        # bing-image-downloader manages its own progress/output
        # It returns None on success, raises Exception on failure.
        # It downloads files into a subdirectory named after the query within output_dir.
        # We need the actual paths of the downloaded files.
        downloader.download(
            query=effective_query,
            limit=limit,
            output_dir=output_dir,
            adult_filter_off=adult_filter_off,
            force_replace=False,  # Don't overwrite existing files with same name
            timeout=timeout,
            filter=extra_filters,  # Pass the constructed filter string
            verbose=False  # Let our script handle most verbosity
        )

        # --- Find downloaded files ---
        # The library creates a subdir named exactly like the query.
        # Need to handle spaces/special chars in query matching the dir name.
        # The library might do some sanitization, let's assume it matches query for now.
        query_based_subdir = os.path.join(output_dir, query)  # Default behavior of the library

        if os.path.isdir(query_based_subdir):
             # List files directly in the created subdirectory
            for filename in os.listdir(query_based_subdir):
                full_path = os.path.join(query_based_subdir, filename)
                if os.path.isfile(full_path):
                    downloaded_files.append(full_path)
            print_success(f"Download process initiated for {len(downloaded_files)} images (check console for details from downloader).")
        else:
            print_warning(f"Could not find expected download subdirectory: {query_based_subdir}. No files processed.")
            # Attempt to list files directly in output_dir as a fallback? Might be messy.

    except KeyboardInterrupt:
        raise  # Propagate KeyboardInterrupt
    except Exception as e:
        print_error(f"Download failed using bing-image-downloader: {e}")
        # Consider more specific error handling based on library's exceptions if available
        return []  # Return empty list on failure

    return downloaded_files


def get_local_file_metadata(file_path: str) -> dict[str, Any]:
    """Extracts metadata (size, dimensions) from a local image file."""
    metadata: dict[str, Any] = {
        "file_size_bytes": "N/A",
        "dimensions": "N/A",
        "error": None
    }
    try:
        if not os.path.exists(file_path):
            metadata["error"] = "File does not exist"
            return metadata

        # Get file size
        metadata["file_size_bytes"] = os.path.getsize(file_path)

        # Get image dimensions using Pillow
        if PIL_AVAILABLE and Image:
            try:
                with Image.open(file_path) as img:
                    metadata["dimensions"] = f"{img.width}x{img.height}"
            except UnidentifiedImageError:
                metadata["error"] = "Cannot identify image file (possibly corrupt or not an image)"
                print_warning(f"Could not get dimensions for {os.path.basename(file_path)}: Not identified as image.")
            except Exception as img_err:
                metadata["error"] = f"Error reading image dimensions: {img_err}"
                print_warning(f"Could not get dimensions for {os.path.basename(file_path)}: {img_err}")
        elif not PIL_AVAILABLE:
            metadata["dimensions"] = "Pillow not installed"
            metadata["error"] = "Pillow not installed"  # Add error state

    except OSError as e:
        metadata["error"] = f"OS error accessing file: {e}"
        print_error(f"Error accessing {os.path.basename(file_path)} for metadata: {e}")
    except Exception as e:
        metadata["error"] = f"Unexpected error getting metadata: {e}"
        print_error(f"Unexpected error processing {os.path.basename(file_path)}: {e}")

    return metadata


def extract_metadata_parallel(image_paths: list[str]) -> list[dict[str, Any]]:
    """Extracts local file metadata for multiple images in parallel."""
    if not image_paths:
        return []

    if not PIL_AVAILABLE:
        print_warning("Pillow library not found. Image dimensions will not be extracted.")
        print_warning("Install it using: pip install Pillow")

    metadata_list: list[dict[str, Any]] = []
    max_workers = min(10, os.cpu_count() or 1 + 4)  # Sensible default for I/O bound tasks

    print_info("Extracting metadata from local files...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = [executor.submit(get_local_file_metadata, path) for path in image_paths]

        # Process results as they complete
        for i, future in enumerate(
            tqdm(futures, desc=Fore.BLUE + "📄 Extracting Metadata", unit="file", ncols=100)
        ):
            try:
                result = future.result()
                metadata_list.append({
                    "original_path_recorded": image_paths[i],  # Record path used for extraction
                    "filename": os.path.basename(image_paths[i]),
                    **result  # Add size, dimensions, error
                    })
            except Exception as e:
                # Should ideally be caught within get_local_file_metadata, but catch here too
                print_error(f"Error processing future for {os.path.basename(image_paths[i])}: {e}")
                metadata_list.append({
                    "original_path_recorded": image_paths[i],
                    "filename": os.path.basename(image_paths[i]),
                    "file_size_bytes": "N/A",
                    "dimensions": "N/A",
                    "error": f"Future processing error: {e}"
                })

    print_success(f"Metadata extraction completed for {len(metadata_list)} files.")
    return metadata_list


def save_metadata(metadata_list: list[dict[str, Any]], output_dir: str, query: str) -> bool:
    """Saves the collected metadata list to a JSON file."""
    if not metadata_list:
        print_warning("No metadata collected to save.")
        return False

    sanitized_query = sanitize_filename(query)
    metadata_filename = f"metadata_{sanitized_query}.json"
    metadata_file_path = os.path.join(output_dir, metadata_filename)

    try:
        with open(metadata_file_path, "w", encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
        print_success(f"Metadata saved successfully to: {metadata_file_path}")
        return True
    except OSError as e:
        print_error(f"Failed to save metadata to {metadata_file_path}: {e}")
        return False
    except Exception as e:
        print_error(f"An unexpected error occurred while saving metadata: {e}")
        return False


# --- User Input Function ---
def get_user_input() -> dict[str, Any]:
    """Gets and validates user input for the download process."""
    inputs: dict[str, Any] = {}

    print_header("🔍 Input Parameters")

    # Query
    while True:
        query = input(Fore.CYAN + "⌨️  Enter Search Query: " + Fore.WHITE).strip()
        if query:
            inputs["query"] = query
            break
        else:
            print_warning("Search query cannot be empty.")

    # Output Directory
    output_dir_base = input(
        Fore.CYAN + f"📂 Enter Base Output Directory (default: {DEFAULT_OUTPUT_DIR}): " + Fore.WHITE
    ).strip() or DEFAULT_OUTPUT_DIR
    inputs["output_dir"] = output_dir_base  # The final dir will include the query subdir

    # Numerical Inputs (Limit, Timeout)
    while True:
        try:
            limit_str = input(Fore.CYAN + "🔢 Max Images to Download (e.g., 50): " + Fore.WHITE).strip()
            limit = int(limit_str)
            if limit > 0:
                inputs["limit"] = limit
                break
            else:
                print_warning("Number of images must be positive.")
        except ValueError:
            print_error("Invalid input. Please enter a whole number.")

    while True:
        try:
            timeout_str = input(Fore.CYAN + "⏳ Download Timeout per image (seconds, e.g., 60): " + Fore.WHITE).strip()
            timeout = int(timeout_str)
            if timeout > 0:
                inputs["timeout"] = timeout
                break
            else:
                print_warning("Timeout must be positive.")
        except ValueError:
            print_error("Invalid input. Please enter a whole number.")

    # Adult Filter
    adult_filter_off_input = input(Fore.CYAN + "🔞 Disable adult filter? (y/N): " + Fore.WHITE).strip().lower()
    inputs["adult_filter_off"] = adult_filter_off_input == 'y'

    # Filter Inputs
    print_header("🎨 Search Filters (Optional)")
    print_info("Leave blank to ignore a filter.")
    inputs["filters"] = {
        "size": input(
            Fore.CYAN + "📏 Size (e.g., small, medium, large, wallpaper): " + Fore.WHITE
        ).strip(),
        "layout": input(
            Fore.CYAN + "🖼️  Layout/Aspect (e.g., square, wide, tall): " + Fore.WHITE
        ).strip(),
         "type": input(
            Fore.CYAN + "🖼️  Type (e.g., photo, clipart, line, animatedgif): " + Fore.WHITE
        ).strip(),
         "license": input(
            Fore.CYAN + "📜 License (e.g., any, public, share, sharecommercially, modify, modifycommercially): " + Fore.WHITE
        ).strip(),
        # Add more filter prompts here if needed
    }
    inputs["site_filter"] = input(
            Fore.CYAN + "🌐 Filter by specific site (e.g., example.com): " + Fore.WHITE
        ).strip()

    return inputs


# --- Main Application ---
def main() -> None:
    """Main function to orchestrate the image downloading and processing."""
    print_header("🌟 Enhanced Bing Image Downloader 🌟")
    print_info("This script requires: requests, bing-image-downloader, colorama, tqdm, Pillow")
    if not PIL_AVAILABLE:
        print_warning("Pillow library not installed. Image dimensions cannot be extracted.")
        print_warning("Consider installing with: pip install Pillow")

    try:
        user_inputs = get_user_input()

        query: str = user_inputs["query"]
        output_dir_base: str = user_inputs["output_dir"]
        limit: int = user_inputs["limit"]
        timeout: int = user_inputs["timeout"]
        adult_filter_off: bool = user_inputs["adult_filter_off"]
        filters_dict: dict[str, str] = user_inputs["filters"]
        site_filter: str | None = user_inputs["site_filter"]

        # Construct the final output directory path (base + query subdir)
        # The downloader library creates the query-specific subdir.
        # We primarily need the base directory for organization and metadata saving.
        if not create_directory(output_dir_base):
            sys.exit(1)  # Exit if base directory creation fails

        # Prepare filter string for the downloader
        filter_string = apply_filters(**filters_dict)

        print_header("🚀 Starting Process")

        # --- Download ---
        # Note: download_images_with_bing returns paths within the query-specific subdir
        downloaded_file_paths = download_images_with_bing(
            query, output_dir_base, limit, timeout, adult_filter_off, filter_string, site_filter
        )

        if not downloaded_file_paths:
            print_error("No images were downloaded or found. Check query, filters, or permissions.")
            return  # Exit gracefully if download fails or yields no files

        # --- Rename ---
        # Pass the *actual* downloaded paths for renaming
        # The renaming happens within the query-specific subdirectory
        renamed_paths = rename_files(downloaded_file_paths, query)

        if not renamed_paths:
            print_warning("No files were successfully renamed.")
            # Decide whether to proceed with metadata extraction on original names or stop
            # Let's try extracting metadata from whatever paths we have (original or renamed)
            paths_for_metadata = downloaded_file_paths  # Fallback to original if renaming failed
        else:
             paths_for_metadata = renamed_paths

        # --- Extract Metadata ---
        # Use the list of *final* file paths (renamed if successful)
        metadata = extract_metadata_parallel(paths_for_metadata)

        # --- Save Metadata ---
        # Save metadata in the *base* output directory for better organization
        save_metadata(metadata, output_dir_base, query)

        # --- Final Summary ---
        print_header("📊 Results Summary")
        if metadata:
            print_info(f"Processed {len(metadata)} files. Metadata saved.")
            # Optionally print a few examples
            for item in metadata[:min(5, len(metadata))]:  # Print first 5
                print(f"  - {Fore.MAGENTA}{item['filename']}{Style.RESET_ALL}: "
                      f"Size: {item.get('file_size_bytes', 'N/A')} bytes, "
                      f"Dims: {item.get('dimensions', 'N/A')}"
                      f"{Fore.RED + ' (Error: ' + item['error'] + ')' if item.get('error') else ''}")
            if len(metadata) > 5:
                print(f"  ... and {len(metadata) - 5} more.")
        else:
            print_warning("No metadata was extracted.")

        print_success("\nOperation completed successfully!")

    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user (Ctrl+C detected).")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nAn unexpected critical error occurred: {e}")
        logger.exception("Unhandled exception trace:")  # Log full traceback for debugging
        sys.exit(1)


if __name__ == "__main__":
    main()
