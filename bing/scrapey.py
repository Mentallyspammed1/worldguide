#!/usr/bin/env python3

"""Enhanced Bing Image Downloader Script
-------------------------------------
Downloads images from Bing based on user queries and filters,
renames them sequentially within a query-specific subfolder,
extracts local file metadata (size, dimensions),
and saves the metadata to a JSON file in the base output directory.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

# Third-party Libraries
# requests is used implicitly by bing_image_downloader
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
METADATA_FILENAME_PREFIX: str = "metadata_"

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
        log_fmt = (
            f"%(asctime)s - {self.COLORS.get(record.levelname, Fore.WHITE)}"
            f"%(levelname)s{Style.RESET_ALL} - %(message)s"
        )
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# Setup Logger
logger = logging.getLogger(__name__)  # Use __name__ for logger hierarchy
logger.propagate = False  # Prevent duplicate logging if root logger is configured
if not logger.handlers:  # Avoid adding handler multiple times
    handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    handler.setFormatter(ColoredFormatter())  # Use custom formatter
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set default logging level


# --- User Feedback Functions ---
def print_header(text: str) -> None:
    """Prints a formatted header."""
    bar = "═" * (len(text) + 4)
    print(Fore.YELLOW + Style.BRIGHT + f"\n{bar}")
    print(Fore.YELLOW + Style.BRIGHT + f"  {text}  ")
    print(Fore.YELLOW + Style.BRIGHT + f"{bar}\n")


def print_success(text: str) -> None:
    """Logs a success message."""
    logger.info(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")


def print_warning(text: str) -> None:
    """Logs a warning message."""
    logger.warning(f"{Fore.YELLOW}! {text}{Style.RESET_ALL}")


def print_error(text: str) -> None:
    """Logs an error message."""
    logger.error(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")


def print_info(text: str) -> None:
    """Prints an informational message directly (distinct from logs)."""
    print(Fore.CYAN + Style.NORMAL + f"➤ {text}")


# --- Utility Functions ---
def sanitize_filename(name: str) -> str:
    """Removes/replaces invalid filename chars and truncates."""
    # Remove characters invalid in most file systems
    sanitized = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    # Replace spaces with underscores and ensure no leading/trailing underscores
    sanitized = '_'.join(filter(None, sanitized.split(' ')))
    sanitized = '_'.join(filter(None, sanitized.split('_')))
    # Limit length
    return sanitized[:MAX_FILENAME_LENGTH]


def create_directory(path: str) -> bool:
    """Creates a directory if it doesn't exist. Returns True on success."""
    try:
        os.makedirs(path, exist_ok=True)
        # logger.debug(f"Ensured directory exists: {path}") # Less verbose
        return True
    except OSError as e:
        print_error(f"Failed to create directory {path}: {e}")
        return False


def rename_files(file_paths: list[str], base_query: str) -> list[str]:
    """Renames downloaded files sequentially with a sanitized query prefix."""
    renamed_paths: list[str] = []
    if not file_paths:
        print_warning("No file paths provided for renaming.")
        return []

    sanitized_query = sanitize_filename(base_query)
    if not sanitized_query:
        sanitized_query = "image"  # Fallback base name
        print_warning(f"Query '{base_query}' sanitized to empty string, using fallback 'image'.")

    # Get the directory from the first file path (assuming all are in the same dir)
    if not os.path.dirname(file_paths[0]):
         print_error("Cannot determine directory for renaming files.")
         return []
    dir_name = os.path.dirname(file_paths[0])

    print_info(f"Renaming {len(file_paths)} files in '{dir_name}' with prefix '{sanitized_query}'...")

    file_paths.sort()  # Ensure consistent ordering if needed

    for idx, old_path in enumerate(
        tqdm(file_paths, desc=Fore.BLUE + "🔄 Renaming Files", unit="file", ncols=100, leave=False), start=1
    ):
        try:
            if not os.path.exists(old_path):
                print_warning(f"File not found for renaming (already renamed or deleted?): {old_path}")
                continue
            if not os.path.isfile(old_path):
                 print_warning(f"Path is not a file, skipping rename: {old_path}")
                 continue

            _, ext = os.path.splitext(old_path)
            # Generate new name, ensuring it's within the same directory
            new_name = f"{sanitized_query}_{idx}{ext}"
            new_path = os.path.join(dir_name, new_name)

            # Handle potential filename collisions (unlikely but possible)
            counter = 1
            original_new_path = new_path
            while os.path.exists(new_path):
                # Check if it's the *same* file we are trying to rename (no actual collision)
                if os.path.samefile(old_path, new_path):
                    logger.debug(f"Skipping rename for {os.path.basename(old_path)} as target name is identical.")
                    renamed_paths.append(old_path)  # Keep track of it
                    break  # Exit the while loop for this file

                # If it's a different file, append counter
                new_name = f"{sanitized_query}_{idx}_{counter}{ext}"
                new_path = os.path.join(dir_name, new_name)
                counter += 1
                if counter > 100:  # Safety break
                    print_error(f"Could not find unique name for {os.path.basename(old_path)} after 100 attempts. Skipping.")
                    new_path = None  # Indicate failure
                    break
            else:  # Only executes if the while loop completes normally (no break)
                try:
                    os.rename(old_path, new_path)
                    renamed_paths.append(new_path)
                except OSError as e:
                    print_error(f"Error renaming {os.path.basename(old_path)} to {os.path.basename(new_path)}: {e}")
                except Exception as e:
                    print_error(f"Unexpected error renaming {os.path.basename(old_path)}: {e}")

        except Exception as e:
            print_error(f"Unexpected error processing {os.path.basename(old_path)} for renaming: {e}")

    if renamed_paths:
        print_success(f"Successfully processed {len(renamed_paths)} files for renaming (check warnings for issues).")
    else:
        print_warning("No files were successfully renamed.")
    return renamed_paths


def apply_filters(**kwargs: str | None) -> str:
    """Generates Bing filter query parameters string (`+filterui:` syntax)
    based on the bing-image-downloader library's 'filter' parameter expectation.
    Note: The library might handle some filters differently. This function
    formats common ones known to work with Bing's `filterui` syntax.
    """
    filters: list[str] = []
    # Map user-friendly keys to Bing filter keys expected by the library or Bing itself
    filter_map: dict[str, str] = {
        "size": "Size:{}",        # e.g., Size:Medium
        "color": "Color:{}",      # e.g., Color:ColorOnly, Color:Monochrome
        "type": "Type:{}",        # e.g., Type:Photo, Type:Clipart
        "layout": "Layout:{}",    # e.g., Layout:Square, Layout:Wide
        "people": "People:{}",    # e.g., People:Face, People:Portrait
        "date": "Date:{}",        # e.g., Date:PastWeek
        "license": "License:{}",  # e.g., License:ShareCommercially
    }
    # Reference: Check bing-image-downloader documentation or source for exact keys if available.
    # Common values:
    # Size: Small, Medium, Large, Wallpaper
    # Color: ColorOnly, Monochrome
    # Type: Photo, Clipart, Line, AnimatedGif, Transparent
    # Layout: Square, Wide, Tall
    # People: Face, Portrait
    # Date: PastDay, PastWeek, PastMonth, PastYear
    # License: Any, Public, Share, ShareCommercially, Modify, ModifyCommercially (or variations like Free)

    for key, value in kwargs.items():
        if value and value.strip():
            # Format the value correctly (e.g., capitalize first letter for some filters)
            formatted_value = value.strip().capitalize()
            # Special handling if needed (e.g., license might need different casing)
            if key == "license":
                 # Example: Library might expect 'ShareCommercially' or 'sharecommercially'
                 # Adjust based on observed behavior or documentation. Let's assume capitalize is a safe default.
                 pass  # Keep capitalized for now
            elif key == "color" and formatted_value.lower() == "monochrome":
                formatted_value = "Monochrome"  # Ensure correct casing
            elif key == "color" and formatted_value.lower() == "coloronly":
                 formatted_value = "ColorOnly"

            if template := filter_map.get(key):
                filters.append(template.format(formatted_value))
            else:
                print_warning(f"Unknown filter key '{key}' provided.")

    # The library expects a simple string with '+' separators for its 'filter' parameter.
    return "+".join(filters)


# --- Core Functions ---
def download_images_with_bing(
    query: str,
    output_dir_base: str,  # The base directory provided by user
    limit: int,
    timeout: int,
    adult_filter_off: bool,
    extra_filters: str,
    site_filter: str | None = None
) -> list[str]:
    """Handles image downloading using bing-image-downloader and returns actual file paths."""
    effective_query = query
    if site_filter:
        # Append site filter to the query string, as expected by the library
        effective_query += f" site:{site_filter}"

    # The library creates a subdirectory named *exactly* after the 'query' argument
    # inside the 'output_dir_base'. Special characters in the query might affect the folder name.
    # We need to use the original `query` (before `site:` filter addition) to predict the subdir name.
    query_based_subdir_name = query  # Library uses the raw query for subdir name
    query_specific_output_dir = os.path.join(output_dir_base, query_based_subdir_name)

    downloaded_files: list[str] = []
    try:
        print_info(f"Starting download for query: '{Fore.YELLOW}{effective_query}{Fore.CYAN}'")
        print_info(f"Output target directory: '{query_specific_output_dir}'")
        print_info(f"Applying filters: '{extra_filters}'" if extra_filters else "No extra filters.")

        # Call the downloader. It manages its own progress/output.
        # It downloads files into the 'query_specific_output_dir'.
        # It returns None on success, raises Exception on failure.
        downloader.download(
            query=effective_query,  # Use the query potentially modified with site:
            limit=limit,
            output_dir=output_dir_base,  # Library prepends this to the query subdir
            adult_filter_off=adult_filter_off,
            force_replace=False,  # Don't overwrite existing files
            timeout=timeout,
            filter=extra_filters,  # Pass the constructed filter string
            verbose=False  # Let our script handle primary feedback
        )
        # If download didn't raise an exception, it likely completed or partially completed.
        print_success("bing-image-downloader process finished.")

        # --- Find downloaded files ---
        print_info(f"Checking for downloaded files in: {query_specific_output_dir}")
        if os.path.isdir(query_specific_output_dir):
            found_count = 0
            for filename in os.listdir(query_specific_output_dir):
                full_path = os.path.join(query_specific_output_dir, filename)
                if os.path.isfile(full_path):
                    downloaded_files.append(full_path)
                    found_count += 1
            if found_count > 0:
                print_success(f"Found {found_count} downloaded file(s) in the target directory.")
            else:
                print_warning(f"Download process finished, but no files were found in {query_specific_output_dir}. "
                              "Check downloader logs or query/filter validity.")
        else:
            print_warning(f"Could not find expected download subdirectory: {query_specific_output_dir}. "
                          "Download might have failed silently or placed files elsewhere.")

    except KeyboardInterrupt:
        print_warning("Download interrupted by user.")
        raise  # Propagate KeyboardInterrupt to main loop
    except Exception as e:
        print_error(f"Download failed using bing-image-downloader: {e}")
        logger.debug("Traceback for downloader error:", exc_info=True)
        return []  # Return empty list on failure

    return downloaded_files


def get_local_file_metadata(file_path: str) -> dict[str, Any]:
    """Extracts metadata (size, dimensions) from a local image file."""
    metadata: dict[str, Any] = {
        "file_path": file_path,  # Record the path used for extraction
        "filename": os.path.basename(file_path),
        "file_size_bytes": None,
        "dimensions": None,  # Format "WxH"
        "error": None
    }
    try:
        if not os.path.exists(file_path):
            metadata["error"] = "File does not exist at time of metadata extraction"
            print_warning(f"File not found for metadata: {metadata['filename']}")
            return metadata
        if not os.path.isfile(file_path):
             metadata["error"] = "Path is not a file"
             print_warning(f"Path is not a file, skipping metadata: {metadata['filename']}")
             return metadata

        # Get file size
        try:
            metadata["file_size_bytes"] = os.path.getsize(file_path)
        except OSError as size_err:
             metadata["error"] = f"OS error getting size: {size_err}"
             print_warning(f"Could not get size for {metadata['filename']}: {size_err}")
             # Continue to try getting dimensions if possible

        # Get image dimensions using Pillow
        if PIL_AVAILABLE and Image:
            try:
                with Image.open(file_path) as img:
                    metadata["dimensions"] = f"{img.width}x{img.height}"
            except UnidentifiedImageError:
                err_msg = "Cannot identify image file (PIL)"
                metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
                # Don't log warning here, let the summary show the error
            except Exception as img_err:
                err_msg = f"Error reading image dimensions: {img_err}"
                metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
                print_warning(f"Could not get dimensions for {metadata['filename']}: {img_err}")
        elif not PIL_AVAILABLE:
            err_msg = "Pillow not installed"
            metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
            # Warning printed once during parallel extraction start

    except OSError as e:
        err_msg = f"OS error accessing file: {e}"
        metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
        print_error(f"Error accessing {metadata['filename']} for metadata: {e}")
    except Exception as e:
        err_msg = f"Unexpected error getting metadata: {e}"
        metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
        print_error(f"Unexpected error processing {metadata['filename']}: {e}")

    return metadata


def extract_metadata_parallel(image_paths: list[str]) -> list[dict[str, Any]]:
    """Extracts local file metadata for multiple images in parallel."""
    if not image_paths:
        return []

    if not PIL_AVAILABLE:
        print_warning("Pillow library not found. Image dimensions will not be extracted.")
        print_warning("Install it using: pip install Pillow")

    metadata_list: list[dict[str, Any]] = []
    # Adjust max_workers based on typical I/O bound nature
    max_workers = min(16, (os.cpu_count() or 1) * 2 + 4)

    print_info(f"Extracting metadata from {len(image_paths)} local files using up to {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {executor.submit(get_local_file_metadata, path): path for path in image_paths}

        # Process results as they complete with tqdm progress bar
        for future in tqdm(futures, total=len(image_paths), desc=Fore.BLUE + "📄 Extracting Metadata", unit="file", ncols=100, leave=False):
            original_path = futures[future]
            try:
                result = future.result()
                metadata_list.append(result)
            except Exception as e:
                # This catch is a fallback; errors should ideally be handled within get_local_file_metadata
                print_error(f"Error processing future result for {os.path.basename(original_path)}: {e}")
                metadata_list.append({
                    "file_path": original_path,
                    "filename": os.path.basename(original_path),
                    "file_size_bytes": None,
                    "dimensions": None,
                    "error": f"Future processing error: {e}"
                })

    print_success(f"Metadata extraction completed for {len(metadata_list)} files.")
    return metadata_list


def save_metadata(metadata_list: list[dict[str, Any]], output_dir_base: str, query: str) -> bool:
    """Saves the collected metadata list to a JSON file in the base output directory."""
    if not metadata_list:
        print_warning("No metadata collected to save.")
        return False

    sanitized_query = sanitize_filename(query)
    if not sanitized_query:
        sanitized_query = "unknown_query"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_filename = f"{METADATA_FILENAME_PREFIX}{sanitized_query}_{timestamp}.json"
    metadata_file_path = os.path.join(output_dir_base, metadata_filename)

    print_info(f"Attempting to save metadata to: {metadata_file_path}")
    try:
        # Ensure the base directory exists (should already, but double-check)
        if not create_directory(output_dir_base):
             print_error(f"Cannot save metadata, base output directory '{output_dir_base}' does not exist and couldn't be created.")
             return False

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
        query = input(Fore.CYAN + "⌨️  Enter Search Query (e.g., 'red cars', 'mountain landscape'): " + Fore.WHITE).strip()
        if query:
            inputs["query"] = query
            break
        else:
            print_warning("Search query cannot be empty.")

    # Output Directory
    output_dir_base = input(
        Fore.CYAN + f"📂 Enter Base Output Directory (images go into a subfolder named after query here) [default: {DEFAULT_OUTPUT_DIR}]: " + Fore.WHITE
    ).strip() or DEFAULT_OUTPUT_DIR
    inputs["output_dir_base"] = output_dir_base

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
    print_header("🎨 Search Filters (Optional - Press Enter to skip)")
    print_info("Examples: Size:Large, Type:Photo, Layout:Wide, License:Share")
    inputs["filters"] = {
        # Keys should match the `filter_map` in `apply_filters`
        "size": input(
            Fore.CYAN + "📏 Size (Small, Medium, Large, Wallpaper): " + Fore.WHITE
        ).strip(),
        "color": input(
            Fore.CYAN + "🎨 Color (ColorOnly, Monochrome): " + Fore.WHITE
        ).strip(),
        "type": input(
            Fore.CYAN + "🖼️  Type (Photo, Clipart, Line, AnimatedGif, Transparent): " + Fore.WHITE
        ).strip(),
        "layout": input(
            Fore.CYAN + "📐 Layout (Square, Wide, Tall): " + Fore.WHITE
        ).strip(),
        "people": input(
            Fore.CYAN + "👥 People (Face, Portrait): " + Fore.WHITE
        ).strip(),
        "date": input(
            Fore.CYAN + "📅 Date (PastDay, PastWeek, PastMonth, PastYear): " + Fore.WHITE
        ).strip(),
        "license": input(
            Fore.CYAN + "📜 License (Any, Public, Share, ShareCommercially, Modify, ModifyCommercially): " + Fore.WHITE
        ).strip(),
    }
    inputs["site_filter"] = input(
            Fore.CYAN + "🌐 Filter by specific site (e.g., wikipedia.org, flickr.com): " + Fore.WHITE
        ).strip()

    return inputs


# --- Main Application ---
def main() -> None:
    """Main function to orchestrate the image downloading and processing."""
    start_time = datetime.now()
    print_header("🌟 Enhanced Bing Image Downloader 🌟")
    print_info("Dependencies: requests, bing-image-downloader, colorama, tqdm, Pillow")
    if not PIL_AVAILABLE:
        print_warning("Pillow library not installed. Image dimensions cannot be extracted.")
        print_warning("Install using: pip install Pillow")

    try:
        user_inputs = get_user_input()

        query: str = user_inputs["query"]
        output_dir_base: str = user_inputs["output_dir_base"]
        limit: int = user_inputs["limit"]
        timeout: int = user_inputs["timeout"]
        adult_filter_off: bool = user_inputs["adult_filter_off"]
        filters_dict: dict[str, str] = user_inputs["filters"]
        site_filter: str | None = user_inputs["site_filter"]

        # Ensure the base output directory exists *before* downloading
        if not create_directory(output_dir_base):
            print_error(f"Cannot proceed without base output directory: {output_dir_base}")
            sys.exit(1)

        # Prepare filter string for the downloader
        filter_string = apply_filters(**filters_dict)

        print_header("🚀 Starting Process")

        # --- Download ---
        # Returns list of full paths to successfully downloaded files
        # Files are located in 'output_dir_base / query_subdir'
        downloaded_file_paths = download_images_with_bing(
            query, output_dir_base, limit, timeout, adult_filter_off, filter_string, site_filter
        )

        if not downloaded_file_paths:
            print_warning("No images were downloaded or found. Check query, filters, or permissions. Exiting.")
            return  # Exit gracefully

        # --- Rename ---
        # Pass the actual downloaded paths for renaming. Renaming happens in-place.
        # Returns list of paths *after* successful renaming attempt.
        renamed_paths = rename_files(downloaded_file_paths, query)

        # Decide which paths to use for metadata extraction
        if not renamed_paths:
            print_warning("Renaming failed or yielded no results. Attempting metadata extraction on original downloaded paths.")
            paths_for_metadata = downloaded_file_paths
        else:
            # Even if some renames failed, use the list returned by rename_files,
            # as it contains paths of successfully renamed files or original paths if rename was skipped/failed per file.
            paths_for_metadata = renamed_paths

        # --- Extract Metadata ---
        # Use the final list of file paths
        metadata = []
        if paths_for_metadata:
             metadata = extract_metadata_parallel(paths_for_metadata)
        else:
            print_warning("No valid file paths remaining after download/rename steps to extract metadata from.")

        # --- Save Metadata ---
        # Save metadata in the *base* output directory for organization
        if metadata:
            save_metadata(metadata, output_dir_base, query)
        else:
            print_warning("No metadata was generated to save.")

        # --- Final Summary ---
        print_header("📊 Results Summary")
        total_downloaded = len(downloaded_file_paths)
        total_renamed = len(renamed_paths)  # This count might be less if renaming failed for some
        total_metadata_extracted = len(metadata)
        errors_in_metadata = sum(1 for item in metadata if item.get("error"))

        print_info(f"Initial files found after download: {total_downloaded}")
        print_info(f"Files successfully processed for renaming: {total_renamed}")  # May include files not actually renamed if names conflicted/identical
        print_info(f"Metadata records generated: {total_metadata_extracted}")
        if errors_in_metadata > 0:
            print_warning(f"Encountered errors during metadata extraction for {errors_in_metadata} file(s). Check '{METADATA_FILENAME_PREFIX}{sanitize_filename(query)}_*.json'.")

        if metadata:
            print_info("First few metadata entries:")
            for item in metadata[:min(5, len(metadata))]:  # Print first 5
                size_str = f"{item.get('file_size_bytes', 'N/A')} bytes"
                dim_str = item.get('dimensions', 'N/A')
                error_str = f"{Fore.RED}(Error: {item['error']})" if item.get("error") else ""
                print(f"  - {Fore.MAGENTA}{item['filename']}{Style.RESET_ALL}: Size: {size_str}, Dims: {dim_str} {error_str}")
            if len(metadata) > 5:
                print(f"  ... and {len(metadata) - 5} more.")
        else:
            print_warning("No metadata was extracted or saved.")

        end_time = datetime.now()
        duration = end_time - start_time
        print_success(f"\nOperation completed in {duration.total_seconds():.2f} seconds!")

    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user (Ctrl+C detected).")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nAn unexpected critical error occurred: {e}")
        logger.exception("Unhandled exception trace:")  # Log full traceback
        sys.exit(1)


if __name__ == "__main__":
    main()
