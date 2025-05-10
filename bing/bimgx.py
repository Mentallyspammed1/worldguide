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
    UnidentifiedImageError = None  # type: ignore # type: ignore

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
    sanitized = '_'.join(filter(None, sanitized.split('_'))) # Collapse multiple underscores
    # Limit length
    return sanitized[:MAX_FILENAME_LENGTH]


def create_directory(path: str) -> bool:
    """Creates a directory if it doesn't exist. Returns True on success."""
    try:
        os.makedirs(path, exist_ok=True)
        # logger.debug(f"Ensured directory exists: {path}") # Can be verbose
        return True
    except OSError as e:
        print_error(f"Failed to create directory {path}: {e}")
        return False


def rename_files(file_paths: list[str], base_query: str) -> list[str]:
    """Renames downloaded files sequentially with a sanitized query prefix.
    Returns a list of final paths for files that were processed.
    If a file is renamed, its new path is in the list.
    If renaming failed or was skipped for an existing file, its original path is in the list.
    Files from input that were not found or were not files are skipped and not in the output list.
    """
    final_paths_after_rename: list[str] = []
    if not file_paths:
        print_warning("No file paths provided for renaming.")
        return []

    sanitized_query = sanitize_filename(base_query)
    if not sanitized_query:
        sanitized_query = "image"  # Fallback base name
        print_warning(f"Query '{base_query}' sanitized to empty string, using fallback 'image'.")

    # Assuming all files are in the same directory, get it from the first valid path.
    # This dir_name is used for joining new filenames.
    first_valid_dir = next((os.path.dirname(p) for p in file_paths if p), None)
    if first_valid_dir is None and file_paths: # All paths are empty strings or None, or file_paths is not empty but contains no valid paths
         print_error("Cannot determine directory for renaming files. All paths are invalid.")
         return list(file_paths) # Return original paths as is, though they are problematic
    elif first_valid_dir is None and not file_paths: # Should be caught by earlier check
         return []
    
    dir_name = first_valid_dir if first_valid_dir is not None else "." # Fallback to CWD if all paths were relative without dir

    print_info(f"Attempting to rename {len(file_paths)} files in '{dir_name}' with prefix '{sanitized_query}'...")

    # Sort to ensure consistent numbering (e.g., _1, _2, _3)
    # Original file_paths order might not be alphabetical/numerical.
    sorted_file_paths = sorted(file_paths)
    
    actual_renames_count = 0

    for idx, old_path in enumerate(
        tqdm(sorted_file_paths, desc=Fore.BLUE + "🔄 Renaming Files", unit="file", ncols=100, leave=False), start=1
    ):
        current_file_final_path = old_path  # Default to original path if any step fails or is skipped

        try:
            if not os.path.exists(old_path):
                print_warning(f"File not found for renaming (already renamed or deleted?): {old_path}")
                # This file won't be added to final_paths_after_rename if we `continue`
                # To ensure metadata step gets it (and potentially reports an error), we could add old_path.
                # However, the contract is to return paths of *processed* files.
                # If it's gone, it can't be processed.
                continue
            if not os.path.isfile(old_path):
                 print_warning(f"Path is not a file, skipping rename: {old_path}")
                 continue # Skip, not added to final_paths_after_rename

            _, ext = os.path.splitext(old_path)
            # Generate new name, ensuring it's within the same directory as the first file
            # (or CWD if dir_name was determined as '')
            new_base_name = f"{sanitized_query}_{idx}"
            new_filename = f"{new_base_name}{ext}"
            potential_new_path = os.path.join(dir_name, new_filename)

            target_path_for_rename = potential_new_path
            collision_counter = 1
            
            # Handle potential filename collisions
            while os.path.exists(target_path_for_rename):
                # Check if it's the *same* file we are trying to rename
                try:
                    if os.path.samefile(old_path, target_path_for_rename):
                        logger.debug(f"Skipping rename for {os.path.basename(old_path)} as target name '{os.path.basename(target_path_for_rename)}' is identical and points to the same file.")
                        # current_file_final_path remains old_path, no actual rename op needed
                        target_path_for_rename = old_path # Signal that no rename op should occur
                        break  # Exit the collision resolution while loop
                except FileNotFoundError: # old_path or target_path_for_rename might disappear
                    print_warning(f"File not found during samefile check: {old_path} or {target_path_for_rename}")
                    target_path_for_rename = old_path # Fallback if issue during check
                    break


                # If it's a different file, append counter to base name
                new_filename = f"{new_base_name}_{collision_counter}{ext}"
                target_path_for_rename = os.path.join(dir_name, new_filename)
                collision_counter += 1
                if collision_counter > 100:  # Safety break
                    print_error(f"Could not find unique name for {os.path.basename(old_path)} after 100 attempts. Skipping rename for this file.")
                    target_path_for_rename = old_path # Fallback to original path
                    break
            
            # At this point, target_path_for_rename is either a unique new path,
            # or old_path if it was samefile or collision resolution failed.

            if old_path == target_path_for_rename:
                # No rename operation needed (already correct, or samefile, or collision fallback)
                # current_file_final_path is already old_path
                pass
            else:
                # Attempt actual rename
                try:
                    os.rename(old_path, target_path_for_rename)
                    current_file_final_path = target_path_for_rename
                    actual_renames_count +=1
                except OSError as e:
                    print_error(f"Error renaming {os.path.basename(old_path)} to {os.path.basename(target_path_for_rename)}: {e}")
                    # current_file_final_path remains old_path
                except Exception as e: # Catch any other unexpected error during rename
                    print_error(f"Unexpected error renaming {os.path.basename(old_path)} to {os.path.basename(target_path_for_rename)}: {e}")
                    # current_file_final_path remains old_path
            
            final_paths_after_rename.append(current_file_final_path)

        except Exception as e: # Catch errors in the processing of a single file
            print_error(f"Unexpected error processing {os.path.basename(old_path)} for renaming: {e}. Retaining original path.")
            # Ensure even with unexpected errors, if the file existed at start, its original path is tracked
            if os.path.exists(old_path): # Check again, might have been affected by the error
                 final_paths_after_rename.append(old_path)
            # If it doesn't exist anymore, it's effectively skipped

    if final_paths_after_rename:
        # This counts files that were confirmed to exist and were processed (renamed or kept original name).
        # `actual_renames_count` tracks how many `os.rename` calls were successful.
        print_success(f"Processed {len(final_paths_after_rename)} files for renaming. Actual renames: {actual_renames_count}.")
    elif file_paths: # Input was not empty, but output is
        print_warning("No files were successfully processed for renaming (e.g., all source files missing).")
    # If file_paths was empty, initial warning already shown.
    
    return final_paths_after_rename


def apply_filters(**kwargs: str | None) -> str:
    """Generates Bing filter query parameters string (`+filterui:` syntax)
    based on the bing-image-downloader library's 'filter' parameter expectation.
    """
    filters: list[str] = []
    filter_map: dict[str, str] = {
        "size": "Size:{}",
        "color": "Color:{}",
        "type": "Type:{}",
        "layout": "Layout:{}",
        "people": "People:{}",
        "date": "Date:{}",
        "license": "License:{}",
    }

    for key, value in kwargs.items():
        if value and value.strip():
            formatted_value = value.strip()
            # Bing expects precise casing for some filters, e.g. 'ColorOnly', 'Monochrome'
            # Capitalizing the first letter is a common pattern but not universal.
            # For simplicity, we'll rely on user input for correct casing or library/Bing flexibility.
            # However, common ones can be normalized:
            if key == "color":
                if formatted_value.lower() == "coloronly": formatted_value = "ColorOnly"
                elif formatted_value.lower() == "monochrome": formatted_value = "Monochrome"
                else: formatted_value = formatted_value.capitalize() # Default for other colors
            elif key in ["size", "type", "layout", "people", "date", "license"]:
                 # Most Bing filters capitalize the value (e.g. Size:Large, Type:Photo)
                 formatted_value = formatted_value.capitalize()


            if template := filter_map.get(key):
                filters.append(template.format(formatted_value))
            else:
                print_warning(f"Unknown filter key '{key}' provided.")

    return "+".join(filters)


# --- Core Functions ---
def download_images_with_bing(
    query: str,
    output_dir_base: str,
    limit: int,
    timeout: int,
    adult_filter_off: bool,
    extra_filters: str,
    site_filter: str | None = None
) -> list[str]:
    """Handles image downloading using bing-image-downloader and returns actual file paths."""
    effective_query = query
    if site_filter:
        effective_query += f" site:{site_filter}"

    # The library creates a subdirectory named *exactly* after the 'query' argument
    # passed to its download() method. This will be 'effective_query'.
    # This name might contain characters problematic for some filesystems or expectations,
    # but we must use what the library uses to find the files.
    query_based_subdir_name = effective_query
    query_specific_output_dir = os.path.join(output_dir_base, query_based_subdir_name)

    downloaded_files: list[str] = []
    try:
        print_info(f"Starting download for query: '{Fore.YELLOW}{effective_query}{Fore.CYAN}'")
        print_info(f"Output target directory for this query: '{query_specific_output_dir}'")
        print_info(f"Applying filters: '{extra_filters}'" if extra_filters else "No extra filters.")

        downloader.download(
            query=effective_query, # This string is used for the subdirectory name by the library
            limit=limit,
            output_dir=output_dir_base,
            adult_filter_off=adult_filter_off,
            force_replace=False,
            timeout=timeout,
            filter=extra_filters,
            verbose=False # Our script handles primary feedback
        )
        print_success("bing-image-downloader process finished.")

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
                          "Download might have failed, produced no results, or the query string led to an unexpected directory structure.")

    except KeyboardInterrupt:
        print_warning("Download interrupted by user.")
        raise
    except Exception as e:
        print_error(f"Download failed using bing-image-downloader: {e}")
        logger.debug("Traceback for downloader error:", exc_info=True)
        return []

    return downloaded_files


def get_local_file_metadata(file_path: str) -> dict[str, Any]:
    """Extracts metadata (size, dimensions) from a local image file."""
    metadata: dict[str, Any] = {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "file_size_bytes": None,
        "dimensions": None,
        "error": None
    }
    try:
        if not os.path.exists(file_path):
            metadata["error"] = "File does not exist at time of metadata extraction"
            # Warning here can be too verbose if many files failed renaming and don't exist
            # logger.warning(f"File not found for metadata: {metadata['filename']}")
            return metadata
        if not os.path.isfile(file_path):
             metadata["error"] = "Path is not a file"
             # logger.warning(f"Path is not a file, skipping metadata: {metadata['filename']}")
             return metadata

        try:
            metadata["file_size_bytes"] = os.path.getsize(file_path)
        except OSError as size_err:
             err_msg = f"OS error getting size: {size_err}"
             metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
             print_warning(f"Could not get size for {metadata['filename']}: {size_err}")

        if PIL_AVAILABLE and Image and UnidentifiedImageError:
            try:
                with Image.open(file_path) as img:
                    metadata["dimensions"] = f"{img.width}x{img.height}"
            except UnidentifiedImageError:
                err_msg = "Cannot identify image file (PIL)"
                metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
            except Exception as img_err: # Catch other PIL errors
                err_msg = f"Error reading image dimensions (PIL): {img_err}"
                metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
                print_warning(f"Could not get dimensions for {metadata['filename']} (PIL): {img_err}")
        elif not PIL_AVAILABLE:
            err_msg = "Pillow not installed (for dimensions)"
            metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
            # Warning printed once at start of parallel extraction

    except OSError as e: # Catch OS errors related to file_path access itself
        err_msg = f"OS error accessing file for metadata: {e}"
        metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
        print_error(f"Error accessing {metadata['filename']} for metadata: {e}")
    except Exception as e:
        err_msg = f"Unexpected error getting metadata for {metadata['filename']}: {e}"
        metadata["error"] = f"{metadata['error']}; {err_msg}" if metadata['error'] else err_msg
        print_error(err_msg) # Log the specific error

    return metadata


def extract_metadata_parallel(image_paths: list[str]) -> list[dict[str, Any]]:
    """Extracts local file metadata for multiple images in parallel."""
    if not image_paths:
        return []

    if not PIL_AVAILABLE:
        print_warning("Pillow library not found. Image dimensions will not be extracted.")
        print_warning("Install it using: pip install Pillow")

    metadata_list: list[dict[str, Any]] = []
    max_workers = min(16, (os.cpu_count() or 1) * 2 + 4)

    print_info(f"Extracting metadata from {len(image_paths)} local files using up to {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_local_file_metadata, path): path for path in image_paths}

        for future in tqdm(futures.keys(), total=len(image_paths), desc=Fore.BLUE + "📄 Extracting Metadata", unit="file", ncols=100, leave=False):
            original_path = futures[future]
            try:
                result = future.result()
                metadata_list.append(result)
            except Exception as e:
                print_error(f"Error processing future result for {os.path.basename(original_path)}: {e}")
                metadata_list.append({
                    "file_path": original_path,
                    "filename": os.path.basename(original_path),
                    "file_size_bytes": None,
                    "dimensions": None,
                    "error": f"Future processing error: {e}"
                })

    # Sort metadata list by filename for consistent output, if desired
    # metadata_list.sort(key=lambda x: x.get("filename", ""))
    print_success(f"Metadata extraction completed for {len(metadata_list)} files.")
    return metadata_list


def save_metadata(metadata_list: list[dict[str, Any]], output_dir_base: str, query: str) -> bool:
    """Saves the collected metadata list to a JSON file in the base output directory."""
    if not metadata_list:
        print_warning("No metadata collected to save.")
        return False

    sanitized_query_for_filename = sanitize_filename(query)
    if not sanitized_query_for_filename:
        sanitized_query_for_filename = "unknown_query"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_filename = f"{METADATA_FILENAME_PREFIX}{sanitized_query_for_filename}_{timestamp}.json"
    metadata_file_path = os.path.join(output_dir_base, metadata_filename)

    print_info(f"Attempting to save metadata to: {metadata_file_path}")
    try:
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

    while True:
        query = input(Fore.CYAN + "⌨️  Enter Search Query (e.g., 'red cars', 'mountain landscape'): " + Fore.WHITE).strip()
        if query:
            inputs["query"] = query
            break
        else:
            print_warning("Search query cannot be empty.")

    output_dir_base = input(
        Fore.CYAN + f"📂 Enter Base Output Directory (images go into a query-specific subfolder here) [default: {DEFAULT_OUTPUT_DIR}]: " + Fore.WHITE
    ).strip() or DEFAULT_OUTPUT_DIR
    inputs["output_dir_base"] = output_dir_base

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

    adult_filter_off_input = input(Fore.CYAN + "🔞 Disable adult filter? (y/N): " + Fore.WHITE).strip().lower()
    inputs["adult_filter_off"] = adult_filter_off_input == 'y'

    print_header("🎨 Search Filters (Optional - Press Enter to skip)")
    print_info("Examples: Size:Large, Type:Photo, Color:Monochrome, License:ShareCommercially")
    print_info("Common values - Size: Small, Medium, Large, Wallpaper")
    print_info("Color: ColorOnly, Monochrome | Type: Photo, Clipart, Line, AnimatedGif, Transparent")
    print_info("Layout: Square, Wide, Tall | People: Face, Portrait")
    print_info("Date: PastDay, PastWeek, PastMonth, PastYear")
    print_info("License: Any, Public, Share, ShareCommercially, Modify, ModifyCommercially")
    
    inputs["filters"] = {
        "size": input(Fore.CYAN + "📏 Size: " + Fore.WHITE).strip(),
        "color": input(Fore.CYAN + "🎨 Color: " + Fore.WHITE).strip(),
        "type": input(Fore.CYAN + "🖼️  Type: " + Fore.WHITE).strip(),
        "layout": input(Fore.CYAN + "📐 Layout: " + Fore.WHITE).strip(),
        "people": input(Fore.CYAN + "👥 People: " + Fore.WHITE).strip(),
        "date": input(Fore.CYAN + "📅 Date: " + Fore.WHITE).strip(),
        "license": input(Fore.CYAN + "📜 License: " + Fore.WHITE).strip(),
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
    print_info("Dependencies: requests, bing-image-downloader, colorama, tqdm, Pillow (optional for dimensions)")
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
        site_filter: str | None = user_inputs["site_filter"] or None # Ensure None if empty

        if not create_directory(output_dir_base):
            print_error(f"Cannot proceed without base output directory: {output_dir_base}")
            sys.exit(1)

        filter_string = apply_filters(**filters_dict)
        print_header("🚀 Starting Process")

        downloaded_file_paths = download_images_with_bing(
            query, output_dir_base, limit, timeout, adult_filter_off, filter_string, site_filter
        )

        if not downloaded_file_paths:
            print_warning("No images were downloaded or found. Check query, filters, or permissions. Exiting.")
            return

        # renamed_file_paths_after_processing will contain final paths of successfully processed files
        # (either new name or original name if rename failed/skipped but file exists).
        # It will be shorter than downloaded_file_paths if some files disappeared before renaming.
        renamed_file_paths_after_processing = rename_files(downloaded_file_paths, query)

        paths_for_metadata = renamed_file_paths_after_processing
        if not paths_for_metadata and downloaded_file_paths:
            # This case: download happened, but rename_files returned empty (e.g., all files vanished before processing)
            # We can still try to get metadata from original paths if they somehow reappeared or if rename_files had an issue.
            print_warning("Renaming process yielded no usable file paths. "
                          "Attempting metadata extraction on original downloaded paths if they exist.")
            paths_for_metadata = [p for p in downloaded_file_paths if os.path.exists(p) and os.path.isfile(p)]


        metadata = []
        if paths_for_metadata:
             metadata = extract_metadata_parallel(paths_for_metadata)
        else:
            print_warning("No valid file paths available after download/rename steps to extract metadata from.")

        if metadata:
            save_metadata(metadata, output_dir_base, query)
        elif paths_for_metadata: # files existed, but metadata extraction yielded nothing (e.g. all failed)
            print_warning("Metadata extraction failed for all processed files or yielded no data.")
        else: # No paths_for_metadata, so metadata list is also empty
            print_warning("No metadata was generated to save as no files were available.")


        print_header("📊 Results Summary")
        total_initial_downloads = len(downloaded_file_paths)
        # `renamed_file_paths_after_processing` contains paths of files that rename_files could account for.
        total_accounted_for_after_rename = len(renamed_file_paths_after_processing)
        total_metadata_extracted = len(metadata) # Should match total_accounted_for_after_rename if all metadata ops succeed
        errors_in_metadata = sum(1 for item in metadata if item.get("error"))

        print_info(f"Initial files found after download: {total_initial_downloads}")
        print_info(f"Files accounted for after renaming attempt: {total_accounted_for_after_rename}")
        print_info(f"Metadata records generated: {total_metadata_extracted}")
        if errors_in_metadata > 0:
            metadata_file_name_part = f"{METADATA_FILENAME_PREFIX}{sanitize_filename(query)}"
            print_warning(f"Encountered errors during metadata extraction for {errors_in_metadata} file(s). "
                          f"Check '{metadata_file_name_part}_*.json' for details in 'error' fields.")

        if metadata:
            print_info("First few metadata entries (up to 5):")
            for item in metadata[:min(5, len(metadata))]:
                size_str = f"{item.get('file_size_bytes', 'N/A')} bytes"
                dim_str = item.get('dimensions', 'N/A')
                error_str = f" {Fore.RED}(Error: {item['error']})" if item.get("error") else ""
                print(f"  - {Fore.MAGENTA}{item.get('filename', 'N/A')}{Style.RESET_ALL}: Size: {size_str}, Dims: {dim_str}{error_str}")
            if len(metadata) > 5:
                print(f"  ... and {len(metadata) - 5} more entries in the JSON file.")
        elif total_accounted_for_after_rename > 0: # Files were there, but metadata list is empty
             print_warning("No metadata was extracted, though files were present after renaming.")
        else:
            print_warning("No metadata to display as no files were processed for metadata.")


        end_time = datetime.now()
        duration = end_time - start_time
        print_success(f"\nOperation completed in {duration.total_seconds():.2f} seconds!")

    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user (Ctrl+C detected).")
        sys.exit(130) # Standard exit code for SIGINT
    except Exception as e:
        print_error(f"\nAn unexpected critical error occurred: {e}")
        logger.exception("Unhandled exception trace:")
        sys.exit(1)


if __name__ == "__main__":
    main()
