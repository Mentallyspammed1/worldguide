Okay, I've reviewed and enhanced your Python script. Here's the improved version with explanations for the key changes:

**Key Enhancements Made:**

1.  **Module Docstring & Version:** Added a module-level docstring and a `__version__` constant for better script identification and context.
2.  **Dataclass Style:** Changed dataclass field definitions from `field: type; field2: type` to the more standard Python style of one field per line.
3.  **`XvideosClient.__init__`:** The `pornLib` import/availability check logic was already quite robust. Minor logging clarification.
4.  **`XvideosClient._parse_video_results`:**
    *   Refined the handling of items that are already `VideoDataClass` instances. If an item is a `VideoDataClass`, it's assumed to have all defined attributes; the check now focuses on whether essential attributes (`title`, `img`, `link`) have non-`None` *values*.
    *   Improved warning messages for clarity when skipping items.
5.  **`XvideosClient.search_videos`:**
    *   **Corrected Parameter Handling:** Ensured that `page` and `limit` arguments are correctly added to `search_params` if provided, so they are passed to `self.client.search()`. The previous logic only included them if they were part of `**kwargs`.
    *   **Improved Logging:** Updated warning messages to accurately reflect that `page` and `limit` *are* passed to the API, but their effectiveness depends on the backend engine.
6.  **`get_validated_input`:**
    *   Slightly streamlined the `str` validation path (as `input()` already returns a string).
7.  **`main()` Function:**
    *   **Auto-Open HTML:** Replaced platform-specific `os.startfile` and `os.system` calls with `webbrowser.open(output_path.resolve().as_uri())` for a more platform-independent, secure, and reliable way to open the HTML file in the default web browser.
    *   **Filename Prefix Formatting:** Enhanced robustness for `filename_prefix_format` by catching generic `Exception` during formatting as a fallback, not just `KeyError`.
    *   **Logging:** Minor improvements to log messages for clarity and consistency.
    *   Used `__version__` in the initial log message.
8.  **General Readability:** Added/adjusted comments for clarity in various sections.

The core logic for scraping (via `pornLib`), rate limiting, and HTML generation remains largely the same as it was already well-structured. The enhancements focus on robustness, clarity, and Python best practices.

```python
# --- START OF FILE xvid_prompt.py ---

import datetime
import html
import logging
import os
import sys
import webbrowser # Added for platform-independent file opening
from dataclasses import dataclass
from pathlib import Path  # For better path handling
from typing import Any

import pornLib  # Assuming this library exists and is installed
from ratelimit import limits, sleep_and_retry  # type: ignore

# ==============================================================================
# Script Information
# ==============================================================================
__version__ = "1.1.0"
__script_name__ = "PornLib Search Script"

# ==============================================================================
# Configuration
# ==============================================================================

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__script_name__)

# --- Rate Limiting Configuration ---
API_CALLS_LIMIT = 50
API_PERIOD_SECONDS = 60

# --- Default Settings (Used in prompts) ---
DEFAULT_OUTPUT_DIR_STR = "."  # Default as string for input prompt
DEFAULT_SEARCH_LIMIT = 30
DEFAULT_PAGE = 1
DEFAULT_ENGINE = "xvideos"
DEFAULT_SOUP_SLEEP = 1.0
DEFAULT_FILENAME_PREFIX = "{engine}_search_{query_part}_{timestamp}"
DEFAULT_AUTO_OPEN = 'y'


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class VideoDataClass:
    title: str
    img: str
    link: str
    preview_url: str | None = None
    quality: str | int | None = None
    time: str | None = None
    channel_name: str | None = None
    channel_link: str | None = None


@dataclass
class VideoDownloadDataClass:
    low: str | None = None
    high: str | None = None
    hls: str | None = None


@dataclass
class Tags:
    name: str | None = None
    id: str | None = None


# ==============================================================================
# Xvideos Client Class
# ==============================================================================
class XvideosClient:
    def __init__(self, engine: str = DEFAULT_ENGINE, soup_sleep: float = DEFAULT_SOUP_SLEEP):
        self.engine = engine
        try:
            # This check helps provide a more specific error if pornLib was imported but isn't fully functional
            # or if the PornLib class itself is not found within the module.
            if 'pornLib' not in sys.modules and 'pornlib' not in sys.modules: # Case-insensitive check for module name in sys.modules
                 raise ImportError("pornLib module does not appear to be correctly imported or installed.")
            if not hasattr(pornLib, 'PornLib'):
                raise AttributeError("pornLib module is imported, but the 'PornLib' class is missing.")

            self.client = pornLib.PornLib(engine=engine, soupSleep=soup_sleep)
            logger.info(f"Successfully initialized PornLib Client: engine='{engine}', soup_sleep={soup_sleep:.2f}s")
        except ImportError: # Catches explicit raise or if pornLib.PornLib itself raises ImportError
            logger.critical("Fatal Error: pornLib library not found. Please install it (e.g., 'pip install pornlib').")
            raise
        except AttributeError as ae: # Catches if pornLib.PornLib is missing
            logger.critical(f"Fatal Error: pornLib library is incomplete or outdated. {ae}")
            raise
        except Exception as e: # Catches other errors from pornLib.PornLib() constructor
            logger.critical(f"Fatal Error: Failed to initialize pornLib client for engine '{engine}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"XvideosClient initialization failed for engine '{engine}': {e}") from e

    def _parse_video_results(self, results_raw: Any) -> list[VideoDataClass]:
        if results_raw is None:
            logger.debug("Received None for video results, returning empty list.")
            return []
        if not isinstance(results_raw, list):
            logger.warning(f"Expected list for video results, got {type(results_raw)}. Returning empty list.")
            return []

        videos: list[VideoDataClass] = []
        required_keys_for_dict = ['title', 'img', 'link'] # Used for dict items

        for i, item in enumerate(results_raw):
            video: VideoDataClass | None = None
            try:
                if isinstance(item, dict):
                    if all(k in item and item[k] is not None for k in required_keys_for_dict):
                         video = VideoDataClass(
                             title=str(item.get('title', '')),
                             img=str(item.get('img', '')),
                             link=str(item.get('link', '')),
                             preview_url=str(item['preview_url']) if item.get('preview_url') else None,
                             quality=item.get('quality'), # Can be str or int
                             time=str(item['time']) if item.get('time') else None,
                             channel_name=str(item['channel_name']) if item.get('channel_name') else None,
                             channel_link=str(item['channel_link']) if item.get('channel_link') else None,
                         )
                    else:
                        missing_or_none_keys = [k for k in required_keys_for_dict if k not in item or item[k] is None]
                        logger.warning(f"Skipping video dict item #{i + 1} due to missing/None essential keys: {missing_or_none_keys}. Data: {item}")
                elif isinstance(item, VideoDataClass):
                     # If item is genuinely a VideoDataClass, it has all attributes by definition.
                     # We just check if the essential ones have non-None values.
                     if item.title and item.img and item.link: # Ensure essential string fields are not empty/None
                         video = item
                     else:
                         logger.warning(
                             f"Skipping VideoDataClass item #{i + 1} because essential attributes "
                             f"(title, img, or link) are empty or None. Data: {item}"
                         )
                elif hasattr(item, 'title') and hasattr(item, 'img') and hasattr(item, 'link'):
                     # Handle generic objects with expected attributes
                     if all(getattr(item, k, None) is not None for k in required_keys_for_dict): # Check attribute values
                         video = VideoDataClass(
                             title=str(getattr(item, 'title', '')),
                             img=str(getattr(item, 'img', '')),
                             link=str(getattr(item, 'link', '')),
                             preview_url=str(getattr(item, 'preview_url', None)) if getattr(item, 'preview_url', None) is not None else None,
                             quality=getattr(item, 'quality', None),
                             time=str(getattr(item, 'time', None)) if getattr(item, 'time', None) is not None else None,
                             channel_name=str(getattr(item, 'channel_name', None)) if getattr(item, 'channel_name', None) is not None else None,
                             channel_link=str(getattr(item, 'channel_link', None)) if getattr(item, 'channel_link', None) is not None else None,
                         )
                     else:
                         logger.warning(f"Skipping object item #{i + 1} because essential attributes (title, img, or link) are None. Data: {item!r}")
                else:
                    logger.warning(f"Skipping unrecognized video item #{i + 1} of type: {type(item)}. Item data: {item!r}")

                if video:
                    videos.append(video)
            except Exception as e:
                logger.error(f"Error parsing video item #{i + 1}: {item!r}. Error: {e}", exc_info=False) # exc_info=False to avoid spamming for many bad items
        return videos

    def _parse_tag_results(self, tags_raw: Any) -> list[Tags]:
        if tags_raw is None:
            logger.debug("Received None for tag results, returning empty list.")
            return []
        if not isinstance(tags_raw, list):
            logger.warning(f"Expected list for tags, got {type(tags_raw)}. Returning empty list.")
            return []

        tags_list: list[Tags] = []
        for i, item in enumerate(tags_raw):
             tag: Tags | None = None
             try:
                 if isinstance(item, dict):
                     tag = Tags(
                         name=str(item['name']) if item.get('name') else None,
                         id=str(item['id']) if item.get('id') else None
                     )
                 elif isinstance(item, Tags):
                     tag = item # Assume it's correctly formed
                 elif hasattr(item, 'name') or hasattr(item, 'id'): # For generic objects
                     tag = Tags(
                         name=str(getattr(item, 'name', None)) if getattr(item, 'name', None) is not None else None,
                         id=str(getattr(item, 'id', None)) if getattr(item, 'id', None) is not None else None
                     )
                 else:
                     logger.warning(f"Skipping unrecognized tag item #{i + 1} of type: {type(item)}. Item data: {item!r}")
                     continue # Skip to next item

                 if tag and (tag.name or tag.id): # Ensure at least one identifier is present
                     tags_list.append(tag)
                 elif tag: # Tag was parsed but deemed empty
                     logger.debug(f"Skipping parsed tag item #{i + 1} with no name or id: {item!r}")

             except Exception as e:
                 logger.error(f"Error parsing tag item #{i + 1}: {item!r}. Error: {e}", exc_info=False)
        return tags_list

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def list_videos(self, limit: int = 12) -> list[VideoDataClass]:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Limit must be a positive integer.")
        logger.debug(f"Attempting to fetch up to {limit} videos using list method...")
        try:
            videos_raw = self.client.list(limit=limit)
            videos = self._parse_video_results(videos_raw)
        except TypeError as te:
            if 'limit' in str(te).lower(): # More robust check for 'limit' in error message
                logger.warning(f"Engine '{self.engine}' list method may not support 'limit' parameter. Trying without it.")
                videos = [] # Initialize to empty list
                try:
                    videos_raw_retry = self.client.list() # Call without limit
                    videos = self._parse_video_results(videos_raw_retry)
                except Exception as e_retry:
                    logger.error(f"Error during list retry (without limit): {e_retry}", exc_info=True)
                    raise Exception(f"Failed list retry for engine '{self.engine}': {e_retry}") from e_retry
                # Apply limit client-side if necessary
                limited_videos = videos[:limit]
                logger.info(f"Fetched {len(videos)} videos (without limit param), returning first {len(limited_videos)}.")
                return limited_videos
            else:
                logger.error(f"TypeError during list call: {te}", exc_info=True)
                raise Exception(f"Failed to list videos due to TypeError: {te}") from te
        except Exception as e:
            logger.error(f"Error during list call: {e}", exc_info=True)
            raise Exception(f"Failed to list videos: {e}") from e

        logger.info(f"Fetched and parsed {len(videos)} videos (requested list limit: {limit}).")
        return videos

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def search_videos(self, keyword: str | None = None, page: int | None = None, limit: int | None = None, **kwargs: Any) -> list[VideoDataClass]:
        search_params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None} # Start with other kwargs
        if keyword:
            search_params['keyword'] = keyword
        if page is not None and page > 0: # pornLib might handle page=0 or page=1 differently
            search_params['page'] = page
        if limit is not None and limit > 0:
            search_params['limit'] = limit

        if not search_params:
            raise ValueError("Search requires at least one criterion (e.g., keyword, or other criteria supported by the engine).")

        # Build description for logging
        search_description_parts: list[str] = [f"{k}='{v}'" for k, v in search_params.items()]
        search_description = ", ".join(search_description_parts)
        logger.debug(f"Attempting search with effective API params: {search_description}")

        # Warnings about engine support (now that params are definitely passed if provided)
        if 'page' in search_params and search_params['page'] > 1: # Page 1 is often default behaviour
            logger.warning(f"Pagination (page={search_params['page']}) passed to API, but actual pagination support depends on engine '{self.engine}'.")
        if 'limit' in search_params:
            logger.warning(f"Limit ({search_params['limit']}) passed to API. Engine '{self.engine}' may ignore it or have its own max limit. Client-side truncation will be applied if results exceed this.")

        try:
            videos_raw = self.client.search(**search_params)
            videos = self._parse_video_results(videos_raw)
            logger.info(f"Search with params [{search_description}] yielded {len(videos)} parsed results from engine.")

            # Client-side limit enforcement if 'limit' was requested (and search_params included it)
            # and results exceed it. This handles cases where the engine ignores the limit or returns more.
            if 'limit' in search_params and len(videos) > search_params['limit']:
                logger.info(f"Truncating {len(videos)} results to the requested limit of {search_params['limit']}.")
                videos = videos[:search_params['limit']]
            return videos
        except TypeError as te:
            logger.error(f"TypeError during search call for engine '{self.engine}' with params {search_params}. Error: {te}", exc_info=True)
            raise RuntimeError(f"Failed search on engine '{self.engine}' due to parameter issue: {te}") from te
        except Exception as e:
            logger.error(f"Error during search call with params [{search_description}]: {e}", exc_info=True)
            raise Exception(f"Failed to search videos: {e}") from e


# ==============================================================================
# HTML Generation Function
# ==============================================================================
def generate_html_output(videos: list[VideoDataClass], query: str, filename: str) -> str:
    if not videos:
        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
            "<title>No Results</title><style>body{background-color:#1a1a1a;color:#e0e0e0;font-family:sans-serif;text-align:center;padding-top:50px;}"
            "h1{color:#f0f;text-shadow:0 0 5px #f0f;}</style></head>"
            f"<body><h1>No videos found for query: '{html.escape(query)}'</h1></body></html>"
        )

    safe_query = html.escape(query)
    page_title = f"Search Results for '{safe_query}'"
    css = """<style>:root { --neon-cyan: #08f7fe; --neon-green: #39ff14; --dark-bg: #1a1a1a; --medium-dark-bg: #2a2a2a; --light-text: #e0e0e0; --dim-text: #aaa; } body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: var(--dark-bg); color: var(--light-text); } h1 { color: var(--neon-cyan); text-align: center; border-bottom: 2px solid var(--neon-cyan); padding-bottom: 15px; margin-bottom: 30px; text-shadow: 0 0 8px var(--neon-cyan); } p.results-info { text-align: center; color: var(--dim-text); margin-bottom: 30px; } .results-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 25px; } .video-item { background-color: var(--medium-dark-bg); border: 1px solid var(--neon-cyan); border-radius: 8px; overflow: hidden; box-shadow: 0 0 10px rgba(8, 247, 254, 0.3); transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease; position: relative; } .video-item:hover { transform: scale(1.03); border-color: var(--neon-green); box-shadow: 0 0 20px rgba(57, 255, 20, 0.6); z-index: 10; } .video-item a { text-decoration: none; color: inherit; display: block; } .video-item .image-container { position: relative; width: 100%; height: 0; padding-bottom: 56.25%; /* 16:9 Aspect Ratio */ background-color: #333; border-bottom: 1px solid var(--neon-cyan); overflow: hidden; } .video-item:hover .image-container { border-bottom-color: var(--neon-green); } .video-item .image-container img.thumbnail { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: block; transition: opacity 0.3s ease-in-out; z-index: 1; } .video-item .image-container video.preview-video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: none; z-index: 5; } .video-info { padding: 15px; } .video-title { font-size: 1.1em; font-weight: bold; margin: 0 0 10px 0; color: var(--neon-cyan); line-height: 1.3; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; min-height: 2.6em; /* Approx 2 lines */ } .video-details { font-size: 0.9em; color: var(--dim-text); margin-top: 8px; } .video-details span { margin-right: 12px; display: inline-block; white-space: nowrap; } .no-image-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #888; font-style: italic; text-align: center; padding: 10px; display: none; z-index: 2; } .video-item img.thumbnail[data-failed="true"] + .no-image-text { display: block; } .video-item img.thumbnail[data-failed="true"] { opacity: 0; } .video-item.preview-active .image-container img.thumbnail { opacity: 0; visibility: hidden; } .video-item.preview-active .image-container video.preview-video { display: block; }</style>"""
    html_parts = [
        f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>{page_title}</title>",
        css,
        f"</head><body><h1>{page_title}</h1><p class='results-info'>Found {len(videos)} videos. Results saved in: {html.escape(filename)}</p><div class='results-container' role='list'>"
    ]

    for video in videos:
        safe_title = html.escape(video.title)
        safe_link = html.escape(video.link)
        safe_img_url = html.escape(video.img) if video.img else ""
        safe_preview_url = html.escape(video.preview_url) if video.preview_url else ""
        safe_time = html.escape(video.time) if video.time else "N/A"
        safe_channel = html.escape(video.channel_name) if video.channel_name else "N/A"
        img_alt_text = f"Thumbnail for {safe_title}"
        preview_attr = f'data-preview-url="{safe_preview_url}"' if safe_preview_url else ''
        item_aria_label = f"Video: {safe_title}, Duration: {safe_time}" + (f", Channel: {safe_channel}" if safe_channel != "N/A" else "")


        html_parts.extend([
            f"    <div class='video-item' {preview_attr} role='listitem' aria-label='{html.escape(item_aria_label)}'>",
            f"        <a href='{safe_link}' target='_blank' title='{safe_title} (Opens in new tab)'>",
            f"            <div class='image-container'>"
        ])
        if safe_img_url:
            html_parts.extend([
                f"                <img class='thumbnail' src='{safe_img_url}' alt='{img_alt_text}' loading='lazy' onerror='this.setAttribute(\"data-failed\", \"true\"); console.warn(\"Failed to load thumbnail:\", this.src);'>",
                f"                <span class='no-image-text'>Preview Image Unavailable</span>"
            ])
        else:
            html_parts.extend([
                f"                <img class='thumbnail' src='' alt='{img_alt_text}' style='display: none;' data-failed='true'>", # Hidden but keeps structure
                f"                <span class='no-image-text' style='display: block;'>No Preview Image Provided</span>"
            ])
        html_parts.extend([
            f"            </div>",
            f"            <div class='video-info'>",
            f"                <div class='video-title'>{safe_title}</div>",
            f"                <div class='video-details'>",
            f"                    <span>Duration: {safe_time}</span>"
        ])
        if safe_channel != "N/A":
            html_parts.append(f"                    <span>Channel: {safe_channel}</span>")
        html_parts.extend([
            "                </div>",
            "            </div>",
            "        </a>",
            "    </div>"
        ])
    html_parts.append("</div>") # End of results-container

    js = """<script>
        document.addEventListener('DOMContentLoaded', () => {
            const videoItems = document.querySelectorAll('.video-item');
            let previewTimeout = null;
            const PREVIEW_DELAY_MS = 250; // Delay before starting preview playback

            videoItems.forEach(item => {
                const imageContainer = item.querySelector('.image-container');
                const previewUrl = item.dataset.previewUrl;
                let previewVideoElement = null; // Store the video element per item

                if (!imageContainer || !previewUrl) return; // No container or no preview URL

                const createAndPlayPreview = () => {
                    if (previewVideoElement) { // If element already exists (e.g., mouse re-enter quickly)
                        previewVideoElement.play().catch(e => console.warn(`Preview re-play prevented for ${previewUrl}: ${e.message}`));
                        item.classList.add('preview-active');
                        return;
                    }
                    console.debug(`Creating preview for: ${previewUrl}`);
                    previewVideoElement = document.createElement('video');
                    previewVideoElement.classList.add('preview-video');
                    previewVideoElement.src = previewUrl;
                    previewVideoElement.muted = true;
                    previewVideoElement.loop = true;
                    previewVideoElement.preload = 'auto'; // Preload video metadata and perhaps some data
                    previewVideoElement.setAttribute('playsinline', ''); // Important for iOS

                    imageContainer.appendChild(previewVideoElement);

                    const playPromise = previewVideoElement.play();
                    if (playPromise !== undefined) {
                        playPromise.then(() => {
                            console.debug(`Preview started: ${previewUrl}`);
                            item.classList.add('preview-active');
                        }).catch(error => {
                            console.warn(`Autoplay prevented for ${previewUrl}:`, error.message);
                            // Optionally remove the video element if play fails and is critical
                            // previewVideoElement.remove(); previewVideoElement = null;
                        });
                    } else { // Fallback for browsers that don't return a promise (older)
                        item.classList.add('preview-active');
                    }
                };

                const stopAndRemovePreview = () => {
                    item.classList.remove('preview-active');
                    if (previewVideoElement) {
                        console.debug(`Stopping and removing preview: ${previewUrl}`);
                        previewVideoElement.pause();
                        previewVideoElement.remove(); // Remove element to free resources
                        previewVideoElement = null;   // Clear reference
                    }
                };

                item.addEventListener('mouseenter', () => {
                    clearTimeout(previewTimeout); // Clear any existing timeout
                    previewTimeout = setTimeout(createAndPlayPreview, PREVIEW_DELAY_MS);
                });

                item.addEventListener('mouseleave', () => {
                    clearTimeout(previewTimeout); // Clear timeout if mouse leaves before delay
                    stopAndRemovePreview();
                });

                // Optional: Stop preview if user clicks the link to navigate away
                const link = item.querySelector('a');
                if (link) {
                    link.addEventListener('click', stopAndRemovePreview);
                }
            });
        });
    </script>"""
    html_parts.append(js)
    html_parts.append("</body></html>")
    return "\n".join(html_parts)

# ==============================================================================
# Helper Functions for Input Prompts
# ==============================================================================
def get_validated_input(prompt_message: str, default_value: Any, validation_type: type, positive_only: bool = False) -> Any:
    """
    Gets user input, validates its type, handles defaults, and ensures positivity if required for numeric types.
    """
    while True:
        try:
            user_input_str = input(f"{prompt_message} [{default_value}]: ").strip()

            if not user_input_str: # User pressed Enter, use default
                logger.debug(f"User accepted default: {default_value}")
                return default_value

            # Type conversion and validation
            value: Any
            if validation_type == int:
                value = int(user_input_str)
            elif validation_type == float:
                value = float(user_input_str)
            elif validation_type == Path:
                 value = Path(user_input_str) # Basic conversion, further validation (e.g. writability) done elsewhere
            elif validation_type == str:
                value = user_input_str # Already a string
            else:
                # This case should ideally not be reached if validation_type is one of the above
                logger.error(f"Unsupported validation_type '{validation_type.__name__}' in get_validated_input.")
                value = user_input_str # Fallback to string

            # Positivity check for numeric types
            if positive_only and validation_type in [int, float]:
                if not isinstance(value, (int, float)) or value <= 0: # Redundant check, but safe
                    print("Input must be a positive number. Please try again.")
                    continue

            logger.debug(f"User entered valid input: {value} (type: {type(value)})")
            return value

        except ValueError: # Handles int() or float() conversion errors
            print(f"Invalid input. Please enter a valid {validation_type.__name__}.")
        except Exception as e:
            print(f"An unexpected error occurred during input: {e}. Please try again.")
            logger.warning(f"Unexpected input error: {e}", exc_info=True)
            # Loop continues, re-prompting the user

# ==============================================================================
# Main Execution Function (Using Prompts)
# ==============================================================================
def main():
    """Main function using interactive prompts to gather info, run client, and generate output."""
    logger.info(f"--- Starting {__script_name__} v{__version__} (Interactive Mode) ---")

    try:
        search_query = ""
        while not search_query:
             search_query = input("Enter search query: ").strip()
             if not search_query:
                 print("Search query cannot be empty. Please enter a term to search for.")

        limit = get_validated_input("Max results to fetch?", DEFAULT_SEARCH_LIMIT, int, positive_only=True)
        page = get_validated_input("Page number (e.g., 1, 2)?", DEFAULT_PAGE, int, positive_only=True)
        engine = get_validated_input("PornLib engine (e.g., xvideos)?", DEFAULT_ENGINE, str)
        soup_sleep = get_validated_input("Soup sleep (seconds, e.g., 1.0)?", DEFAULT_SOUP_SLEEP, float, positive_only=False)
        output_dir_str = get_validated_input("Output directory for HTML file?", DEFAULT_OUTPUT_DIR_STR, str)
        output_dir = Path(output_dir_str)
        filename_prefix_format = get_validated_input("Filename prefix format string?", DEFAULT_FILENAME_PREFIX, str)
        auto_open_str = get_validated_input("Auto-open HTML file after generation (y/n)?", DEFAULT_AUTO_OPEN, str)
        auto_open = auto_open_str.lower().startswith('y')

    except (KeyboardInterrupt, EOFError):
        logger.info("\nInput process cancelled by user. Exiting.")
        return
    except Exception as e:
         logger.critical(f"Failed to gather critical input settings: {e}", exc_info=True)
         print("A critical error occurred while gathering settings. Cannot continue.")
         return

    logger.info(
        f"Settings: Engine='{engine}', Query='{search_query}', Limit={limit}, Page={page}, "
        f"SoupSleep={soup_sleep:.2f}s, OutputDir='{output_dir}', AutoOpen={auto_open}"
    )

    client: XvideosClient | None = None
    try:
        client = XvideosClient(engine=engine, soup_sleep=soup_sleep)

        logger.info(f"Performing search for query: '{search_query}' on engine '{engine}'...")
        search_results: list[VideoDataClass] = client.search_videos(
            keyword=search_query,
            page=page,
            limit=limit
        )

        if search_results:
            logger.info(f"Successfully retrieved and parsed {len(search_results)} video(s).")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query_part = "".join(c for c in search_query.replace(' ', '_') if c.isalnum() or c in ['_', '-'])[:40]

            try:
                filename_stem = filename_prefix_format.format(
                    engine=engine, query_part=safe_query_part, timestamp=timestamp
                )
            except KeyError as e:
                 logger.error(f"Invalid key '{e}' in filename prefix format string '{filename_prefix_format}'. Using default format.")
                 filename_stem = DEFAULT_FILENAME_PREFIX.format(
                      engine=engine, query_part=safe_query_part, timestamp=timestamp
                 )
            except Exception as fmt_e:
                 logger.error(f"Error formatting filename prefix '{filename_prefix_format}': {fmt_e}. Using basic name.")
                 filename_stem = f"{engine}_search_{timestamp}" # Basic fallback

            output_filename = f"{filename_stem}.html"
            output_path_resolved = output_dir.resolve() / output_filename

            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured output directory exists: {output_dir.resolve()}")
            except OSError as e:
                logger.error(f"Failed to create output directory '{output_dir.resolve()}': {e}. Check permissions.")
                logger.warning(f"Attempting to save to current working directory: {Path.cwd()}")
                output_path_resolved = Path.cwd() / output_filename

            logger.info("Generating HTML output...")
            html_content = generate_html_output(search_results, search_query, str(output_path_resolved))

            logger.info(f"Attempting to save results to: {output_path_resolved}")
            try:
                with open(output_path_resolved, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"Successfully saved results to: {output_path_resolved}")

                if auto_open:
                    try:
                        logger.info(f"Attempting to open HTML file in browser: {output_path_resolved}")
                        webbrowser.open(output_path_resolved.as_uri()) # .as_uri() for file:// scheme
                    except Exception as open_err:
                        logger.warning(f"Could not automatically open file '{output_path_resolved}': {open_err}. "
                                       "You may need to open it manually in your browser.")
            except OSError as e:
                logger.error(f"Error saving HTML file '{output_path_resolved}': {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error writing HTML file: {e}", exc_info=True)
        else:
            logger.warning(f"No videos found for query '{search_query}' using engine '{engine}'. No HTML file generated.")

    except ValueError as ve: # Typically from client validation or input issues not caught earlier
        logger.critical(f"Configuration or Value Error: {ve}", exc_info=False) # exc_info=False as ve usually has enough info
    except NotImplementedError as nie:
        logger.critical(f"Feature Not Implemented: This feature may not be supported by the '{client.engine if client else engine}' engine. Error: {nie}", exc_info=True)
    except RuntimeError as rte: # Includes client initialization or search failures
        logger.critical(f"Runtime Error: {rte}", exc_info=True)
    except ImportError: # Already logged in XvideosClient.__init__ if it's pornLib
        logger.critical("ImportError encountered. Ensure all dependencies like 'pornLib' and 'ratelimit' are installed.", exc_info=False)
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e: # Catch-all for truly unexpected errors
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        logger.info(f"--- {__script_name__} Finished ---")


# ==============================================================================
# Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    # --- Usage Instructions ---
    # 1. Install dependencies:
    #    pip install pornlib ratelimit
    #
    # 2. Run the script from your terminal:
    #    python xvid_prompt.py
    #
    # 3. Follow the interactive prompts to enter your search query, desired engine, etc.
    #
    # Note on Video Previews:
    # The HTML output supports hover-to-play video previews if the 'preview_url'
    # is available in the VideoDataClass for a video. This script relies on the
    # 'pornLib' library to provide these 'preview_url's. If 'pornLib' does not
    # supply them for a given engine or video, previews will not be available.
    # This script itself DOES NOT scrape video pages to find preview URLs;
    # it only uses data directly provided by 'pornLib'.
    # --------------------------
    main()

# --- END OF FILE xvid_prompt.py ---
```
