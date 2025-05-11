# --- START OF FILE xvid3.py ---

import datetime
import html
import logging
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Attempt to import pornLib, critical for operation
try:
    import pornLib
except ImportError:
    # Basic logging setup for pre-main execution issues.
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).critical(
        "Fatal Error: pornLib library not found. Please install it (e.g., 'pip install pornlib'). "
        "Script cannot continue."
    )
    sys.exit(1) # Exit if core dependency is missing

# Attempt to import ratelimit
try:
    from ratelimit import limits, sleep_and_retry # type: ignore
except ImportError:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).critical(
        "Fatal Error: ratelimit library not found. Please install it (e.g., 'pip install ratelimit'). "
        "Script cannot continue."
    )
    sys.exit(1)


# ==============================================================================
# Script Information
# ==============================================================================
__version__ = "1.1.1" # Incremented version for changes
__script_name__ = "PornLib Search Script"
# Module docstring
"""
PornLib Search Script - xvid3.py

This script provides a command-line interface to search for videos using the
pornLib library, targeting various adult video platforms. It allows users to
specify search queries, engines, result limits, and other parameters.
The search results are then compiled into an HTML file with interactive
video previews (if available from pornLib).

Key features:
- Interactive prompts for search parameters.
- Support for multiple pornLib engines.
- Rate limiting for API calls to avoid being blocked.
- Generation of a browseable HTML report of search results.
- Hover-to-play video previews in the HTML report.
- Robust error handling and logging.
"""

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
DEFAULT_OUTPUT_DIR_STR = "."
DEFAULT_SEARCH_LIMIT = 30
DEFAULT_PAGE = 1
DEFAULT_ENGINE = "xvideos" # Default engine, can be changed by user
DEFAULT_SOUP_SLEEP = 1.0 # Default soup sleep for pornLib
DEFAULT_FILENAME_PREFIX = "{engine}_search_{query_part}_{timestamp}"
DEFAULT_AUTO_OPEN = 'y'


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class VideoDataClass:
    """Represents metadata for a single video."""
    title: str
    img: str # URL to the thumbnail image
    link: str # URL to the video page
    preview_url: str | None = None # URL to a short preview video/gif
    quality: str | int | None = None # Video quality (e.g., "720p", 1080)
    time: str | None = None # Duration of the video (e.g., "10:30")
    channel_name: str | None = None
    channel_link: str | None = None


@dataclass
class VideoDownloadDataClass: # Potentially for future use or internal pornLib structures
    """Represents video download links at different qualities."""
    low: str | None = None
    high: str | None = None
    hls: str | None = None


@dataclass
class Tags: # Potentially for future use or internal pornLib structures
    """Represents a tag associated with a video or search."""
    name: str | None = None
    id: str | None = None


# ==============================================================================
# Xvideos Client Class
# ==============================================================================
class XvideosClient:
    """
    A client to interact with video platforms via the pornLib library.
    Handles API calls, result parsing, and rate limiting.
    """
    def __init__(self, engine: str = DEFAULT_ENGINE, soup_sleep: float = DEFAULT_SOUP_SLEEP):
        """
        Initializes the PornLib client.

        Args:
            engine: The pornLib engine to use (e.g., 'xvideos').
            soup_sleep: Sleep duration for web scraping politeness, passed to pornLib.

        Raises:
            ImportError: If pornLib's 'PornLib' class is missing (indicates corrupted/outdated library).
            RuntimeError: If client initialization fails for other reasons.
        """
        self.engine = engine
        try:
            # Check if pornLib.PornLib class exists. The module import itself is checked at script start.
            if not hasattr(pornLib, 'PornLib'):
                # This error indicates a problem with the pornLib installation (e.g. corrupted, outdated)
                raise AttributeError("pornLib module is imported, but the 'PornLib' class is missing or not found.")

            self.client = pornLib.PornLib(engine=engine, soupSleep=soup_sleep)
            logger.info(f"Successfully initialized PornLib Client: engine='{engine}', soup_sleep={soup_sleep:.2f}s")
        except AttributeError as ae: # Catches if pornLib.PornLib is missing
            logger.critical(f"Fatal Error: pornLib library is incomplete or outdated. {ae}")
            # Raising ImportError here as it's fundamentally about a missing part of the library.
            raise ImportError(f"pornLib.PornLib class not found: {ae}") from ae
        except Exception as e: # Catches other errors from pornLib.PornLib() constructor
            logger.critical(f"Fatal Error: Failed to initialize pornLib client for engine '{engine}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"XvideosClient initialization failed for engine '{engine}': {e}") from e

    def _parse_video_results(self, results_raw: Any) -> list[VideoDataClass]:
        """Parses raw video results from pornLib into a list of VideoDataClass objects."""
        if results_raw is None:
            logger.debug("Received None for video results, returning empty list.")
            return []
        if not isinstance(results_raw, list):
            logger.warning(f"Expected list for video results, got {type(results_raw)}. Returning empty list.")
            return []

        videos: list[VideoDataClass] = []
        required_keys_for_dict = ['title', 'img', 'link']

        for i, item in enumerate(results_raw):
            video: VideoDataClass | None = None
            try:
                if isinstance(item, dict):
                    if all(k in item and item[k] is not None for k in required_keys_for_dict):
                        raw_preview_url = item.get('preview_url')
                        raw_time = item.get('time')
                        raw_channel_name = item.get('channel_name')
                        raw_channel_link = item.get('channel_link')
                        video = VideoDataClass(
                            title=str(item.get('title', '')),
                            img=str(item.get('img', '')),
                            link=str(item.get('link', '')),
                            preview_url=str(raw_preview_url) if raw_preview_url is not None else None,
                            quality=item.get('quality'),
                            time=str(raw_time) if raw_time is not None else None,
                            channel_name=str(raw_channel_name) if raw_channel_name is not None else None,
                            channel_link=str(raw_channel_link) if raw_channel_link is not None else None,
                        )
                    else:
                        missing_or_none_keys = [k for k in required_keys_for_dict if k not in item or item[k] is None]
                        logger.warning(f"Skipping video dict item #{i + 1} due to missing/None essential keys: {missing_or_none_keys}. Data: {item}")
                elif isinstance(item, VideoDataClass):
                    if item.title and item.img and item.link: # Ensure essential string fields are not empty/None
                        video = item
                    else:
                        logger.warning(
                            f"Skipping VideoDataClass item #{i + 1} because essential attributes "
                            f"(title, img, or link) are empty or None. Data: {item}"
                        )
                elif hasattr(item, 'title') and hasattr(item, 'img') and hasattr(item, 'link'):
                    if all(getattr(item, k, None) is not None for k in required_keys_for_dict):
                        raw_preview_url = getattr(item, 'preview_url', None)
                        raw_time = getattr(item, 'time', None)
                        raw_channel_name = getattr(item, 'channel_name', None)
                        raw_channel_link = getattr(item, 'channel_link', None)
                        video = VideoDataClass(
                            title=str(getattr(item, 'title', '')),
                            img=str(getattr(item, 'img', '')),
                            link=str(getattr(item, 'link', '')),
                            preview_url=str(raw_preview_url) if raw_preview_url is not None else None,
                            quality=getattr(item, 'quality', None),
                            time=str(raw_time) if raw_time is not None else None,
                            channel_name=str(raw_channel_name) if raw_channel_name is not None else None,
                            channel_link=str(raw_channel_link) if raw_channel_link is not None else None,
                        )
                    else:
                        logger.warning(f"Skipping object item #{i + 1} because essential attributes (title, img, or link) are None. Data: {item!r}")
                else:
                    logger.warning(f"Skipping unrecognized video item #{i + 1} of type: {type(item)}. Item data: {item!r}")

                if video:
                    videos.append(video)
            except Exception as e:
                logger.error(f"Error parsing video item #{i + 1}: {item!r}. Error: {e}", exc_info=False)
        return videos

    def _parse_tag_results(self, tags_raw: Any) -> list[Tags]:
        """Parses raw tag results from pornLib into a list of Tags objects."""
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
                    raw_name = item.get('name')
                    raw_id = item.get('id')
                    tag = Tags(
                        name=str(raw_name) if raw_name is not None else None,
                        id=str(raw_id) if raw_id is not None else None
                    )
                elif isinstance(item, Tags):
                    tag = item
                elif hasattr(item, 'name') or hasattr(item, 'id'):
                    raw_name = getattr(item, 'name', None)
                    raw_id = getattr(item, 'id', None)
                    tag = Tags(
                        name=str(raw_name) if raw_name is not None else None,
                        id=str(raw_id) if raw_id is not None else None
                    )
                else:
                    logger.warning(f"Skipping unrecognized tag item #{i + 1} of type: {type(item)}. Item data: {item!r}")
                    continue

                if tag and (tag.name or tag.id):
                    tags_list.append(tag)
                elif tag:
                    logger.debug(f"Skipping parsed tag item #{i + 1} with no name or id: {item!r}")

            except Exception as e:
                logger.error(f"Error parsing tag item #{i + 1}: {item!r}. Error: {e}", exc_info=False)
        return tags_list

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def list_videos(self, limit: int = 12) -> list[VideoDataClass]:
        """
        Fetches a list of videos, often 'new' or 'popular', depending on the engine.

        Args:
            limit: The maximum number of videos to return.

        Returns:
            A list of VideoDataClass objects.

        Raises:
            ValueError: If limit is not a positive integer.
            RuntimeError: If listing videos fails due to a TypeError (e.g. unsupported parameter).
            Exception: If the API call fails or parsing encounters critical issues.
        """
        if not isinstance(limit, int) or limit <= 0:
            logger.error(f"Invalid limit value: {limit}. Must be a positive integer.")
            raise ValueError("Limit must be a positive integer.")
        logger.debug(f"Attempting to fetch up to {limit} videos using list method for engine '{self.engine}'...")
        try:
            videos_raw = self.client.list(limit=limit)
            videos = self._parse_video_results(videos_raw)
        except TypeError as te:
            if 'limit' in str(te).lower(): # Pragmatic check if pornLib signals 'limit' is an issue
                logger.warning(f"Engine '{self.engine}' list method may not support 'limit' parameter. Trying without it.")
                videos_raw_retry = self.client.list() # Call without limit
                parsed_videos_retry = self._parse_video_results(videos_raw_retry)
                # Apply limit client-side
                videos = parsed_videos_retry[:limit]
                logger.info(f"Fetched {len(parsed_videos_retry)} videos (without limit param), returning first {len(videos)} for engine '{self.engine}'.")
            else:
                logger.error(f"TypeError during list call for engine '{self.engine}': {te}", exc_info=True)
                raise RuntimeError(f"Failed to list videos for engine '{self.engine}' due to TypeError: {te}") from te
        except Exception as e:
            logger.error(f"Error during list call for engine '{self.engine}': {e}", exc_info=True)
            raise Exception(f"Failed to list videos for engine '{self.engine}': {e}") from e

        logger.info(f"Fetched and parsed {len(videos)} videos for engine '{self.engine}' (requested list limit: {limit}).")
        return videos

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def search_videos(self, keyword: str | None = None, page: int | None = None, limit: int | None = None, **kwargs: Any) -> list[VideoDataClass]:
        """
        Searches for videos based on a keyword and other criteria.

        Args:
            keyword: The search term.
            page: The page number for paginated results.
            limit: The maximum number of results per page/request.
            **kwargs: Additional search parameters specific to the engine.

        Returns:
            A list of VideoDataClass objects matching the search criteria.

        Raises:
            ValueError: If no search criteria are provided.
            RuntimeError: If the search call fails due to parameter issues (TypeError).
            Exception: If the API call fails or parsing encounters critical issues.
        """
        search_params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
        if keyword:
            search_params['keyword'] = keyword
        if page is not None and page > 0:
            search_params['page'] = page
        if limit is not None and limit > 0:
            search_params['limit'] = limit

        if not search_params: # Ensure at least one search criterion is present
            logger.error("Search initiated without any criteria (keyword, page, limit, or other kwargs).")
            raise ValueError("Search requires at least one criterion (e.g., keyword).")

        search_description_parts: list[str] = [f"{k}='{v}'" for k, v in search_params.items()]
        search_description = ", ".join(search_description_parts)
        logger.debug(f"Attempting search on engine '{self.engine}' with effective API params: {search_description}")

        if 'page' in search_params and search_params['page'] > 1:
            logger.warning(f"Pagination (page={search_params['page']}) passed to API. Actual support depends on engine '{self.engine}'.")
        if 'limit' in search_params:
            logger.warning(f"Limit ({search_params['limit']}) passed to API. Engine '{self.engine}' may ignore or have its own max. Client-side truncation applied if needed.")

        try:
            videos_raw = self.client.search(**search_params)
            videos = self._parse_video_results(videos_raw)
            logger.info(f"Search on engine '{self.engine}' with params [{search_description}] yielded {len(videos)} parsed results.")

            if 'limit' in search_params and len(videos) > search_params['limit']:
                logger.info(f"Truncating {len(videos)} results to the requested limit of {search_params['limit']}.")
                videos = videos[:search_params['limit']]
            return videos
        except TypeError as te:
            logger.error(f"TypeError during search call for engine '{self.engine}' with params {search_params}. Error: {te}", exc_info=True)
            raise RuntimeError(f"Failed search on engine '{self.engine}' due to parameter issue: {te}") from te
        except Exception as e:
            logger.error(f"Error during search call with params [{search_description}] for engine '{self.engine}': {e}", exc_info=True)
            raise Exception(f"Failed to search videos on engine '{self.engine}': {e}") from e


# ==============================================================================
# HTML Generation Function
# ==============================================================================
def generate_html_output(videos: list[VideoDataClass], query: str, filename: str) -> str:
    """Generates an HTML string to display video search results."""
    if not videos:
        # Basic HTML for no results
        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
            "<title>No Results</title><style>body{background-color:#1a1a1a;color:#e0e0e0;font-family:sans-serif;text-align:center;padding-top:50px;}"
            "h1{color:#f0f;text-shadow:0 0 5px #f0f;}</style></head>"
            f"<body><h1>No videos found for query: '{html.escape(query)}'</h1></body></html>"
        )

    safe_query = html.escape(query)
    page_title = f"Search Results for '{safe_query}'"
    # CSS is embedded for single-file output convenience.
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
                f"                <span class='no-image-text'>Preview Image Unavailable</span>" # Shown via CSS if image fails
            ])
        else:
            html_parts.extend([
                f"                <img class='thumbnail' src='' alt='{img_alt_text}' style='display: none;' data-failed='true'>", # Hidden, keeps structure, marks as failed
                f"                <span class='no-image-text' style='display: block;'>No Preview Image Provided</span>" # Directly visible
            ])
        html_parts.extend([
            f"            </div>", # End image-container
            f"            <div class='video-info'>",
            f"                <div class='video-title'>{safe_title}</div>",
            f"                <div class='video-details'>",
            f"                    <span>Duration: {safe_time}</span>"
        ])
        if safe_channel != "N/A":
            html_parts.append(f"                    <span>Channel: {safe_channel}</span>")
        html_parts.extend([
            "                </div>", # End video-details
            "            </div>", # End video-info
            "        </a>",
            "    </div>" # End video-item
        ])
    html_parts.append("</div>") # End results-container

    # JavaScript for previews is embedded for single-file output.
    js = """<script>
        document.addEventListener('DOMContentLoaded', () => {
            const videoItems = document.querySelectorAll('.video-item');
            let previewTimeout = null;
            const PREVIEW_DELAY_MS = 250; // Delay before starting preview playback

            videoItems.forEach(item => {
                const imageContainer = item.querySelector('.image-container');
                const previewUrl = item.dataset.previewUrl;
                let previewVideoElement = null; // Store the video element per item

                if (!imageContainer || !previewUrl) return; // Skip if no container or no preview URL

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
                            // Optionally remove video element if play fails and is critical
                            // if (previewVideoElement) { previewVideoElement.remove(); previewVideoElement = null; }
                        });
                    } else { // Fallback for older browsers that don't return a promise
                        item.classList.add('preview-active'); // Assume play if no promise
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
    For float types, `positive_only=False` allows zero and negative numbers unless further restricted.
    """
    while True:
        try:
            user_input_str = input(f"{prompt_message} [{default_value}]: ").strip()

            if not user_input_str: # User pressed Enter, use default
                logger.debug(f"User accepted default: {default_value}")
                return default_value

            value: Any
            if validation_type == int:
                value = int(user_input_str)
            elif validation_type == float:
                value = float(user_input_str)
            elif validation_type == Path:
                value = Path(user_input_str) # Basic conversion
            elif validation_type == str:
                value = user_input_str # Already a string
            else:
                # This case should ideally not be reached if validation_type is one of the above
                logger.error(f"Unsupported validation_type '{validation_type.__name__}' in get_validated_input. Falling back to string.")
                value = user_input_str

            # Positivity check for numeric types (int, float)
            if positive_only and validation_type in [int, float]:
                if not isinstance(value, (int, float)) or value <= 0: # Redundant type check, but safe
                    print("Input must be a positive number. Please try again.")
                    continue
            
            # For float, if not positive_only, it can be < 0. Specific checks (e.g. soup_sleep >= 0) are outside.

            logger.debug(f"User entered valid input: {value} (type: {type(value)})")
            return value

        except ValueError: # Handles int() or float() conversion errors
            print(f"Invalid input. Please enter a valid {validation_type.__name__}.")
        except Exception as e: # Catch any other unexpected error during input processing
            print(f"An unexpected error occurred during input: {e}. Please try again.")
            logger.warning(f"Unexpected input error: {e}", exc_info=True)

# ==============================================================================
# Main Execution Function (Using Prompts)
# ==============================================================================
def main():
    """Main function: gathers user input, performs search, and generates HTML report."""
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
        
        # For soup_sleep, allow 0.0 but not negative. positive_only=False allows 0.
        soup_sleep_input = get_validated_input(
            "Soup sleep (seconds, e.g., 1.0, min 0.0)?", 
            DEFAULT_SOUP_SLEEP, 
            float, 
            positive_only=False # Allows 0 and negative; we'll clamp negative to 0.
        )
        if soup_sleep_input < 0:
            logger.warning(f"Soup sleep cannot be negative ({soup_sleep_input}). Clamping to 0.0s.")
            soup_sleep = 0.0
        else:
            soup_sleep = soup_sleep_input

        output_dir_str = get_validated_input("Output directory for HTML file?", DEFAULT_OUTPUT_DIR_STR, str)
        output_dir = Path(output_dir_str)
        filename_prefix_format = get_validated_input("Filename prefix format string?", DEFAULT_FILENAME_PREFIX, str)
        auto_open_str = get_validated_input("Auto-open HTML file after generation (y/n)?", DEFAULT_AUTO_OPEN, str)
        auto_open = auto_open_str.lower().startswith('y')

    except (KeyboardInterrupt, EOFError):
        logger.info("\nInput process cancelled by user. Exiting.")
        sys.exit(0) # Clean exit for user cancellation
    except Exception as e: # Catch any other error during input gathering
        logger.critical(f"Failed to gather critical input settings: {e}", exc_info=True)
        print("A critical error occurred while gathering settings. Cannot continue.")
        sys.exit(1) # Exit due to error

    logger.info(
        f"Settings: Engine='{engine}', Query='{search_query}', Limit={limit}, Page={page}, "
        f"SoupSleep={soup_sleep:.2f}s, OutputDir='{output_dir}', AutoOpen={auto_open}"
    )

    client: XvideosClient | None = None # Initialize for broader scope (error messages, finally block)
    try:
        client = XvideosClient(engine=engine, soup_sleep=soup_sleep)

        logger.info(f"Performing search for query: '{search_query}' on engine '{engine}'...")
        search_results = client.search_videos(
            keyword=search_query,
            page=page,
            limit=limit
        )

        if search_results:
            logger.info(f"Successfully retrieved and parsed {len(search_results)} video(s).")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize query part for filename: replace spaces, keep alphanumeric, underscore, hyphen
            safe_query_part = "".join(c for c in search_query.replace(' ', '_') if c.isalnum() or c in ['_', '-'])[:40]

            try:
                filename_stem = filename_prefix_format.format(
                    engine=engine, query_part=safe_query_part, timestamp=timestamp
                )
            except KeyError as e_fmt_key:
                logger.error(f"Invalid key '{e_fmt_key}' in filename prefix format string '{filename_prefix_format}'. Using default format.")
                filename_stem = DEFAULT_FILENAME_PREFIX.format( # Fallback to default format string
                    engine=engine, query_part=safe_query_part, timestamp=timestamp
                )
            except Exception as e_fmt: # Catch other formatting errors
                logger.error(f"Error formatting filename prefix '{filename_prefix_format}': {e_fmt}. Using basic name.")
                # Basic fallback filename stem, ensuring all parts are likely safe
                safe_engine = "".join(c for c in engine if c.isalnum() or c == '_')
                filename_stem = f"{safe_engine}_search_{safe_query_part}_{timestamp}"


            output_filename = f"{filename_stem}.html"
            
            # Resolve path and attempt to create directory
            # Using resolve() early to get absolute path for logging and operations
            resolved_output_dir = output_dir.resolve()
            output_path = resolved_output_dir / output_filename

            try:
                resolved_output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured output directory exists: {resolved_output_dir}")
            except OSError as e_dir:
                logger.error(f"Failed to create output directory '{resolved_output_dir}': {e_dir}. Check permissions.")
                logger.warning(f"Attempting to save to current working directory: {Path.cwd()}")
                output_path = Path.cwd() / output_filename # Fallback path

            logger.info("Generating HTML output...")
            html_content = generate_html_output(search_results, search_query, str(output_path))

            logger.info(f"Attempting to save results to: {output_path}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"Successfully saved results to: {output_path}")

                if auto_open:
                    try:
                        logger.info(f"Attempting to open HTML file in browser: {output_path}")
                        webbrowser.open(output_path.as_uri()) # .as_uri() for file:// scheme
                    except Exception as e_open:
                        logger.warning(f"Could not automatically open file '{output_path}': {e_open}. "
                                       "Please open it manually in your browser.")
            except OSError as e_write:
                logger.error(f"Error saving HTML file '{output_path}': {e_write}", exc_info=True)
            except Exception as e_html: # Other unexpected errors during HTML write/open
                logger.error(f"Unexpected error processing HTML file: {e_html}", exc_info=True)
        else:
            logger.warning(f"No videos found for query '{search_query}' using engine '{engine}'. No HTML file generated.")

    except ValueError as ve: # Typically from client validation or input issues not caught earlier
        logger.critical(f"Configuration or Value Error: {ve}", exc_info=False) # exc_info=False as ve usually has enough info
    except NotImplementedError as nie: # Should ideally be caught if specific engine features are known missing
        engine_name = client.engine if client else engine # Use client's engine if available, else user input
        logger.critical(f"Feature Not Implemented: This may not be supported by the '{engine_name}' engine. Error: {nie}", exc_info=True)
    except RuntimeError as rte: # Includes client initialization or search failures from client methods
        logger.critical(f"Runtime Error: {rte}", exc_info=True)
    except ImportError: # Should be caught by XvideosClient.__init__ or top-level imports
        # This is a fallback; initial imports are checked earlier.
        logger.critical("ImportError encountered. This should have been caught earlier. Ensure all dependencies are installed.", exc_info=False)
    except KeyboardInterrupt: # If Ctrl+C during client operations or other parts of main
        logger.info("\nProcess interrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e: # Catch-all for truly unexpected errors in the main block
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally:
        logger.info(f"--- {__script_name__} Finished ---")


# ==============================================================================
# Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    # --- Usage Instructions (for reference when running the script) ---
    # 1. Ensure Python is installed on your system.
    # 2. Install required dependencies:
    #    pip install pornlib ratelimit
    #    (or pip install -r requirements.txt if a file is provided with these)
    #
    # 3. Run the script from your terminal:
    #    python xvid3.py
    #
    # 4. Follow the interactive prompts to:
    #    - Enter your search query.
    #    - Specify the maximum number of results (limit).
    #    - Set the page number for results.
    #    - Choose the pornLib engine (e.g., xvideos, pornhub).
    #    - Configure soup_sleep (delay for web scraping, affects pornLib).



    #    - Define the output directory for the HTML results file.
    #    - Set the filename prefix format.
    #    - Decide whether to auto-open the HTML file.
    #
    # Note on Video Previews:
    # The generated HTML report supports hover-to-play video previews.
    # This feature relies on 'preview_url' being provided by the 'pornLib'
    # library for each video. If 'pornLib' does not supply these URLs for a
    # particular engine or video, previews will not be available in the report.
    # This script uses data directly from 'pornLib' and does not perform
    # additional scraping to find preview URLs.
    # --------------------------------------------------------------------
    main()

# --- END OF FILE xvid3.py ---


Max results to fetch? [30]:
Page number (e.g., 1, 2)? [1]:
PornLib engine (e.g., xvideos)? [xvideos]:
Soup sleep (seconds, e.g., 1.0, min 0.0)? [1.0]:
Output directory for HTML file? [.]:
Filename prefix format string? [{engine}_search_{query_part}_{timestamp}]:
Auto-open HTML file after generation (y/n)? [y]:
2025-05-10 14:11:56,685 - PornLib Search Script - INFO - Settings: Engine='xvideos', Query='suck off hj', Limit=30, Page=1, SoupSleep=1.00s, OutputDir='.', AutoOpen=True
2025-05-10 14:11:56,685 - PornLib Search Script - INFO - Successfully initialized PornLib Client: engine='xvideos', soup_sleep=1.00s
2025-05-10 14:11:56,686 - PornLib Search Script - INFO - Performing search for query: 'suck off hj' on engine 'xvideos'...
2025-05-10 14:11:56,686 - PornLib Search Script - WARNING - Limit (30) passed to API. Engine 'xvideos' may ignore or have its own max. Client-side truncation applied if needed.
2025-05-10 14:11:56,686 - PornLib Search Script - ERROR - TypeError during search call for engine 'xvideos' with params {'keyword': 'suck off hj', 'page': 1, 'limit': 30}. Error: PornLib.search() got an unexpected keyword argument 'page'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/bing/xvid4.py", line 350, in search_videos
    videos_raw = self.client.search(**search_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: PornLib.search() got an unexpected keyword argument 'page'
2025-05-10 14:11:56,690 - PornLib Search Script - CRITICAL - Runtime Error: Failed search on engine 'xvideos' due to parameter issue: PornLib.search() got an unexpected keyword argument 'page'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/bing/xvid4.py", line 350, in search_videos
    videos_raw = self.client.search(**search_params)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: PornLib.search() got an unexpected keyword argument 'page'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/bing/xvid4.py", line 613, in main
    search_results = client.search_videos(
                     ^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ratelimit/decorators.py", line 113, in wrapper
    return func(*args, **kargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ratelimit/decorators.py", line 80, in wrapper
    return func(*args, **kargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/home/worldguide/bing/xvid4.py", line 360, in search_videos
    raise RuntimeError(f"Failed search on engine '{self.engine}' due to parameter issue: {te}") from te
RuntimeError: Failed search on engine 'xvideos' due to parameter issue: PornLib.search() got an unexpected keyword argument 'page'
2025-05-10 14:11:56,695 - PornLib Search Script - INFO - --- PornLib Search
