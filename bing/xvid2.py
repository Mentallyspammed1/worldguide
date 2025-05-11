# --- START OF FILE xvid_prompt.py ---

import datetime
import html
import logging
import os
import sys
from dataclasses import dataclass

# No longer need argparse
from pathlib import Path  # For better path handling
from typing import Any

import pornLib  # Assuming this library exists and is installed
from ratelimit import limits, sleep_and_retry  # type: ignore

# ==============================================================================
# Configuration
# ==============================================================================

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO  # Set default log level directly
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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
# (DataClasses remain the same: VideoDataClass, VideoDownloadDataClass, Tags)
@dataclass
class VideoDataClass:
    title: str; img: str; link: str
    preview_url: str | None = None; quality: str | int | None = None
    time: str | None = None; channel_name: str | None = None; channel_link: str | None = None


@dataclass
class VideoDownloadDataClass:
    low: str | None = None; high: str | None = None; hls: str | None = None


@dataclass
class Tags:
    name: str | None = None; id: str | None = None


# ==============================================================================
# Xvideos Client Class
# ==============================================================================
# (XvideosClient class remains the same, including its methods:
#  __init__, _parse_video_results, _parse_tag_results, list_videos, search_videos)
class XvideosClient:
    def __init__(self, engine: str = DEFAULT_ENGINE, soup_sleep: float = DEFAULT_SOUP_SLEEP):
        self.engine = engine
        try:
            if 'pornLib' not in sys.modules and 'pornlib' not in sys.modules:
                 raise ImportError("pornLib module does not appear to be correctly imported or installed.")
            self.client = pornLib.PornLib(engine=engine, soupSleep=soup_sleep)
            logger.info(f"Successfully initialized PornLib Client: engine='{engine}', soup_sleep={soup_sleep:.2f}s")
        except ImportError:
            logger.critical("Fatal Error: pornLib library not found. Please install it (e.g., 'pip install pornlib').")
            raise
        except Exception as e:
            logger.critical(f"Fatal Error: Failed to initialize pornLib client for engine '{engine}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"XvideosClient initialization failed for engine '{engine}': {e}") from e

    def _parse_video_results(self, results_raw: Any) -> list[VideoDataClass]:
        if results_raw is None: logger.debug("Received None for video results, returning empty list."); return []
        if not isinstance(results_raw, list): logger.warning(f"Expected list for video results, got {type(results_raw)}. Returning empty."); return []
        videos: list[VideoDataClass] = []
        required_keys = ['title', 'img', 'link']
        for i, item in enumerate(results_raw):
            video = None
            try:
                if isinstance(item, dict):
                    if all(k in item and item[k] is not None for k in required_keys):
                         video = VideoDataClass(
                             title=str(item.get('title', '')), img=str(item.get('img', '')), link=str(item.get('link', '')),
                             preview_url=str(item['preview_url']) if item.get('preview_url') else None, quality=item.get('quality'),
                             time=str(item['time']) if item.get('time') else None, channel_name=str(item['channel_name']) if item.get('channel_name') else None,
                             channel_link=str(item['channel_link']) if item.get('channel_link') else None,
                         )
                    else: logger.warning(f"Skipping video dict item #{i + 1} missing keys: {[k for k in required_keys if k not in item or item[k] is None]}. Data: {item}")
                elif isinstance(item, VideoDataClass):
                     if all([item.title, item.img, item.link]):
                          if not hasattr(item, 'preview_url'): item.preview_url = None; video = item
                     else: logger.warning(f"Skipping VideoDataClass item #{i + 1} missing essential attributes.")
                elif hasattr(item, 'title') and hasattr(item, 'img') and hasattr(item, 'link'):
                     if all(getattr(item, k, None) is not None for k in required_keys):
                         video = VideoDataClass(
                             title=str(getattr(item, 'title', '')), img=str(getattr(item, 'img', '')), link=str(getattr(item, 'link', '')),
                             preview_url=str(getattr(item, 'preview_url', None)), quality=getattr(item, 'quality', None),
                             time=str(getattr(item, 'time', None)), channel_name=str(getattr(item, 'channel_name', None)),
                             channel_link=str(getattr(item, 'channel_link', None)),
                         )
                     else: logger.warning(f"Skipping object item #{i + 1} missing essential attributes.")
                else: logger.warning(f"Skipping unrecognized video item #{i + 1} type: {type(item)}. Item: {item!r}")
                if video: videos.append(video)
            except Exception as e: logger.error(f"Error parsing video item #{i + 1}: {item!r}. Error: {e}", exc_info=False)
        return videos

    def _parse_tag_results(self, tags_raw: Any) -> list[Tags]:
        if tags_raw is None: return []
        if not isinstance(tags_raw, list): logger.warning(f"Expected list for tags, got {type(tags_raw)}. Returning empty."); return []
        tags_list: list[Tags] = []
        for i, item in enumerate(tags_raw):
             tag = None
             try:
                 if isinstance(item, dict): tag = Tags(name=str(item['name']) if item.get('name') else None, id=str(item['id']) if item.get('id') else None)
                 elif isinstance(item, Tags): tag = item
                 elif hasattr(item, 'name') or hasattr(item, 'id'): tag = Tags(name=str(getattr(item, 'name', None)), id=str(getattr(item, 'id', None)))
                 else: logger.warning(f"Skipping unrecognized tag item #{i + 1} type: {type(item)}"); continue
                 if tag and (tag.name or tag.id): tags_list.append(tag)
                 elif tag: logger.debug(f"Skipping parsed tag item #{i + 1} with no name or id: {item!r}")
             except Exception as e: logger.error(f"Error parsing tag item #{i + 1}: {item!r}. Error: {e}", exc_info=False)
        return tags_list

    @sleep_and_retry  # type: ignore
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)  # type: ignore
    def list_videos(self, limit: int = 12) -> list[VideoDataClass]:
        if not isinstance(limit, int) or limit <= 0: raise ValueError("Limit must be a positive integer.")
        logger.debug(f"Attempting to fetch {limit} videos using list method...")
        try: videos_raw = self.client.list(limit=limit); videos = self._parse_video_results(videos_raw)
        except TypeError as te:
            if 'limit' in str(te):
                logger.warning(f"Engine '{self.engine}' list method may not support 'limit'. Trying without limit."); videos = []
                try: videos_raw = self.client.list(); videos = self._parse_video_results(videos_raw)
                except Exception as e_retry: logger.error(f"Error during list retry: {e_retry}", exc_info=True); raise Exception(f"Failed list retry: {e_retry}") from e_retry
                limited_videos = videos[:limit]; logger.info(f"Fetched {len(videos)}, returning first {len(limited_videos)}."); return limited_videos
            else: logger.error(f"TypeError during list: {te}", exc_info=True); raise Exception(f"Failed list videos: {te}") from te
        except Exception as e: logger.error(f"Error during list call: {e}", exc_info=True); raise Exception(f"Failed list videos: {e}") from e
        logger.info(f"Fetched and parsed {len(videos)} videos (requested list limit: {limit})."); return videos

    @sleep_and_retry  # type: ignore
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)  # type: ignore
    def search_videos(self, keyword: str | None = None, page: int | None = None, limit: int | None = None, **kwargs: Any) -> list[VideoDataClass]:
        search_params: dict[str, Any] = {k: v for k, v in kwargs.items() if v is not None}
        if keyword: search_params['keyword'] = keyword
        search_description_parts: list[str] = [f"{k}='{v}'" for k, v in search_params.items()]
        if page and page > 1: search_description_parts.append(f"page={page} (Note: may be ignored by engine)")
        if limit: search_description_parts.append(f"limit={limit} (Note: may be ignored by engine)")
        if page and page > 1: logger.warning(f"Pagination (page={page}) requested, but may not be supported by engine '{self.engine}'. Ignoring for API call.")
        if limit: logger.warning(f"Limit ({limit}) requested, but may not be supported by engine '{self.engine}'. Ignoring for API call. Results depend on engine default.")
        if not search_params: raise ValueError("Search requires at least one supported criterion (e.g., keyword).")
        search_description = ", ".join(search_description_parts)
        logger.debug(f"Attempting search with effective params: {search_params} (User requested: {search_description})")
        try:
            videos_raw = self.client.search(**search_params); videos = self._parse_video_results(videos_raw)
            logger.info(f"Search yielded {len(videos)} parsed results (based on engine defaults/behavior).")
            if limit and len(videos) > limit: logger.info(f"Truncating {len(videos)} results to requested limit of {limit}."); videos = videos[:limit]
            return videos
        except TypeError as te: logger.error(f"TypeError during search call for engine '{self.engine}' with params {search_params}. Error: {te}", exc_info=True); raise RuntimeError(f"Failed search on engine '{self.engine}' due to parameter issue: {te}") from te
        except Exception as e: logger.error(f"Error during search call: {e}", exc_info=True); raise Exception(f"Failed search videos: {e}") from e


# ==============================================================================
# HTML Generation Function
# ==============================================================================
# (generate_html_output remains the same as the previous corrected version)
def generate_html_output(videos: list[VideoDataClass], query: str, filename: str) -> str:
    if not videos: return ("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>" "<title>No Results</title><style>body{background-color:#1a1a1a;color:#e0e0e0;font-family:sans-serif;text-align:center;padding-top:50px;}" "h1{color:#f0f;text-shadow:0 0 5px #f0f;}</style></head>" f"<body><h1>No videos found for query: '{html.escape(query)}'</h1></body></html>")
    safe_query = html.escape(query); page_title = f"Search Results for '{safe_query}'"
    css = """<style>:root { --neon-cyan: #08f7fe; --neon-green: #39ff14; --dark-bg: #1a1a1a; --medium-dark-bg: #2a2a2a; --light-text: #e0e0e0; --dim-text: #aaa; } body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: var(--dark-bg); color: var(--light-text); } h1 { color: var(--neon-cyan); text-align: center; border-bottom: 2px solid var(--neon-cyan); padding-bottom: 15px; margin-bottom: 30px; text-shadow: 0 0 8px var(--neon-cyan); } p.results-info { text-align: center; color: var(--dim-text); margin-bottom: 30px; } .results-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 25px; } .video-item { background-color: var(--medium-dark-bg); border: 1px solid var(--neon-cyan); border-radius: 8px; overflow: hidden; box-shadow: 0 0 10px rgba(8, 247, 254, 0.3); transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease; position: relative; } .video-item:hover { transform: scale(1.03); border-color: var(--neon-green); box-shadow: 0 0 20px rgba(57, 255, 20, 0.6); z-index: 10; } .video-item a { text-decoration: none; color: inherit; display: block; } .video-item .image-container { position: relative; width: 100%; height: 0; padding-bottom: 56.25%; background-color: #333; border-bottom: 1px solid var(--neon-cyan); overflow: hidden; } .video-item:hover .image-container { border-bottom-color: var(--neon-green); } .video-item .image-container img.thumbnail { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: block; transition: opacity 0.3s ease-in-out; z-index: 1; } .video-item .image-container video.preview-video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; display: none; z-index: 5; } .video-info { padding: 15px; } .video-title { font-size: 1.1em; font-weight: bold; margin: 0 0 10px 0; color: var(--neon-cyan); line-height: 1.3; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; min-height: 2.6em; } .video-details { font-size: 0.9em; color: var(--dim-text); margin-top: 8px; } .video-details span { margin-right: 12px; display: inline-block; white-space: nowrap; } .no-image-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #888; font-style: italic; text-align: center; padding: 10px; display: none; z-index: 2; } .video-item img.thumbnail[data-failed="true"] + .no-image-text { display: block; } .video-item img.thumbnail[data-failed="true"] { opacity: 0; } .video-item.preview-active .image-container img.thumbnail { opacity: 0; visibility: hidden; } .video-item.preview-active .image-container video.preview-video { display: block; }</style>"""
    html_parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>{page_title}</title>", css, f"</head><body><h1>{page_title}</h1><p class='results-info'>Found {len(videos)} videos. Results saved in: {html.escape(filename)}</p><div class='results-container' role='list'>"]
    for video in videos:
        safe_title = html.escape(video.title); safe_link = html.escape(video.link); safe_img_url = html.escape(video.img) if video.img else ""; safe_preview_url = html.escape(video.preview_url) if video.preview_url else ""; safe_time = html.escape(video.time) if video.time else "N/A"; safe_channel = html.escape(video.channel_name) if video.channel_name else "N/A"; img_alt_text = f"Thumbnail for {safe_title}"; preview_attr = f'data-preview-url="{safe_preview_url}"' if safe_preview_url else ''
        html_parts.extend([f"    <div class='video-item' {preview_attr} role='listitem'><a href='{safe_link}' target='_blank' title='{safe_title}'><div class='image-container'>"])
        if safe_img_url: html_parts.extend([f"<img class='thumbnail' src='{safe_img_url}' alt='{img_alt_text}' loading='lazy' onerror='this.setAttribute(\"data-failed\", \"true\"); console.warn(\"Failed to load thumbnail:\", this.src);'><span class='no-image-text'>Preview Unavailable</span>"])
        else: html_parts.extend([f"<img class='thumbnail' src='' alt='{img_alt_text}' style='display: none;' data-failed='true'><span class='no-image-text' style='display: block;'>No Preview Available</span>"])
        html_parts.extend([f"</div><div class='video-info'><div class='video-title'>{safe_title}</div><div class='video-details'><span>Duration: {safe_time}</span>"])
        if safe_channel != "N/A": html_parts.append(f"<span>Channel: {safe_channel}</span>")
        html_parts.extend(["</div></div></a></div>"])
    html_parts.append("</div>")
    js = """<script>
        document.addEventListener('DOMContentLoaded', () => { const videoItems = document.querySelectorAll('.video-item'); let previewTimeout = null; const PREVIEW_DELAY_MS = 250; videoItems.forEach(item => { const imageContainer = item.querySelector('.image-container'); const previewUrl = item.dataset.previewUrl; let previewVideoElement = null; if (!imageContainer || !previewUrl) return; const createAndPlayPreview = () => { if (previewVideoElement) { previewVideoElement.play().catch(e => console.warn(`Re-play prevented: ${e.message}`)); item.classList.add('preview-active'); return; } console.debug(`Creating preview for: ${previewUrl}`); previewVideoElement = document.createElement('video'); previewVideoElement.classList.add('preview-video'); previewVideoElement.src = previewUrl; previewVideoElement.muted = true; previewVideoElement.loop = true; previewVideoElement.preload = 'auto'; previewVideoElement.setAttribute('playsinline', ''); imageContainer.appendChild(previewVideoElement); const playPromise = previewVideoElement.play(); if (playPromise !== undefined) { playPromise.then(() => { console.debug(`Preview started: ${previewUrl}`); item.classList.add('preview-active'); }).catch(error => { console.warn(`Autoplay prevented for ${previewUrl}:`, error.message); }); } else { item.classList.add('preview-active'); } }; const stopAndRemovePreview = () => { item.classList.remove('preview-active'); if (previewVideoElement) { console.debug(`Stopping preview: ${previewUrl}`); previewVideoElement.pause(); previewVideoElement.remove(); previewVideoElement = null; } }; item.addEventListener('mouseenter', () => { clearTimeout(previewTimeout); previewTimeout = setTimeout(createAndPlayPreview, PREVIEW_DELAY_MS); }); item.addEventListener('mouseleave', () => { clearTimeout(previewTimeout); stopAndRemovePreview(); }); const link = item.querySelector('a'); if (link) { link.addEventListener('click', stopAndRemovePreview); } }); });
    </script>"""
    html_parts.append(js); html_parts.append("</body></html>"); return "\n".join(html_parts)

# ==============================================================================
# Helper Functions for Input Prompts
# ==============================================================================


def get_validated_input(prompt: str, default: Any, validation_type: type, positive_only: bool = False) -> Any:
    """Gets user input, validates type, handles defaults, and ensures positivity if needed."""
    while True:
        try:
            user_input_str = input(f"{prompt} [{default}]: ").strip()
            if not user_input_str:
                # User pressed Enter, return default
                logger.debug(f"User accepted default: {default}")
                return default

            # Attempt type conversion
            if validation_type == int:
                value = int(user_input_str)
            elif validation_type == float:
                value = float(user_input_str)
            elif validation_type == Path:
                 # Basic validation for path, more robust checks happen during usage
                 value = Path(user_input_str)
            else:  # Assume string
                value = str(user_input_str)  # Use str() for consistency

            # Check for positivity if required
            if positive_only and validation_type in [int, float] and value <= 0:
                 print("Input must be a positive number. Please try again.")
                 continue  # Re-prompt

            logger.debug(f"User entered valid input: {value}")
            return value

        except ValueError:
            print(f"Invalid input. Please enter a valid {validation_type.__name__}.")
        except Exception as e:
            print(f"An unexpected error occurred during input: {e}")
            # Decide if you want to re-prompt or exit on other errors
            # For simplicity, we re-prompt here
            continue

# ==============================================================================
# Main Execution Function (Using Prompts)
# ==============================================================================


def main():
    """Main function using interactive prompts to gather info, run client, and generate output."""
    logger.info("--- Starting PornLib Search Script (Interactive Mode) ---")

    # --- Gather Input via Prompts ---
    try:
        search_query = ""
        while not search_query:
             search_query = input("Enter search query: ").strip()
             if not search_query: print("Search query cannot be empty.")

        limit = get_validated_input("Max results to fetch?", DEFAULT_SEARCH_LIMIT, int, positive_only=True)
        page = get_validated_input("Page number?", DEFAULT_PAGE, int, positive_only=True)
        engine = get_validated_input("PornLib engine?", DEFAULT_ENGINE, str)
        soup_sleep = get_validated_input("Soup sleep (seconds)?", DEFAULT_SOUP_SLEEP, float, positive_only=False)  # Sleep can be 0
        output_dir_str = get_validated_input("Output directory?", DEFAULT_OUTPUT_DIR_STR, str)
        output_dir = Path(output_dir_str)  # Convert to Path after getting input
        filename_prefix_format = get_validated_input("Filename prefix format?", DEFAULT_FILENAME_PREFIX, str)
        auto_open_str = get_validated_input("Auto-open HTML file (y/n)?", DEFAULT_AUTO_OPEN, str)
        auto_open = auto_open_str.lower().startswith('y')

    except (KeyboardInterrupt, EOFError):
        logger.info("\nInput cancelled by user. Exiting.")
        return
    except Exception as e:
         logger.critical(f"Failed to gather input settings: {e}", exc_info=True)
         return  # Exit if input fails critically

    logger.info(f"Settings: Engine='{engine}', Query='{search_query}', Limit={limit}, Page={page}, Output='{output_dir}', AutoOpen={auto_open}")

    client: XvideosClient | None = None
    try:
        # --- Initialize the client ---
        client = XvideosClient(engine=engine, soup_sleep=soup_sleep)

        # --- Perform Search ---
        logger.info(f"Performing search for query: '{search_query}'...")
        search_results: list[VideoDataClass] = client.search_videos(
            keyword=search_query,
            page=page,
            limit=limit
        )

        if search_results:
            logger.info(f"Successfully retrieved and parsed {len(search_results)} videos.")

            # --- Generate Filename ---
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query_part = "".join(c for c in search_query.replace(' ', '_') if c.isalnum() or c in ['_', '-'])[:40]
            try:
                filename_stem = filename_prefix_format.format(
                    engine=engine, query_part=safe_query_part, timestamp=timestamp
                )
            except KeyError as e:
                 logger.error(f"Invalid key in filename prefix format string: {e}. Using default format.")
                 filename_stem = DEFAULT_FILENAME_PREFIX.format(
                      engine=engine, query_part=safe_query_part, timestamp=timestamp
                 )
            except Exception as fmt_e:  # Catch other formatting errors
                 logger.error(f"Error formatting filename prefix: {fmt_e}. Using basic name.")
                 filename_stem = f"{engine}_search_{timestamp}"  # Fallback

            output_filename = f"{filename_stem}.html"
            output_path = output_dir.resolve() / output_filename

            # --- Ensure Output Directory Exists ---
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured output directory exists: {output_dir.resolve()}")
            except OSError as e:
                logger.error(f"Failed to create output directory '{output_dir}': {e}. Check permissions.")
                logger.warning(f"Attempting to save to current directory: {Path.cwd()}")
                output_path = Path.cwd() / output_filename  # Fallback

            # --- Generate HTML ---
            logger.info("Generating HTML output...")
            html_content = generate_html_output(search_results, search_query, output_filename)

            # --- Save HTML to File ---
            logger.info(f"Attempting to save results to: {output_path}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f: f.write(html_content)
                logger.info(f"Successfully saved results to: {output_path}")
                # --- Auto-open File ---
                if auto_open:
                    try:
                        logger.info("Attempting to open HTML file...")
                        if sys.platform.startswith('win'): os.startfile(output_path)
                        elif sys.platform.startswith('darwin'): os.system(f'open "{output_path}"')
                        else: os.system(f'xdg-open "{output_path}"')
                    except Exception as open_err: logger.warning(f"Could not automatically open file '{output_path}': {open_err}")
            except OSError as e: logger.error(f"Error saving HTML file '{output_path}': {e}", exc_info=True)
            except Exception as e: logger.error(f"Unexpected error writing HTML file: {e}", exc_info=True)

        else:
            logger.warning(f"No videos found for query '{search_query}' using engine '{engine}'. No HTML file generated.")

    # --- Error Handling ---
    except ValueError as ve: logger.critical(f"Input/Validation Error: {ve}", exc_info=False)
    except NotImplementedError as nie: logger.critical(f"Feature Error: {nie}", exc_info=True)
    except RuntimeError as rte: logger.critical(f"Runtime Error: {rte}", exc_info=True)
    except ImportError: logger.critical("ImportError: pornLib library missing/corrupted.", exc_info=False)
    except KeyboardInterrupt: logger.info("\nProcess interrupted by user (Ctrl+C). Exiting.")
    except Exception as e: logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    finally: logger.info("--- PornLib Search Script Finished ---")


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == "__main__":
    # --- Usage Instructions ---
    # Install dependencies: pip install pornlib ratelimit
    # Run the script: python xvid_prompt.py
    # Follow the interactive prompts.
    #
    # Note on Previews: Hover previews require 'preview_url' in VideoDataClass.
    # This script DOES NOT automatically scrape these URLs from video pages.
    # --------------------------
    main()
