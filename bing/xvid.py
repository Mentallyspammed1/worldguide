import datetime  # For filenames and potential use with time fields
import html  # For escaping potentially problematic characters in HTML
import logging
import os  # For path manipulation
import sys
from dataclasses import dataclass
from typing import Any

import pornLib  # Assuming this library exists and is installed
from ratelimit import limits, sleep_and_retry

# ==============================================================================
# Configuration
# ==============================================================================

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log explicitly to standard output
    ]
)
logger = logging.getLogger(__name__)

# --- Rate Limiting Configuration ---
# Example: Allow 50 API calls within any 60-second window
API_CALLS_LIMIT = 50
API_PERIOD_SECONDS = 60

# --- Output Configuration ---
# Default directory to save HTML files (can be overridden)
DEFAULT_OUTPUT_DIR = "."  # Current directory
# Maximum number of search results to fetch per query (adjust as needed)
DEFAULT_SEARCH_LIMIT = 30  # pornLib might have its own upper limits too

# ==============================================================================
# Data Classes
# ==============================================================================
# Using dataclasses for structured data representation with type hints.


@dataclass
class VideoDataClass:
    """Represents metadata for a single video retrieved from the platform.
    Fields are optional where the underlying library might not provide them.

    Attributes:
        title: The title of the video. (Required for display)
        img: URL of the video's thumbnail image. (Required for display)
        link: URL to the video's page. (Required for display)
        quality: An indicator of video quality (e.g., 720, 'HD'). Optional.
        time: Duration of the video as a string (e.g., "10:30"). Optional.
        channel_name: The name of the channel or uploader. Optional.
        channel_link: URL to the channel's page. Optional.
    """
    title: str
    img: str
    link: str
    quality: Any | None = None  # Use Any if type varies (int, str)
    time: str | None = None
    channel_name: str | None = None
    channel_link: str | None = None


@dataclass
class VideoDownloadDataClass:
    """Represents available download links for a video in various qualities.

    Attributes:
        low: URL for the low-quality video download, if available.
        high: URL for the high-quality video download, if available.
        hls: URL for the HLS (HTTP Live Streaming) manifest, if available.
    """
    low: str | None = None
    high: str | None = None
    hls: str | None = None


@dataclass
class Tags:
    """Represents a tag associated with videos or search results.

    Attributes:
        name: The display name of the tag (e.g., "Big Tits"). Optional.
        id: A unique identifier for the tag, often used for searching. Optional.
    """
    name: str | None = None
    id: str | None = None

# ==============================================================================
# Xvideos Client Class
# ==============================================================================


class XvideosClient:
    """A client for interacting with the xvideos platform via the pornLib library.

    Encapsulates listing, searching, fetching tags, and retrieving download links,
    incorporating rate limiting, logging, robust parsing, and error handling.
    """

    def __init__(self, engine: str = "xvideos", soup_sleep: float = 1.0):
        """Initializes the Xvideos client using the pornLib library.

        Args:
            engine (str): The target site engine identifier for pornLib.
            soup_sleep (float): Delay for the library between certain actions.

        Raises:
            ImportError: If `pornLib` is not installed.
            RuntimeError: If pornLib fails to initialize.
        """
        try:
            self.client = pornLib.PornLib(engine=engine, soupSleep=soup_sleep)
            logger.info(f"Successfully initialized XvideosClient: engine='{engine}', soup_sleep={soup_sleep}s")
        except ImportError:
            logger.error("Critical Error: pornLib library not found. Please install it (e.g., 'pip install pornlib').")
            raise
        except Exception as e:
            logger.error(f"Critical Error: Failed to initialize pornLib client for engine '{engine}'. Error: {e}", exc_info=True)
            raise RuntimeError(f"XvideosClient initialization failed for engine '{engine}': {e}") from e

    # --- Helper methods for parsing ---

    def _parse_video_results(self, results_raw: Any) -> list[VideoDataClass]:
        """Safely parses raw results from pornLib into a list of VideoDataClass objects.
        Handles None, non-list inputs, dictionaries, objects, and missing fields.

        Args:
            results_raw: The raw output from a pornLib method returning video data.

        Returns:
            A list of valid VideoDataClass instances.
        """
        if results_raw is None:
            logger.debug("Received None for video results, returning empty list.")
            return []
        if not isinstance(results_raw, list):
            logger.warning(f"Expected a list of video results, but got {type(results_raw)}. Returning empty list.")
            return []

        videos = []
        for item in results_raw:
            try:
                video = None
                if isinstance(item, dict):
                    # Check for essential keys before creating dataclass
                    if all(k in item for k in ['title', 'img', 'link']):
                         video = VideoDataClass(
                             title=str(item.get('title', '')),  # Ensure string type
                             img=str(item.get('img', '')),
                             link=str(item.get('link', '')),
                             quality=item.get('quality'),  # Keep original type or None
                             time=str(item.get('time')) if item.get('time') else None,
                             channel_name=str(item.get('channel_name')) if item.get('channel_name') else None,
                             channel_link=str(item.get('channel_link')) if item.get('channel_link') else None,
                         )
                    else:
                        logger.warning(f"Skipping video dictionary item due to missing essential keys (title, img, link): {item}")

                elif isinstance(item, VideoDataClass):
                     # Already the correct type
                     video = item
                elif hasattr(item, 'title') and hasattr(item, 'img') and hasattr(item, 'link'):
                     # Handle other object types (like SimpleNamespace) if they have needed attrs
                     video = VideoDataClass(
                         title=str(getattr(item, 'title', '')),
                         img=str(getattr(item, 'img', '')),
                         link=str(getattr(item, 'link', '')),
                         quality=getattr(item, 'quality', None),
                         time=str(getattr(item, 'time', None)) if hasattr(item, 'time') else None,
                         channel_name=str(getattr(item, 'channel_name', None)) if hasattr(item, 'channel_name') else None,
                         channel_link=str(getattr(item, 'channel_link', None)) if hasattr(item, 'channel_link') else None,
                     )
                else:
                    logger.warning(f"Skipping unrecognized video item type: {type(item)}. Item: {item!r}")
                    continue

                # Final validation and appending
                if video and video.title and video.link and video.img:
                    videos.append(video)
                elif video:  # Log if created but failed validation
                     logger.warning(f"Skipping parsed video item with missing essential data: title='{video.title}', link='{video.link}', img='{video.img}'")

            except Exception as e:
                # Log error for the specific item but continue processing others
                logger.error(f"Error parsing individual video item: {item!r}. Error: {e}", exc_info=False)  # Keep traceback minimal per item
        return videos

    def _parse_tag_results(self, tags_raw: Any) -> list[Tags]:
        """Safely parses raw results from pornLib into a list of Tags objects.
        Handles None, non-list inputs, dictionaries, and objects.

        Args:
            tags_raw: The raw output from a pornLib method returning tag data.

        Returns:
            A list of valid Tags instances.
        """
        if tags_raw is None:
            logger.debug("Received None for tag results, returning empty list.")
            return []
        if not isinstance(tags_raw, list):
            logger.warning(f"Expected a list of tags, but got {type(tags_raw)}. Returning empty list.")
            return []

        tags_list = []
        for item in tags_raw:
            try:
                tag = None
                if isinstance(item, dict):
                    tag = Tags(
                        name=str(item.get('name')) if item.get('name') else None,
                        id=str(item.get('id')) if item.get('id') else None
                    )
                elif isinstance(item, Tags):
                    tag = item
                elif hasattr(item, 'name') or hasattr(item, 'id'):  # Check if it looks like a tag
                    tag = Tags(
                        name=str(getattr(item, 'name', None)),
                        id=str(getattr(item, 'id', None))
                    )
                else:
                    logger.warning(f"Skipping unrecognized tag item type: {type(item)}. Item: {item!r}")
                    continue

                # Add tag only if it has at least a name or an id
                if tag and (tag.name or tag.id):
                     tags_list.append(tag)
                elif tag:
                     logger.debug(f"Skipping parsed tag item with no name or id: {item!r}")

            except Exception as e:
                 logger.error(f"Error parsing individual tag item: {item!r}. Error: {e}", exc_info=False)
        return tags_list

    # --- Public API Methods ---

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def list_videos(self, limit: int = 12) -> list[VideoDataClass]:
        """Fetches a list of the latest or trending videos from the platform.

        Applies rate limiting and retries. Uses helper to parse results.

        Args:
            limit (int): Maximum number of videos to retrieve. Must be positive.

        Returns:
            List[VideoDataClass]: Parsed video data objects. Empty list on failure/no results.

        Raises:
            ValueError: If limit is invalid.
            Exception: For unexpected errors during API call or processing.
        """
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Invalid limit specified. Limit must be a positive integer.")

        logger.debug(f"Attempting to fetch {limit} videos...")
        try:
            videos_raw = self.client.list(limit=limit)
            videos = self._parse_video_results(videos_raw)
            logger.info(f"Successfully fetched and parsed {len(videos)} videos (requested limit: {limit}).")
            return videos
        except Exception as e:
            logger.error(f"Error fetching video list (limit={limit}): {e}", exc_info=True)
            # Re-raise the exception to allow upstream handling
            raise Exception(f"Failed to list videos due to: {e}") from e

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def get_tags(self, keyword: str | None = None) -> list[Tags]:
        """Fetches tags from the platform, optionally filtered by keyword.

        Applies rate limiting and retries. Uses helper to parse results.

        Args:
            keyword (Optional[str]): Keyword to filter tags by.

        Returns:
            List[Tags]: Parsed tag objects. Empty list on failure/no results.

        Raises:
            Exception: For unexpected errors during API call or processing.
        """
        action_desc = "all available tags" if not keyword else f"tags matching keyword '{keyword}'"
        logger.debug(f"Attempting to fetch {action_desc}...")
        try:
            tags_raw = self.client.tags(keyword=keyword)
            tags_result = self._parse_tag_results(tags_raw)
            logger.info(f"Successfully fetched and parsed {len(tags_result)} {action_desc}.")
            return tags_result
        except Exception as e:
            logger.error(f"Error fetching tags (keyword={keyword}): {e}", exc_info=True)
            raise Exception(f"Failed to get tags due to: {e}") from e

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def search_videos(self,
                      keyword: str | None = None,
                      channel: str | None = None,
                      tag: str | None = None,  # Usually tag ID or name
                      tags: list[Tags] | None = None,  # List of Tag objects
                      best: str | None = None,  # e.g., 'month', check pornLib docs
                      page: int | None = None,
                      limit: int | None = None) -> list[VideoDataClass]:
        """Searches for videos based on various criteria.

        Applies rate limiting and retries. Uses helper to parse results.
        Consult `pornLib` documentation for specific parameter interactions.

        Args:
            keyword: Search by keyword.
            channel: Filter by channel name/ID.
            tag: Filter by single tag name/ID.
            tags: Filter by list of Tags (consult pornLib how it expects this - IDs? Names?).
            best: Filter for top videos in a period.
            page: Page number for results.
            limit: Max results (if supported by pornLib search).

        Returns:
            List[VideoDataClass]: Parsed video data objects matching criteria.

        Raises:
            ValueError: If no primary search criteria are provided.
            Exception: For unexpected errors during API call or processing.
        """
        search_params: dict[str, Any] = {}
        search_description_parts: list[str] = []

        # Build search parameters and description
        if keyword: search_params['keyword'] = keyword; search_description_parts.append(f"keyword='{keyword}'")
        if channel: search_params['channel'] = channel; search_description_parts.append(f"channel='{channel}'")
        if tag: search_params['tag'] = tag; search_description_parts.append(f"tag='{tag}'")
        if tags:
            # Verify how pornLib expects 'tags'. Assuming list of IDs/names.
            tag_ids_or_names = [t.id or t.name for t in tags if t.id or t.name]
            if tag_ids_or_names:
                 search_params['tags'] = tag_ids_or_names  # Adjust key if needed ('tag' vs 'tags')
                 search_description_parts.append(f"tags={tag_ids_or_names}")
            else:
                 logger.warning("Provided 'tags' list, but no usable names or IDs found in Tag objects.")
        if best: search_params['best'] = best; search_description_parts.append(f"best='{best}'")
        if page is not None and isinstance(page, int) and page > 0:
            search_params['page'] = page; search_description_parts.append(f"page={page}")
        else:
             if page is not None: logger.warning(f"Invalid page number ignored: {page}")
        if limit is not None and isinstance(limit, int) and limit > 0:
            search_params['limit'] = limit; search_description_parts.append(f"limit={limit}")
        else:
             if limit is not None: logger.warning(f"Invalid limit ignored: {limit}")

        # Validate search criteria
        if not any([keyword, channel, tag, tags, best]):  # Check if any primary criterion exists
            raise ValueError("Search requires at least one criterion: keyword, channel, tag, tags, or best.")

        # Log warning if multiple criteria are used (behavior depends on library)
        active_criteria_count = sum(1 for criteria in [keyword, channel, tag, 'tags' in search_params, best] if criteria)
        if active_criteria_count > 1:
            logger.warning("Multiple primary search criteria used. Behavior depends on pornLib implementation.")

        search_description = ", ".join(search_description_parts) if search_description_parts else "provided criteria"
        logger.debug(f"Attempting to search videos with {search_description}...")

        try:
            # Use **search_params to pass arguments to pornLib
            videos_raw = self.client.search(**search_params)
            videos = self._parse_video_results(videos_raw)
            logger.info(f"Search with {search_description} yielded {len(videos)} parsed results.")
            return videos
        except Exception as e:
            logger.error(f"Error searching videos with params {search_params}: {e}", exc_info=True)
            raise Exception(f"Failed to search videos due to: {e}") from e

    @sleep_and_retry
    @limits(calls=API_CALLS_LIMIT, period=API_PERIOD_SECONDS)
    def get_download_link(self, video_link: str) -> VideoDownloadDataClass | None:
        """Fetches available download links (low, high, HLS) for a specific video URL.

        Applies rate limiting and retries. Parses result into VideoDownloadDataClass.

        Args:
            video_link (str): The full URL of the video page.

        Returns:
            Optional[VideoDownloadDataClass]: Object with download URLs, or None if unavailable/error.

        Raises:
            ValueError: If video_link is invalid.
            NotImplementedError: If the download link feature is missing in pornLib.
            Exception: For unexpected errors during API call or processing.
        """
        if not isinstance(video_link, str) or not video_link.strip():
            raise ValueError("Invalid video link provided: must be a non-empty string URL.")
        if not video_link.startswith(('http://', 'https://')):
             logger.warning(f"Video link '{video_link}' does not start with http/https. Proceeding cautiously.")

        logger.debug(f"Attempting to fetch download links for video: {video_link}")
        try:
            # Verify the exact method name in your pornLib version. Assume 'getDownloadLink'.
            method_name = 'getDownloadLink'
            if not hasattr(self.client, method_name):
                 alt_method_name = 'get_download_link'  # Common alternative
                 if hasattr(self.client, alt_method_name):
                     method_name = alt_method_name
                 else:
                     # Neither common name exists, raise NotImplementedError directly
                      raise NotImplementedError(f"Download link functionality ('{method_name}' or '{alt_method_name}') not available in this pornLib setup.")

            download_data_raw = getattr(self.client, method_name)(video_link)

            # Handle cases where pornLib might return None (e.g., 404)
            if download_data_raw is None:
                 logger.warning(f"Could not retrieve download links for {video_link}. pornLib returned None (video might not exist or links unavailable).")
                 return None

            # Process the returned data (adapt based on actual return type)
            result = None
            if isinstance(download_data_raw, dict):
                logger.debug("Received dict, converting to VideoDownloadDataClass.")
                result = VideoDownloadDataClass(
                    low=download_data_raw.get('low'),
                    high=download_data_raw.get('high'),
                    hls=download_data_raw.get('hls')
                )
            elif isinstance(download_data_raw, VideoDownloadDataClass):
                 logger.debug("Received VideoDownloadDataClass instance directly.")
                 result = download_data_raw
            elif hasattr(download_data_raw, 'low') or hasattr(download_data_raw, 'high') or hasattr(download_data_raw, 'hls'):
                 logger.debug(f"Received object type {type(download_data_raw)}, attempting attribute extraction.")
                 result = VideoDownloadDataClass(
                      low=getattr(download_data_raw, 'low', None),
                      high=getattr(download_data_raw, 'high', None),
                      hls=getattr(download_data_raw, 'hls', None)
                 )
            else:
                 logger.error(f"Received unexpected data format for download links: {type(download_data_raw)}. Content: {download_data_raw!r}. Could not parse.")
                 return None  # Cannot reliably determine links

            # Log if no actual links were found in the parsed data
            if result and not any([result.low, result.high, result.hls]):
                logger.warning(f"Successfully processed response for {video_link}, but no download links (low, high, hls) were found.")
                # Return the object with all None fields

            logger.info(f"Successfully fetched and processed download links for video: {video_link}")
            return result

        except AttributeError as ae:
             # This might catch other attribute errors, but we already checked for the method
             logger.error(f"An unexpected AttributeError occurred during download link retrieval: {ae}", exc_info=True)
             raise Exception(f"An attribute error occurred: {ae}") from ae
        except NotImplementedError as nie:  # Catch the specific error raised above
             logger.error(f"Feature Error: {nie}", exc_info=True)
             raise  # Re-raise NotImplementedError
        except Exception as e:
            logger.error(f"Error fetching or processing download link for {video_link}: {e}", exc_info=True)
            raise Exception(f"Failed to get download links due to: {e}") from e


# ==============================================================================
# HTML Generation Function (Dark/Neon Theme)
# ==============================================================================

def generate_html_output(videos: list[VideoDataClass], query: str, filename: str) -> str:
    """Generates an HTML string containing video results with previews and links,
    styled with a dark theme and neon accents.

    Args:
        videos: A list of VideoDataClass objects.
        query: The search query string used.
        filename: The target filename (used for title).

    Returns:
        An HTML string representing the results page.
    """
    if not videos:
        # Basic dark theme for the "No Results" page too
        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
            "<title>No Results</title><style>body { background-color: #1a1a1a; color: #e0e0e0; font-family: sans-serif; text-align: center; padding-top: 50px; } "
            "h1 { color: #ff00ff; /* Neon Pink for emphasis */ text-shadow: 0 0 5px #ff00ff;} </style></head>"
            f"<body><h1>No videos found for query: '{html.escape(query)}'</h1></body></html>"
        )

    # Sanitize query for display
    safe_query = html.escape(query)
    page_title = f"Search Results for '{safe_query}'"

    # Dark theme CSS with Neon accents (Cyan/Blue and Green)
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a; /* Dark background */
            color: #e0e0e0; /* Light grey text */
        }
        h1 {
            color: #08f7fe; /* Neon Cyan */
            text-align: center;
            border-bottom: 2px solid #08f7fe;
            padding-bottom: 15px;
            margin-bottom: 30px;
            text-shadow: 0 0 8px rgba(8, 247, 254, 0.7); /* Neon glow */
        }
        p.results-info { /* Class for the info paragraph */
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
        }
        .results-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); /* Slightly larger min size */
            gap: 25px;
        }
        .video-item {
            background-color: #2a2a2a; /* Slightly lighter dark shade */
            border: 1px solid #08f7fe; /* Neon Cyan border */
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(8, 247, 254, 0.3); /* Subtle neon glow */
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }
        .video-item:hover {
            transform: scale(1.03); /* Slightly zoom in */
            border-color: #39ff14; /* Neon Green border on hover */
            box-shadow: 0 0 20px rgba(57, 255, 20, 0.6); /* Brighter Neon Green glow */
        }
        .video-item a {
            text-decoration: none;
            color: inherit; /* Inherit body text color */
            display: block;
        }
        .video-item .image-container { /* Container for image and placeholder */
             position: relative; /* Needed for absolute positioning of placeholder if used */
             width: 100%;
             height: 0; /* Set height via padding-bottom for aspect ratio */
             padding-bottom: 56.25%; /* 16:9 aspect ratio */
             background-color: #333; /* Background shown while loading or if image fails */
             border-bottom: 1px solid #08f7fe; /* Match border color */
        }
        .video-item:hover .image-container {
             border-bottom-color: #39ff14; /* Match hover border color */
        }
        .video-item img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover; /* Crop image nicely */
            display: block;
        }
        .video-info {
            padding: 15px;
        }
        .video-title {
            font-size: 1.1em;
            font-weight: bold;
            margin: 0 0 10px 0;
            color: #08f7fe; /* Neon Cyan title */
            line-height: 1.3;
             /* Prevent long titles breaking layout awkwardly */
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2; /* Limit to 2 lines */
            -webkit-box-orient: vertical;
            min-height: 2.6em; /* Ensure space for 2 lines */
        }
        .video-details {
            font-size: 0.9em;
            color: #aaa; /* Dimmer grey for details */
            margin-top: 8px;
        }
        .video-details span {
            margin-right: 12px;
            display: inline-block; /* Better spacing */
            white-space: nowrap; /* Prevent wrapping */
        }
        .no-image-text { /* Placeholder text styling */
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #888;
            font-style: italic;
            text-align: center;
            padding: 10px;
            display: none; /* Hidden by default */
        }
        /* Show placeholder text if image fails */
        .video-item img[data-failed="true"] + .no-image-text {
            display: block;
        }
        /* Hide image visually if it fails */
         .video-item img[data-failed="true"] {
            opacity: 0;
        }
    </style>
    """

    # Start HTML document
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"  <title>{page_title}</title>",
        css,
        "</head>",
        "<body>",
        f"  <h1>{page_title}</h1>",
        f"  <p class='results-info'>Found {len(videos)} videos. Results saved in: {html.escape(filename)}</p>",  # Added class
        "  <div class='results-container'>"
    ]

    # Add each video item
    for video in videos:
        # Escape data for safe HTML embedding
        safe_title = html.escape(video.title)
        safe_link = html.escape(video.link)
        safe_img_url = html.escape(video.img) if video.img else ""
        safe_time = html.escape(video.time) if video.time else "N/A"
        safe_channel = html.escape(video.channel_name) if video.channel_name else "N/A"
        img_alt_text = f"Thumbnail for {safe_title}"

        html_parts.append("    <div class='video-item'>")
        html_parts.append(f"      <a href='{safe_link}' target='_blank' title='{safe_title}'>")

        # Image container for aspect ratio and placeholder
        html_parts.append("        <div class='image-container'>")
        if safe_img_url:
             # Added onerror to set a data attribute, CSS handles showing placeholder
             html_parts.append(f"          <img src='{safe_img_url}' alt='{img_alt_text}' loading='lazy' onerror='this.setAttribute(\"data-failed\", \"true\");'>")
             html_parts.append("          <span class='no-image-text'>Preview Unavailable</span>")  # Placeholder text
        else:
             # If no image URL, show placeholder directly
             html_parts.append(f"          <img src='' alt='{img_alt_text}' style='display: none;' data-failed='true'>")  # Hidden img triggers placeholder via CSS
             html_parts.append("          <span class='no-image-text' style='display: block;'>No Preview Available</span>")  # Make placeholder visible
        html_parts.append("        </div>")  # end image-container

        # Video Information
        html_parts.append("        <div class='video-info'>")
        html_parts.append(f"          <div class='video-title'>{safe_title}</div>")
        html_parts.append("          <div class='video-details'>")
        html_parts.append(f"              <span>Duration: {safe_time}</span>")
        if video.channel_name:  # Only show channel if available
             html_parts.append(f"              <span>Channel: {safe_channel}</span>")
        # Add Quality if available (example)
        # if video.quality:
        #      html_parts.append(f"              <span>Quality: {html.escape(str(video.quality))}</span>")
        html_parts.append("          </div>")  # end video-details
        html_parts.append("        </div>")  # end video-info
        html_parts.append("      </a>")  # end link
        html_parts.append("    </div>")  # end video-item

    # Close HTML document
    html_parts.append("  </div>")  # end results-container
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)

# ==============================================================================
# Main Execution Function
# ==============================================================================


def main():
    """Main function to initialize the client, prompt the user for a search query,
    perform the search, and save the results to a styled HTML file.
    """
    logger.info("--- Starting Xvideos Client ---")
    client: XvideosClient | None = None

    try:
        # --- Initialize the client ---
        # Adjust soup_sleep based on testing; 1.0 or 1.5 is often reasonable.
        client = XvideosClient(soup_sleep=1.0)

        # --- Get User Input ---
        search_query = ""
        while True:
            try:
                raw_input = input("Enter your search query (or press Enter to exit): ").strip()
                if not raw_input:
                    logger.info("No query entered. Exiting.")
                    return  # Exit the main function if user enters nothing
                search_query = raw_input
                logger.info(f"User entered search query: '{search_query}'")
                break
            except EOFError:  # Handle Ctrl+D or piped input ending
                 logger.info("Input stream closed. Exiting.")
                 return

        # --- Perform Search ---
        logger.info(f"Searching for videos matching '{search_query}' (limit {DEFAULT_SEARCH_LIMIT})...")
        search_results: list[VideoDataClass] = []
        try:
            # Use the search limit from configuration
            search_results = client.search_videos(keyword=search_query, limit=DEFAULT_SEARCH_LIMIT)

            if search_results:
                logger.info(f"Found {len(search_results)} videos matching '{search_query}'.")

                # --- Generate Filename ---
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize query for filename (allow letters, numbers, underscore, hyphen)
                safe_query_part = "".join(c if c.isalnum() or c in ['_', '-'] else '' for c in search_query.replace(' ', '_'))[:40]  # Limit length
                filename = f"xvideos_search_{safe_query_part}_{timestamp}.html"
                filepath = os.path.join(DEFAULT_OUTPUT_DIR, filename)

                # Ensure output directory exists
                try:
                    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
                except OSError as e:
                     logger.error(f"Failed to create output directory '{DEFAULT_OUTPUT_DIR}': {e}. Saving to current directory instead.")
                     filepath = filename  # Fallback to current dir

                # --- Generate HTML ---
                logger.info("Generating HTML output...")
                html_content = generate_html_output(search_results, search_query, filename)

                # --- Save HTML to File ---
                logger.info(f"Attempting to save results to: {filepath}")
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    logger.info(f"Successfully saved {len(search_results)} results to: {filepath}")
                    # Try to open the file automatically (optional, platform dependent)
                    try:
                        if sys.platform.startswith('win'):
                            os.startfile(filepath)
                        elif sys.platform.startswith('darwin'):  # macOS
                            os.system(f'open "{filepath}"')
                        else:  # Linux variants
                            os.system(f'xdg-open "{filepath}"')
                    except Exception as open_err:
                        logger.warning(f"Could not automatically open the file: {open_err}")

                except OSError as e:
                    logger.error(f"Error saving results to HTML file '{filepath}': {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"An unexpected error occurred while writing the HTML file: {e}", exc_info=True)

            else:
                logger.warning(f"No videos found matching the query '{search_query}'. No HTML file generated.")

        except ValueError as ve:  # Catch specific validation errors from search
            logger.error(f"Input Error during search: {ve}")
        except NotImplementedError as nie:  # Catch if search/list is not implemented
            logger.error(f"Feature Error: {nie}")
        except Exception as e:  # Catch broader errors during search API call
            logger.error(f"Failed to search videos with query '{search_query}': {e}", exc_info=True)

    # --- Broader Error Handling for Initialization / Main Flow ---
    except ValueError as ve:  # e.g., invalid limit in list_videos if called directly
        logger.critical(f"Configuration or Input Error: {ve}", exc_info=True)
    except NotImplementedError as nie:  # e.g., download link method missing
        logger.critical(f"Feature Error: A required feature is not implemented or available: {nie}", exc_info=True)
    except RuntimeError as rte:  # e.g., Client init failed
         logger.critical(f"Runtime Error (e.g., client initialization failed): {rte}", exc_info=True)
    except ImportError:
         # Already logged in __init__, but good practice
         logger.critical("ImportError: pornLib library is missing. Please install it.", exc_info=False)
    except KeyboardInterrupt:
         logger.info("\nProcess interrupted by user (Ctrl+C). Exiting.")
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.critical(f"An unexpected critical error occurred in main execution: {e}", exc_info=True)
    finally:
        # Potential cleanup code could go here if the client needed explicit closing
        # if client and hasattr(client, 'close'):
        #     logger.info("Closing client resources...")
        #     client.close()
        logger.info("--- Xvideos Client Finished ---")

# ==============================================================================
# Script Entry Point
# ==============================================================================


if __name__ == "__main__":
    # Execute the main function when the script is run directly
    main()
