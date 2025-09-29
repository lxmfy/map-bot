import io
import logging
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
import LXMF
from lxmfy import Attachment, AttachmentType, LXMFBot, IconAppearance, pack_icon_appearance_field
from mgrs import MGRS
from PIL import Image
from pmtiles import reader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAP_ZOOM_LEVEL = 14
MAP_GRID_SIZE = 3
TILE_SIZE = 256
USER_AGENT = "lxmfy_map_bot/1.0 (requests)"

TILE_PROVIDERS = {
    "pmtiles": {
        "type": "pmtiles",
        "attribution": "Protomaps - Offline tiles",
    },
}

geolocator = Nominatim(user_agent="lxmfy_map_bot/1.0")
m = MGRS()

PMTILES_DIR = Path("data/pmtiles")
PMTILES_URL_BASE = "https://build.protomaps.com"

# Bot icon appearance configuration
BOT_ICON = IconAppearance(
    icon_name="map",  # Material Symbols map icon
    fg_color=b'\xFF\xFF\xFF',  # White foreground
    bg_color=b'\x1E\x88\xE5'   # Blue background (OpenStreetMap blue)
)

BOT_ICON_FIELD = pack_icon_appearance_field(BOT_ICON)


def download_latest_pmtiles() -> str | None:
    """
    Download the latest PMTiles file from Protomaps.

    Returns:
        str | None: Path to downloaded PMTiles file if successful, None if failed
    """
    try:
        PMTILES_DIR.mkdir(parents=True, exist_ok=True)

        for days_ago in range(7):
            date = datetime.now() - timedelta(days=days_ago)
            date_str = date.strftime("%Y%m%d")
            filename = f"{date_str}.pmtiles"
            url = f"{PMTILES_URL_BASE}/{filename}"
            local_path = PMTILES_DIR / filename

            logger.info("Trying to download PMTiles: %s", url)

            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(local_path, 'wb') as f:
                    downloaded = 0
                    last_progress = -1
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                progress_int = int(progress)
                                if progress_int > last_progress:
                                    logger.info("Download progress: %d%%", progress_int)
                                    last_progress = progress_int

                logger.info("Successfully downloaded PMTiles: %s (%d bytes)", filename, downloaded)
                return str(local_path)

            except requests.exceptions.RequestException as e:
                logger.warning("Failed to download %s: %s", url, e)
                continue

        logger.error("Could not download any PMTiles file in the last 7 days")
        return None

    except Exception as e:
        logger.error("Error in download_latest_pmtiles: %s", e)
        return None


def get_pmtiles_path() -> str | None:
    """
    Get the path to the most recent PMTiles file (existing files only).

    Returns:
        str | None: Path to PMTiles file if available, None if no files exist
    """
    try:
        if PMTILES_DIR.exists():
            pmtiles_files = list(PMTILES_DIR.glob("*.pmtiles"))
            if pmtiles_files:
                latest_file = max(pmtiles_files, key=lambda f: f.stat().st_mtime)
                logger.info("Using existing PMTiles file: %s", latest_file)
                return str(latest_file)

        logger.info("No PMTiles files found locally")
        return None

    except Exception as e:
        logger.error("Error getting PMTiles path: %s", e)
        return None


def download_pmtiles_by_date(date_str: str | None = None) -> str | None:
    """
    Download PMTiles file for a specific date or the latest available.

    Args:
        date_str: Date in YYYYMMDD format, or None for latest

    Returns:
        str | None: Path to downloaded PMTiles file if successful, None if failed
    """
    try:
        PMTILES_DIR.mkdir(parents=True, exist_ok=True)

        if date_str:
            filename = f"{date_str}.pmtiles"
            url = f"{PMTILES_URL_BASE}/{filename}"
            local_path = PMTILES_DIR / filename

            logger.info("Downloading PMTiles for date %s: %s", date_str, url)

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(local_path, 'wb') as f:
                downloaded = 0
                last_progress = -1
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_int = int(progress)
                            if progress_int > last_progress:
                                logger.info("Download progress: %d%%", progress_int)
                                last_progress = progress_int

            logger.info("Successfully downloaded PMTiles: %s (%d bytes)", filename, downloaded)
            return str(local_path)
        else:
            return download_latest_pmtiles()

    except Exception as e:
        logger.error("Error downloading PMTiles: %s", e)
        return None


def update_pmtiles() -> str | None:
    """
    Update to the latest PMTiles if it's newer than the current local file.

    Returns:
        str | None: Path to updated/new PMTiles file if successful, None if failed
    """
    try:
        current_file = get_pmtiles_path()
        current_date = None

        if current_file:
            filename = os.path.basename(current_file)
            if filename.endswith('.pmtiles'):
                date_part = filename[:-8]
                try:
                    current_date = datetime.strptime(date_part, "%Y%m%d").date()
                    logger.info("Current PMTiles date: %s", current_date)
                except ValueError:
                    logger.warning("Could not parse date from filename: %s", filename)

        latest_path = download_latest_pmtiles()

        if latest_path:
            if current_file and latest_path != current_file:
                logger.info("Updated to newer PMTiles: %s", os.path.basename(latest_path))
            return latest_path

        return current_file

    except Exception as e:
        logger.error("Error updating PMTiles: %s", e)
        return get_pmtiles_path()


def _build_tile_url(
    provider: str, zoom: int, xtile: int, ytile: int, layer: str | None = None
) -> str:
    provider_info = TILE_PROVIDERS.get(provider, TILE_PROVIDERS["pmtiles"])
    if provider_info.get("type") == "pmtiles":
        return f"pmtiles://{zoom}/{xtile}/{ytile}"
    return f"pmtiles://{zoom}/{xtile}/{ytile}"


def _fetch_single_tile(
    provider: str, layer: str | None, zoom: int, xtile: int, ytile: int
) -> bytes | None:
    """
    Fetch a single map tile from OpenStreetMap or PMTiles file.

    Args:
        provider (str): Tile provider
        layer (Optional[str]): Layer/file path for the tile (used for PMTiles)
        zoom (int): Zoom level of the tile
        xtile (int): X coordinate of the tile
        ytile (int): Y coordinate of the tile

    Returns:
        bytes | None: Raw image data if successful, None if failed
    """
    provider_info = TILE_PROVIDERS.get(provider, TILE_PROVIDERS["osm"])

    if provider_info.get("type") == "pmtiles":
        if not layer:
            layer = get_pmtiles_path()
            if not layer:
                logger.error("No PMTiles file available and download failed")
                return None

        try:
            with open(layer, 'rb') as f:
                def get_bytes(offset, length):
                    f.seek(offset)
                    return f.read(length)

                pmtiles_reader = reader.Reader(get_bytes)
                tile_data = pmtiles_reader.get(zoom, xtile, ytile)

                if tile_data:
                    logger.debug("Fetched PMTiles tile: %d/%d/%d from %s", zoom, xtile, ytile, layer)
                    return tile_data
                else:
                    logger.debug("Tile %d/%d/%d not found in PMTiles file %s", zoom, xtile, ytile, layer)
                    return None

        except FileNotFoundError:
            logger.warning("PMTiles file not found: %s", layer)
            return None
        except Exception as e:
            logger.error("Error reading PMTiles tile %d/%d/%d from %s: %s", zoom, xtile, ytile, layer, e)
            return None
    else:
        tile_url = _build_tile_url(provider, zoom, xtile, ytile, layer)
        logger.debug("Fetching single map tile: %s", tile_url)
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(tile_url, headers=headers, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "image" in content_type:
                return response.content
            else:
                logger.warning(
                    "Tile %d/%d/%d not an image. Content-Type: %s",
                    zoom,
                    xtile,
                    ytile,
                    content_type,
                )
                return None
        except requests.exceptions.RequestException as e:
            logger.warning("Error fetching tile %d/%d/%d: %s", zoom, xtile, ytile, e)
            return None
        except Exception as e:
            logger.error(
                "Unexpected error fetching tile %d/%d/%d: %s", zoom, xtile, ytile, e
            )
            return None


def get_openstreetmap_stitched_image(
    lat: float,
    lon: float,
    zoom: int = MAP_ZOOM_LEVEL,
    grid_size: int = MAP_GRID_SIZE,
    provider: str = "osm",
    layer: str | None = None,
) -> bytes | None:
    """
    Generate a stitched map image from OpenStreetMap tiles.

    Args:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        zoom (int, optional): Zoom level. Defaults to MAP_ZOOM_LEVEL.
        grid_size (int, optional): Size of the grid (must be odd). Defaults to MAP_GRID_SIZE.
        provider (str, optional): Tile provider. Defaults to 'osm'.
        layer (Optional[str], optional): Layer for the tile. Defaults to None.

    Returns:
        bytes | None: PNG image data if successful, None if failed
    """
    if grid_size < 1 or grid_size % 2 == 0:
        logger.error("Invalid grid_size: %d. Must be a positive odd number.", grid_size)
        grid_size = 3

    n = 2.0**zoom
    center_xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    center_ytile = int(n * (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2)

    logger.info(
        "Generating %dx%d map around tile %d/%d/%d",
        grid_size,
        grid_size,
        zoom,
        center_xtile,
        center_ytile,
    )

    radius = (grid_size - 1) // 2
    min_xtile = center_xtile - radius
    min_ytile = center_ytile - radius

    total_width = grid_size * TILE_SIZE
    total_height = grid_size * TILE_SIZE
    composite_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))

    tiles_fetched = 0
    for x_offset in range(grid_size):
        for y_offset in range(grid_size):
            xtile = min_xtile + x_offset
            ytile = min_ytile + y_offset

            tile_data = _fetch_single_tile(provider, layer, zoom, xtile, ytile)

            if tile_data:
                try:
                    tile_image = Image.open(io.BytesIO(tile_data)).convert("RGB")
                    paste_x = x_offset * TILE_SIZE
                    paste_y = y_offset * TILE_SIZE
                    composite_image.paste(tile_image, (paste_x, paste_y))
                    tiles_fetched += 1
                except Exception as e:
                    logger.warning(
                        "Failed to open or paste tile %d/%d/%d: %s",
                        zoom,
                        xtile,
                        ytile,
                        e,
                    )

    if tiles_fetched == 0:
        logger.error("Failed to fetch any tiles for the composite map.")
        return None

    logger.info(
        "Successfully stitched %d / %d tiles.", tiles_fetched, grid_size * grid_size
    )

    try:
        img_byte_arr = io.BytesIO()
        composite_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        logger.error("Failed to save composite image to buffer: %s", e)
        return None


def get_coords_from_city(city_name: str) -> tuple[float, float] | None:
    """
    Get latitude and longitude coordinates for a city name using geocoding.

    Args:
        city_name (str): Name of the city to geocode

    Returns:
        tuple[float, float] | None: (latitude, longitude) if successful, None if failed
    """
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            logger.info(
                "Geocoded '%s' to (%s, %s)",
                city_name,
                location.latitude,
                location.longitude,
            )
            return location.latitude, location.longitude
        else:
            logger.warning("Could not geocode city: %s", city_name)
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.error("Geocoding error for '%s': %s", city_name, e)
        return None
    except Exception as e:
        logger.error("Unexpected geocoding error: %s", e)
        return None


def get_coords_from_mgrs(mgrs_coord: str) -> tuple[float, float] | None:
    """
    Convert MGRS (Military Grid Reference System) coordinates to latitude and longitude.

    Args:
        mgrs_coord (str): MGRS coordinate string

    Returns:
        tuple[float, float] | None: (latitude, longitude) if successful, None if failed
    """
    try:
        mgrs_coord_upper = mgrs_coord.upper().replace(" ", "")
        match = re.match(
            r"^(\d{1,2}[C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d+)$", mgrs_coord_upper
        )

        if not match:
            logger.warning("Could not parse MGRS structure: '%s'", mgrs_coord)
            return None

        gzd = match.group(1)
        square_id = match.group(2)
        numerics = match.group(3)

        if len(numerics) % 2 != 0:
            logger.warning(
                "Invalid MGRS numerical part (odd length): '%s' in '%s'",
                numerics,
                mgrs_coord,
            )
            return None

        split_point = len(numerics) // 2
        easting = numerics[:split_point]
        northing = numerics[split_point:]

        formatted_mgrs = f"{gzd} {square_id} {easting} {northing}"
        logger.info("Attempting conversion with formatted MGRS: '%s'", formatted_mgrs)

        lat, lon = m.toLatLon(formatted_mgrs.encode("utf-8"))
        logger.info(
            "Converted MGRS '%s' (formatted as '%s') to (%s, %s)",
            mgrs_coord,
            formatted_mgrs,
            lat,
            lon,
        )
        return lat, lon
    except Exception as e:
        logger.warning(
            "Could not convert MGRS coordinate '%s' (tried format '%s'): %s",
            mgrs_coord,
            formatted_mgrs,
            e,
        )
        return None


def parse_lat_lon(coord_string: str) -> tuple[float, float] | None:
    """
    Parse a latitude/longitude coordinate string.

    Args:
        coord_string (str): String containing latitude and longitude (e.g., "45.123, -122.456")

    Returns:
        tuple[float, float] | None: (latitude, longitude) if valid, None if invalid
    """
    match = re.match(
        r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)\s*$", coord_string
    )
    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                logger.info("Parsed Lat/Lon: (%s, %s)", lat, lon)
                return lat, lon
            else:
                logger.warning("Invalid Lat/Lon range: (%s, %s)", lat, lon)
                return None
        except ValueError:
            return None
    return None


class MapBot:
    """
    A bot that provides map functionality through LXMF messaging.

    This bot allows users to request maps for locations specified in various formats:
    - MGRS (Military Grid Reference System) coordinates
    - Latitude/Longitude coordinates
    - City/Town names

    The bot responds with a stitched map image and an interactive OpenStreetMap link.

    Attributes:
        debug_mode (bool): Whether to enable detailed logging
        bot (LXMFBot): The underlying LXMF bot instance
    """

    def __init__(self, debug_mode=False):
        """
        Initialize the Map Bot.

        Args:
            debug_mode (bool, optional): Enable detailed logging. Defaults to False.
        """
        self.debug_mode = debug_mode
        self.bot = LXMFBot(
            name="Map Bot",
            command_prefix="",
            storage_type="json",
            storage_path="data/map_bot",
            announce=600,
            announce_immediately=False,
            first_message_enabled=False,
            signature_verification_enabled=True,
        )
        self.bot.command(
            name="map",
            description="Get offline map for MGRS, Lat/Lon, or City. Optional: zoom=N (1-19)",
        )(self.handle_map_command)
        # Help command to show usage instructions
        self.bot.command(name="help", description="Show usage instructions")(
            self.handle_help_command
        )
        self.bot.command(name="/help", description="Show usage instructions")(
            self.handle_help_command
        )
        self.bot.command(
            name="reverse", description="Reverse geocode Lat/Lon to address"
        )(self.handle_reverse_command)

    def send_message_with_icon(self, destination, message, title=None, **kwargs):
        """
        Send a message with the bot's icon appearance.

        Args:
            destination: Message destination
            message: Message content
            title: Optional message title
            **kwargs: Additional arguments passed to bot.send()
        """
        lxmf_fields = kwargs.pop('lxmf_fields', {})
        lxmf_fields.update(BOT_ICON_FIELD)

        return self.bot.send(
            destination=destination,
            message=message,
            title=title,
            lxmf_fields=lxmf_fields,
            **kwargs
        )

    def handle_map_command(self, ctx):
        """
        Handle the map command from users.

        Processes location queries in various formats and returns an offline map image.
        Supports optional zoom level specification.

        Args:
            ctx: The command context containing the message and sender information
        """
        zoom = MAP_ZOOM_LEVEL
        provider = "pmtiles"
        layer = None
        location_args = []
        zoom_override = None

        for arg in ctx.args:
            if arg.lower().startswith("zoom="):
                try:
                    val = int(arg.split("=", 1)[1])
                    if 1 <= val <= 19:
                        zoom_override = val
                        logger.info("User specified zoom: %s", zoom_override)
                    else:
                        ctx.reply(
                            f"Invalid zoom level specified: {val}. Must be between 1 and 19."
                        )
                        return
                except (ValueError, IndexError):
                    ctx.reply(
                        f"Invalid format for zoom argument: '{arg}'. Use 'zoom=N'."
                    )
                    return
            elif arg.lower().startswith("layer="):
                layer = arg.split("=", 1)[1]
            else:
                location_args.append(arg)

        if zoom_override is not None:
            zoom = zoom_override
        if not location_args:
            ctx.reply(
                "Please provide a location (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N]"
            )
            return

        location_query = " ".join(location_args)
        sender = ctx.sender

        logger.info(
            "Received map request for: '%s' (zoom=%s) from %s",
            location_query,
            zoom,
            sender,
        )
        query = location_query.strip()
        lat, lon = None, None
        map_source_type = "location"

        if re.match(
            r"^[0-9]{1,2}[C-HJ-NP-X]\s?[A-HJ-NP-Z]{2}", query.upper().split()[0]
        ):
            coords = get_coords_from_mgrs(query.upper())
            if coords:
                lat, lon = coords
                map_source_type = f"MGRS coordinate {query}"

        if lat is None and lon is None:
            coords = parse_lat_lon(query)
            if coords:
                lat, lon = coords
                map_source_type = f"Lat/Lon coordinate {query}"

        if lat is None and lon is None and map_source_type == "location":
            coords = get_coords_from_city(query)
            if coords:
                lat, lon = coords
                map_source_type = f"city/town '{query}'"

        if lat is not None and lon is not None:
            pmtiles_path = get_pmtiles_path()
            if not pmtiles_path:
                ctx.reply(
                    "No PMTiles data available. Please download PMTiles first using:\n"
                    "--download-latest (for latest)\n"
                    "--update (to update existing)\n"
                    "--download YYYYMMDD (for specific date)"
                )
                return

            image_data = get_openstreetmap_stitched_image(
                lat, lon, zoom, MAP_GRID_SIZE, provider, layer
            )
            osm_link = f"https://www.openstreetmap.org/#map={zoom}/{lat:.5f}/{lon:.5f}"

            if image_data:
                try:
                    attachment = Attachment(
                        type=AttachmentType.IMAGE,
                        name=f"map_{lat:.4f}_{lon:.4f}_z{zoom}.png",
                        data=image_data,
                        format="png",
                    )
                    message = (
                        f"Map for {map_source_type} (Zoom: {zoom}, {MAP_GRID_SIZE}x{MAP_GRID_SIZE} area):\n"
                        f"Interactive map (center): {osm_link}\n\n"
                        f"Stitched map preview:"
                    )
                    self.bot.send_with_attachment(
                        destination=sender,
                        message=message,
                        attachment=attachment,
                        title="Map Location",
                        lxmf_fields=BOT_ICON_FIELD,
                    )
                    logger.info("Sent map image and link for '%s' to %s", query, sender)

                except Exception as e:
                    logger.error("Failed to create or send map attachment: %s", e)
                    ctx.reply(
                        f"Sorry, I encountered an error trying to send the map image: {e}"
                    )
            else:
                ctx.reply(
                    f"Sorry, I couldn't retrieve the map image for {map_source_type} (zoom {zoom})."
                )
        else:
            ctx.reply(
                f"Sorry, I couldn't understand or find the location: '{location_query}'. Please check the format (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N]"
            )

    @staticmethod
    def handle_help_command(ctx):
        help_msg = (
            "Usage: map <location> [zoom=N]\n\n"
            "Get an offline stitched map image using PMTiles data:\n"
            "- MGRS coordinates (e.g., 38SMB12345678)\n"
            "- Latitude/Longitude (e.g., 40.7128,-74.0060)\n"
            "- City/Town name (e.g., New York City)\n\n"
            "Optional zoom levels: zoom=1 through zoom=19\n"
            "PMTiles data must be downloaded first using command-line flags.\n\n"
            "Commands:\n"
            "map <location> [zoom=N] - Get map for location\n"
            "reverse <lat,lon> - Reverse geocode coordinates to address\n\n"
            "Download PMTiles data first:\n"
            "--download-latest  - Download latest available\n"
            "--update           - Update if newer available\n"
            "--download YYYYMMDD - Download specific date\n\n"
            "Examples:\n"
            "map 38SMB12345678\n"
            "map 40.7128,-74.0060 zoom=12\n"
            'map "New York City"\n\n'
            "Type 'help' or '/help' to see this message again."
        )
        ctx.reply(help_msg)

    @staticmethod
    def handle_reverse_command(ctx):
        """
        Reverse geocode a latitude/longitude to a human-readable address.
        """
        if not ctx.args:
            ctx.reply("Usage: reverse <lat,lon>")
            return
        query = " ".join(ctx.args)
        coords = parse_lat_lon(query)
        if not coords:
            ctx.reply("Invalid Lat/Lon format. Use e.g. 40.7128,-74.0060")
            return
        lat, lon = coords
        try:
            location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, timeout=10)
            if location and location.address:
                ctx.reply(f"Reverse geocode for ({lat}, {lon}): {location.address}")
            else:
                ctx.reply(f"No address found for ({lat}, {lon})")
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            ctx.reply(f"Geocoding error: {e}")
        except Exception as e:
            ctx.reply(f"Unexpected error: {e}")


    def run(self):
        """
        Start the Map Bot.

        Initializes the bot and begins listening for commands.
        """
        logger.info("Starting Map Bot...")
        self.bot.run()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the LXMFy Map Bot.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed logging (currently mainly for startup/errors).",
    )

    pmtiles_group = parser.add_mutually_exclusive_group()
    pmtiles_group.add_argument(
        "--download-latest",
        action="store_true",
        help="Download the latest available PMTiles file from Protomaps.",
    )
    pmtiles_group.add_argument(
        "--update",
        action="store_true",
        help="Update to the latest PMTiles if newer than current local file.",
    )
    pmtiles_group.add_argument(
        "--download",
        metavar="DATE",
        help="Download PMTiles file for specific date (YYYYMMDD format).",
    )

    args = parser.parse_args()
    if args.download_latest:
        print("Downloading latest PMTiles...")
        path = download_pmtiles_by_date()
        if path:
            file_size = os.path.getsize(path) / (1024 * 1024)
            print(f"Successfully downloaded: {os.path.basename(path)} ({file_size:.1f} MB)")
        else:
            print("Failed to download PMTiles.")
        return

    elif args.update:
        print("Updating PMTiles...")
        path = update_pmtiles()
        if path:
            file_size = os.path.getsize(path) / (1024 * 1024)
            print(f"PMTiles ready: {os.path.basename(path)} ({file_size:.1f} MB)")
        else:
            print("Failed to update PMTiles.")
        return

    elif args.download:
        print(f"Downloading PMTiles for date {args.download}...")
        path = download_pmtiles_by_date(args.download)
        if path:
            file_size = os.path.getsize(path) / (1024 * 1024)
            print(f"Successfully downloaded: {os.path.basename(path)} ({file_size:.1f} MB)")
        else:
            print("Failed to download PMTiles for specified date.")
        return
    map_bot = MapBot(debug_mode=args.debug)
    map_bot.run()

    print(
        "<< Remember to install dependencies: pip install requests geopy python-mgrs >>"
    )


if __name__ == "__main__":
    main()
