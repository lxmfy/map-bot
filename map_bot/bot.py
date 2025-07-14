import io
import logging
import math
import re

import requests
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from lxmfy import Attachment, AttachmentType, LXMFBot
from mgrs import MGRS
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAP_ZOOM_LEVEL = 14
MAP_GRID_SIZE = 3
TILE_SIZE = 256
USER_AGENT = "lxmfy_map_bot/1.0 (requests)"

# Tile providers configuration
TILE_PROVIDERS = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
    },
    "openfree": {
        "url": "https://tile.openfreemap.org/hike/{z}/{x}/{y}.png",
        "attribution": "© OpenFreeMap",
    },
}

geolocator = Nominatim(user_agent="lxmfy_map_bot/1.0")
m = MGRS()


def _build_tile_url(
    provider: str, zoom: int, xtile: int, ytile: int, layer: str | None = None
) -> str:
    provider_info = TILE_PROVIDERS.get(provider, TILE_PROVIDERS["osm"])
    url_template = provider_info["url"]
    return url_template.format(z=zoom, x=xtile, y=ytile)


def _fetch_single_tile(
    provider: str, layer: str | None, zoom: int, xtile: int, ytile: int
) -> bytes | None:
    """
    Fetch a single map tile from OpenStreetMap.

    Args:
        provider (str): Tile provider
        layer (Optional[str]): Layer for the tile
        zoom (int): Zoom level of the tile
        xtile (int): X coordinate of the tile
        ytile (int): Y coordinate of the tile

    Returns:
        bytes | None: Raw image data if successful, None if failed
    """
    tile_url = _build_tile_url(provider, zoom, xtile, ytile, layer)
    logger.debug(f"Fetching single map tile: {tile_url}")
    try:
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(tile_url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "image" in content_type:
            return response.content
        else:
            logger.warning(
                f"Tile {zoom}/{xtile}/{ytile} not an image. Content-Type: {content_type}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching tile {zoom}/{xtile}/{ytile}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching tile {zoom}/{xtile}/{ytile}: {e}")
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
        logger.error(f"Invalid grid_size: {grid_size}. Must be a positive odd number.")
        grid_size = 3

    n = 2.0**zoom
    center_xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    center_ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    logger.info(
        f"Generating {grid_size}x{grid_size} map around tile {zoom}/{center_xtile}/{center_ytile}"
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
                        f"Failed to open or paste tile {zoom}/{xtile}/{ytile}: {e}"
                    )

    if tiles_fetched == 0:
        logger.error("Failed to fetch any tiles for the composite map.")
        return None

    logger.info(
        f"Successfully stitched {tiles_fetched} / {grid_size * grid_size} tiles."
    )

    try:
        img_byte_arr = io.BytesIO()
        composite_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        logger.error(f"Failed to save composite image to buffer: {e}")
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
                f"Geocoded '{city_name}' to ({location.latitude}, {location.longitude})"
            )
            return location.latitude, location.longitude
        else:
            logger.warning(f"Could not geocode city: {city_name}")
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        logger.error(f"Geocoding error for '{city_name}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected geocoding error: {e}")
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
            logger.warning(f"Could not parse MGRS structure: '{mgrs_coord}'")
            return None

        gzd = match.group(1)
        square_id = match.group(2)
        numerics = match.group(3)

        if len(numerics) % 2 != 0:
            logger.warning(
                f"Invalid MGRS numerical part (odd length): '{numerics}' in '{mgrs_coord}'"
            )
            return None

        split_point = len(numerics) // 2
        easting = numerics[:split_point]
        northing = numerics[split_point:]

        formatted_mgrs = f"{gzd} {square_id} {easting} {northing}"
        logger.info(f"Attempting conversion with formatted MGRS: '{formatted_mgrs}'")

        lat, lon = m.toLatLon(formatted_mgrs.encode("utf-8"))
        logger.info(
            f"Converted MGRS '{mgrs_coord}' (formatted as '{formatted_mgrs}') to ({lat}, {lon})"
        )
        return lat, lon
    except Exception as e:
        logger.warning(
            f"Could not convert MGRS coordinate '{mgrs_coord}' (tried format '{formatted_mgrs}'): {e}"
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
                logger.info(f"Parsed Lat/Lon: ({lat}, {lon})")
                return lat, lon
            else:
                logger.warning(f"Invalid Lat/Lon range: ({lat}, {lon})")
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
        )
        self.bot.command(
            name="map",
            description="Get map for MGRS, Lat/Lon, or City. Optional: zoom=N (1-19)",
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

    def handle_map_command(self, ctx):
        """
        Handle the map command from users.

        Processes location queries in various formats and returns a map image and link.
        Supports optional zoom level specification.

        Args:
            ctx: The command context containing the message and sender information
        """
        zoom = MAP_ZOOM_LEVEL
        provider = "osm"
        layer = None
        location_args = []
        zoom_override = None
        provider_override = None

        for arg in ctx.args:
            if arg.lower().startswith("zoom="):
                try:
                    val = int(arg.split("=", 1)[1])
                    if 1 <= val <= 19:
                        zoom_override = val
                        logger.info(f"User specified zoom: {zoom_override}")
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
            elif arg.lower().startswith("provider="):
                val = arg.split("=", 1)[1].lower()
                if val in TILE_PROVIDERS:
                    provider_override = val
                else:
                    ctx.reply(
                        f"Invalid provider specified: {val}. Available: {', '.join(TILE_PROVIDERS.keys())}"
                    )
                    return
            elif arg.lower().startswith("layer="):
                layer = arg.split("=", 1)[1].lower()
            else:
                location_args.append(arg)

        if zoom_override is not None:
            zoom = zoom_override
        if provider_override:
            provider = provider_override
        if not location_args:
            ctx.reply(
                "Please provide a location (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N] [provider=<provider>] [layer=<layer>]"
            )
            return

        location_query = " ".join(location_args)
        sender = ctx.sender

        logger.info(
            f"Received map request for: '{location_query}' (zoom={zoom}) from {sender}"
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
                    )
                    logger.info(f"Sent map image and link for '{query}' to {sender}")

                except Exception as e:
                    logger.error(f"Failed to create or send map attachment: {e}")
                    ctx.reply(
                        f"Sorry, I encountered an error trying to send the map image: {e}"
                    )
            else:
                ctx.reply(
                    f"Sorry, I couldn't retrieve the map image for {map_source_type} (zoom {zoom})."
                )
        else:
            ctx.reply(
                f"Sorry, I couldn't understand or find the location: '{location_query}'. Please check the format (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N] [provider=<provider>] [layer=<layer>]"
            )

    def handle_help_command(self, ctx):
        """
        Provide usage instructions for the map command.
        """
        help_msg = (
            "Usage: map <location> [zoom=N] [provider=<provider>] [layer=<layer>]\n\n"
            "Get a stitched OpenStreetMap image based on:\n"
            "- MGRS coordinates (e.g., 38SMB12345678)\n"
            "- Latitude/Longitude (e.g., 40.7128,-74.0060)\n"
            "- City/Town name (e.g., New York City)\n\n"
            "Optional zoom levels: zoom=1 through zoom=19\n"
            "Optional tile providers: provider=osm,stamen,openfree\n"
            "Optional layers (for stamen): layer=terrain,toner,watercolor\n"
            "Examples:\n"
            "map 38SMB12345678\n"
            "map 40.7128,-74.0060 zoom=12\n"
            'map "New York City"\n\n'
            "Type 'help' or '/help' to see this message again."
        )
        ctx.reply(help_msg)

    def handle_reverse_command(self, ctx):
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
    args = parser.parse_args()

    map_bot = MapBot(debug_mode=args.debug)
    map_bot.run()

    print(
        "<< Remember to install dependencies: pip install requests geopy python-mgrs >>"
    )


if __name__ == "__main__":
    main()
