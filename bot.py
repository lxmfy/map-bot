import io
import logging
import math  # Moved import to top level
import re
from typing import Optional

import requests
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

# Use imports based on the lxmfy examples
from lxmfy import Attachment, AttachmentType, LXMFBot, command
from mgrs import MGRS
from PIL import Image  # Added Pillow import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Map settings
MAP_ZOOM_LEVEL = 14
MAP_GRID_SIZE = 3 # How many tiles wide/high (e.g., 3x3)
TILE_SIZE = 256 # Standard OSM tile size
USER_AGENT = 'lxmfy_map_bot/1.0 (requests)' # Define User-Agent once

# --- Geocoding and MGRS ---
geolocator = Nominatim(user_agent="lxmfy_map_bot/1.0")
m = MGRS()

# --- Map Generation Functions ---

def _fetch_single_tile(zoom: int, xtile: int, ytile: int) -> bytes | None:
    """Fetches a single map tile from OSM."""
    tile_url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
    logger.debug(f"Fetching single map tile: {tile_url}")
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(tile_url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '').lower()
        if 'image' in content_type:
             # Return raw image bytes
             return response.content
        else:
            logger.warning(f"Tile {zoom}/{xtile}/{ytile} not an image. Content-Type: {content_type}")
            return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching tile {zoom}/{xtile}/{ytile}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching tile {zoom}/{xtile}/{ytile}: {e}")
        return None

def get_openstreetmap_stitched_image(lat: float, lon: float, zoom: int = MAP_ZOOM_LEVEL, grid_size: int = MAP_GRID_SIZE) -> bytes | None:
    """Fetches multiple map tiles around lat/lon and stitches them together."""
    if grid_size < 1 or grid_size % 2 == 0:
        logger.error(f"Invalid grid_size: {grid_size}. Must be a positive odd number.")
        grid_size = 3 # Fallback to default

    n = 2.0 ** zoom
    center_xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    center_ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    logger.info(f"Generating {grid_size}x{grid_size} map around tile {zoom}/{center_xtile}/{center_ytile}")

    # Calculate tile range
    radius = (grid_size - 1) // 2
    min_xtile = center_xtile - radius
    min_ytile = center_ytile - radius

    # Create blank canvas
    total_width = grid_size * TILE_SIZE
    total_height = grid_size * TILE_SIZE
    composite_image = Image.new('RGB', (total_width, total_height), (255, 255, 255)) # White background

    # Fetch and paste tiles
    tiles_fetched = 0
    for x_offset in range(grid_size):
        for y_offset in range(grid_size):
            xtile = min_xtile + x_offset
            ytile = min_ytile + y_offset

            tile_data = _fetch_single_tile(zoom, xtile, ytile)

            if tile_data:
                try:
                    tile_image = Image.open(io.BytesIO(tile_data)).convert('RGB')
                    paste_x = x_offset * TILE_SIZE
                    paste_y = y_offset * TILE_SIZE
                    composite_image.paste(tile_image, (paste_x, paste_y))
                    tiles_fetched += 1
                except Exception as e:
                    logger.warning(f"Failed to open or paste tile {zoom}/{xtile}/{ytile}: {e}")
            else:
                # Optionally draw a placeholder for missing tiles
                pass

    if tiles_fetched == 0:
        logger.error("Failed to fetch any tiles for the composite map.")
        return None

    logger.info(f"Successfully stitched {tiles_fetched} / {grid_size*grid_size} tiles.")

    # Save composite image to in-memory bytes buffer
    try:
        img_byte_arr = io.BytesIO()
        composite_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        logger.error(f"Failed to save composite image to buffer: {e}")
        return None

def get_coords_from_city(city_name: str) -> tuple[float, float] | None:
    """Geocode city name to latitude and longitude."""
    try:
        location = geolocator.geocode(city_name, timeout=10)
        if location:
            logger.info(f"Geocoded '{city_name}' to ({location.latitude}, {location.longitude})")
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
    """Convert MGRS coordinate string to latitude and longitude."""
    try:
        # Regex to capture GZD, Square ID, and Numerics, ignoring spaces
        mgrs_coord_upper = mgrs_coord.upper().replace(' ', '')
        match = re.match(r'^(\d{1,2}[C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d+)$', mgrs_coord_upper)

        if not match:
            logger.warning(f"Could not parse MGRS structure: '{mgrs_coord}'")
            return None

        gzd = match.group(1)
        square_id = match.group(2)
        numerics = match.group(3)

        # Ensure numerics part has an even length for splitting
        if len(numerics) % 2 != 0:
            logger.warning(f"Invalid MGRS numerical part (odd length): '{numerics}' in '{mgrs_coord}'")
            return None

        # Split numerics into Easting and Northing
        split_point = len(numerics) // 2
        easting = numerics[:split_point]
        northing = numerics[split_point:]

        # Format for the MGRS library (GZD<space>SquareID<space>Easting<space>Northing)
        formatted_mgrs = f"{gzd} {square_id} {easting} {northing}"
        logger.info(f"Attempting conversion with formatted MGRS: '{formatted_mgrs}'")

        lat, lon = m.toLatLon(formatted_mgrs.encode('utf-8'))
        logger.info(f"Converted MGRS '{mgrs_coord}' (formatted as '{formatted_mgrs}') to ({lat}, {lon})")
        return lat, lon
    except Exception as e:
        logger.warning(f"Could not convert MGRS coordinate '{mgrs_coord}' (tried format '{formatted_mgrs}'): {e}")
        return None

def parse_lat_lon(coord_string: str) -> tuple[float, float] | None:
    """Parse a string potentially containing latitude and longitude."""
    match = re.match(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)\s*$", coord_string)
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

# --- LXMFy Bot Class ---

class MapBot:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.bot = LXMFBot(
            name="Map Bot",
            command_prefix="", # Example prefix, adjust if needed
            storage_type="json", # Or memory, etc.
            storage_path="data/map_bot", # Example path
            announce=600, # Announce interval in seconds
            announce_immediately=False,
            first_message_enabled=False # Disable default welcome
        )
        # Register command using the decorator within the class
        self.bot.command(name="map", description="Get map for MGRS, Lat/Lon, or City. Optional: zoom=N (1-19)")(self.handle_map_command)

    # Reverted to synchronous command handler
    def handle_map_command(self, ctx):
        """
        Handles the !map command to provide a map image based on user query.
        Expected query formats:
        - MGRS: e.g., "31UDQ1234567890" or "31U DQ 12345 67890"
        - Lat/Lon: e.g., "40.7128, -74.0060" or "40.7128 -74.0060"
        - City/Town: e.g., "New York City"
        Optional argument: zoom=N (where N is 1-19)
        """
        zoom = MAP_ZOOM_LEVEL # Default zoom
        location_args = []
        zoom_override = None

        # Parse args for location and optional zoom
        for arg in ctx.args:
            if arg.lower().startswith("zoom="):
                try:
                    val = int(arg.split('=', 1)[1])
                    if 1 <= val <= 19: # OSM zoom range
                        zoom_override = val
                        logger.info(f"User specified zoom: {zoom_override}")
                    else:
                        ctx.reply(f"Invalid zoom level specified: {val}. Must be between 1 and 19.")
                        return
                except (ValueError, IndexError):
                    ctx.reply(f"Invalid format for zoom argument: '{arg}'. Use 'zoom=N'.")
                    return
            else:
                location_args.append(arg)

        if zoom_override is not None:
            zoom = zoom_override

        # Get arguments from context
        if not location_args:
            ctx.reply("Please provide a location (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N]")
            return

        location_query = " ".join(location_args)
        sender = ctx.sender # Get sender from context

        logger.info(f"Received map request for: '{location_query}' (zoom={zoom}) from {sender}")
        query = location_query.strip()
        lat, lon = None, None
        map_source_type = "location" # Default description

        # 1. Try parsing as MGRS
        # Simple check: starts like MGRS grid zone designator (GZD) + 100km square ID
        if re.match(r"^[0-9]{1,2}[C-HJ-NP-X]\s?[A-HJ-NP-Z]{2}", query.upper().split()[0]):
             coords = get_coords_from_mgrs(query.upper())
             if coords:
                 lat, lon = coords
                 map_source_type = f"MGRS coordinate {query}"

        # 2. Try parsing as Lat/Lon (if not already found)
        if lat is None and lon is None:
            coords = parse_lat_lon(query)
            if coords:
                lat, lon = coords
                map_source_type = f"Lat/Lon coordinate {query}"

        # 3. Try geocoding as City/Town (if not already found)
        if lat is None and lon is None and map_source_type == "location":
            # Avoid trying to geocode if it looked like MGRS or Lat/Lon but failed parsing
            coords = get_coords_from_city(query)
            if coords:
                lat, lon = coords
                map_source_type = f"city/town '{query}'"

        # --- Generate and Send Map ---
        if lat is not None and lon is not None:
            # Use the new stitching function
            image_data = get_openstreetmap_stitched_image(lat, lon, zoom)

            # Generate interactive map link (using center lat/lon)
            osm_link = f"https://www.openstreetmap.org/#map={zoom}/{lat:.5f}/{lon:.5f}"

            if image_data:
                try:
                    attachment = Attachment(
                        type=AttachmentType.IMAGE,
                        # Updated filename slightly
                        name=f"map_{lat:.4f}_{lon:.4f}_z{zoom}.png",
                        data=image_data,
                        format="png"
                    )
                    # Use send_with_attachment based on meme_bot example
                    message = (
                        f"Map for {map_source_type} (Zoom: {zoom}, {MAP_GRID_SIZE}x{MAP_GRID_SIZE} area):\n"
                        f"Interactive map (center): {osm_link}\n\n"
                        f"Stitched map preview:"
                    )
                    self.bot.send_with_attachment(
                        destination=sender,
                        message=message,
                        attachment=attachment,
                        title="Map Location" # Optional title
                    )
                    logger.info(f"Sent map image and link for '{query}' to {sender}")

                except Exception as e:
                    logger.error(f"Failed to create or send map attachment: {e}")
                    ctx.reply(f"Sorry, I encountered an error trying to send the map image: {e}")
            else:
                ctx.reply(f"Sorry, I couldn't retrieve the map image for {map_source_type} (zoom {zoom}).")
        else:
            ctx.reply(f"Sorry, I couldn't understand or find the location: '{location_query}'. Please check the format (MGRS, Lat/Lon, or City/Town). Usage: map <location> [zoom=N]")

    def run(self):
        """Starts the bot."""
        logger.info("Starting Map Bot...")
        self.bot.run()

# --- Main Execution Block ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the LXMFy Map Bot.')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed logging (currently mainly for startup/errors).'
    )
    args = parser.parse_args()

    # Instantiate and run the bot
    map_bot = MapBot(debug_mode=args.debug)
    map_bot.run()

    print("<< Remember to install dependencies: pip install requests geopy python-mgrs >>")
    # Example Usage (for testing functions directly without full bot)
    # print(get_coords_from_mgrs("31UDQ 12345 67890"))
    # print(parse_lat_lon("40.7128, -74.0060"))
    # print(get_coords_from_city("Paris, France"))
    # img_data = get_openstreetmap_image(48.8566, 2.3522) # Paris
    # if img_data:
    #     with open("test_map.png", "wb") as f:
    #         f.write(img_data)
    #     print("Saved test_map.png")