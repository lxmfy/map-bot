# LXMFy Map Bot

A privacy-focused LXMFy bot that generates offline map images using PMTiles data. No internet connection required for map generation - works completely offline once PMTiles data is downloaded.

## Features

- **OpenStreetMap Tiles**: Uses OpenStreetMap tiles for map generation - for now
- **Multiple Coordinate Systems**: Supports MGRS, latitude/longitude, and city names

Goal is to move to a offline solution in the future.

## Installation

```bash
pipx install git+https://github.com/lxmfy/map-bot
```

## Usage

### Start the Bot
```bash
map-bot
```

### Map Commands

Once running, send the `map` command to the bot:

```
map <location> [zoom=N]
```

#### Supported Location Formats:
- **MGRS coordinates**: `map 38SMB12345678`
- **Latitude/Longitude**: `map 40.7128,-74.0060`
- **City/Place names**: `map "New York City"`

#### Optional Parameters:
- `zoom=N`: Zoom level (1-19, default: 14)

### Other Commands
- `reverse <lat,lon>`: Reverse geocode coordinates to address
- `help`: Show usage instructions

## Examples

```bash
# Start the bot
map-bot

# Then use in LXMF chat:
map 38SMB12345678
map 40.7128,-74.0060 zoom=12
map "New York City"
reverse 40.7128,-74.0060
```
