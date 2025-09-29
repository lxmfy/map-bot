# LXMFy Map Bot

A privacy-focused LXMFy bot that generates offline map images using PMTiles data. No internet connection required for map generation - works completely offline once PMTiles data is downloaded.

## Features

- **Offline Operation**: Uses PMTiles for completely offline map generation
- **Multiple Coordinate Systems**: Supports MGRS, latitude/longitude, and city names
- **Privacy-Focused**: No data sent to external mapping services during operation except OSM nominatim for geocoding

## Installation

```bash
pipx install git+https://github.com/lxmfy/map-bot
```

## Setup & Data Download

**Important**: You must download PMTiles data before using the bot.

### Download Latest PMTiles
```bash
map-bot --download-latest
```

### Download Specific Date
```bash
map-bot --download 20250929
```

### Update Existing Data
```bash
map-bot --update
```

PMTiles data is stored in `data/pmtiles/` and sourced from [Protomaps](https://maps.protomaps.com).

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
# Download latest map data first
map-bot --download-latest

# Start the bot
map-bot

# Then use in LXMF chat:
map 38SMB12345678
map 40.7128,-74.0060 zoom=12
map "New York City"
reverse 40.7128,-74.0060
```
