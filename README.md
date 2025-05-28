# LXMFy Map Bot

A simple LXMFy bot that generates and sends stitched OpenStreetMap images based on MGRS, latitude/longitude, or city names.

## Installation

```bash
pip install requests geopy python-mgrs pillow lxmfy
```

## Usage

```bash
python map_bot/bot.py
```

Once running, send the `map` command to the bot:

```
map <location> [zoom=N]
```

Examples:

- `map 38SMB12345678`
- `map 40.7128,-74.0060 zoom=12`
- `map "New York City"`

The bot will reply with a stitched map image preview and an interactive OpenStreetMap link.

## Development

```bash
poetry install
poetry run map-bot
```
