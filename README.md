# Screen Keyer FTB

Pre-alpha desktop keyer and blackout tool for live production use.

## Description
Screen Keyer FTB keys a chosen source over a chosen target display, with fast operator controls for overlay on/off, overlay black, and full target screen black. It can use either a real monitor capture source or an NDI source when the Python 3.13 + `cyndilib` runtime is available.

This is currently a pre-alpha build. It is functional in parts, but still needs workflow polish, deeper testing, and reliability work before live-critical use.

## Current Features
- Monitor source capture
- NDI source discovery and NDI input path when available
- Target monitor selection
- Overlay position, size, opacity, and aspect controls
- Luma and chroma key modes
- Overlay black dissolve
- Full target screen black dissolve
- Scene save, load, and delete
- Optional NDI output path

## Current State
- Kinda working, but still in active development
- Good enough for iterative testing
- Not yet confirmed ready for a show-critical environment

## Launch
Preferred launcher:
- `run_screen_keyer.bat`

That launcher prefers the local Python 3.13 install because the NDI support currently lives there.

## Notes
Known work still needed:
- more live testing with real NDI sources
- better operator UI polish
- stronger fallback behavior when NDI drops or reconnects
- validation of NDI output in downstream apps
- packaging and simpler setup
