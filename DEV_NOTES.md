# Dev Notes

## Status
This repo is currently in pre-alpha.

## Session Summary
- Expanded the app from a simple screen keyer into a mixed source tool with monitor and NDI source modes.
- Added target overlay geometry controls.
- Added separate overlay black and full-screen black dissolves.
- Added scene save/load/delete support.
- Added a Python 3.13 launcher for the NDI-capable runtime.

## Important Runtime Notes
- Python 3.14 is installed and can compile the app, but the NDI package stack was not reliable there.
- Python 3.13 has `cyndilib`, `opencv-python`, `mss`, and `PySide6` installed.
- Use `run_screen_keyer.bat` for the best chance of NDI working.

## What To Test Next
- live NDI source selection on a real network sender
- overlay positioning on different monitor layouts
- blackout transitions during active playback
- scene recall in repeated operator workflows
- NDI output acceptance by receiving apps
