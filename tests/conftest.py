"""Test settings shared by every test that runs the app."""
from pathlib import Path

from ha_voice_agent import guard, warm

# The app's tests use fake upstreams; a background warm-up request would hit them too.
warm.WARM = False
guard.ACTION_TOOLS_FILE = str(Path(__file__).parent.parent / "config" / "action-tools.json")
