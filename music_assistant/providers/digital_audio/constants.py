"""Constants for the Digital Audio (direct PA sink) provider."""

from __future__ import annotations

import uuid

# Provider-level configuration key — the target PulseAudio sink name.
CONF_PA_SINK_NAME = "pa_sink_name"

# Player defaults
DEFAULT_PLAYER_VOLUME = 80
DEFAULT_HARDWARE_VOLUME_CEILING = 95

# Cache category for volume/mute persistence across restarts.
CACHE_CATEGORY_PREV_STATE = "digital_audio_prev_state"

# Stable UUID namespace for deterministic player ID generation from sink name.
DEVICE_UUID_NAMESPACE = uuid.UUID("8a4c2f1e-5b6d-7e8f-9a0b-1c2d3e4f5a6b")
