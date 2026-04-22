"""Constants for the Multichannel Audio player provider."""

from __future__ import annotations

import uuid

# Distinct UUID namespace from local_audio to avoid player ID collisions
DEVICE_UUID_NAMESPACE = uuid.UUID("3413d5ed-7099-426e-9367-da3a6b057ec5")

# Category for caching previous player state (volume/mute)
CACHE_CATEGORY_PREV_STATE = 1

# Multichannel layout identifiers
MULTICHANNEL_LAYOUT_51 = "5.1"
MULTICHANNEL_LAYOUT_71 = "7.1"

# Channel counts per layout
MULTICHANNEL_CHANNELS: dict[str, int] = {
    MULTICHANNEL_LAYOUT_51: 6,
    MULTICHANNEL_LAYOUT_71: 8,
}

# PulseAudio channel map strings — must match the PA sink channel_map
PA_CHANNEL_MAP_51 = "front-left,front-right,front-center,lfe,rear-left,rear-right"
PA_CHANNEL_MAP_71 = (
    "front-left,front-right,front-center,lfe,rear-left,rear-right,side-left,side-right"
)
PA_CHANNEL_MAPS: dict[str, str] = {
    MULTICHANNEL_LAYOUT_51: PA_CHANNEL_MAP_51,
    MULTICHANNEL_LAYOUT_71: PA_CHANNEL_MAP_71,
}

# Volume control mode constants — match local_audio naming
VOLUME_CONTROL_HARDWARE = "hardware"
VOLUME_CONTROL_SOFTWARE = "software"
VOLUME_CONTROL_DISABLED = "disabled"

# Config entry keys
CONF_MULTICHANNEL_LAYOUT = "multichannel_layout"
CONF_VOLUME_CONTROL = "volume_control"

# Defaults
DEFAULT_PLAYER_VOLUME = 25
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BIT_DEPTH = 24
DEFAULT_HARDWARE_VOLUME_CEILING = 85
