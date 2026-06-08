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

# Volume control mode constants (internal use only — not exposed in config)
VOLUME_CONTROL_HARDWARE = "hardware"
VOLUME_CONTROL_SOFTWARE = "software"

# Channel map preset identifiers
CHANNEL_MAP_FLAC = "flac"    # FL FR FC LFE RL RR [SL SR] — FLAC/PCM standard, MA default
CHANNEL_MAP_DVD = "dvd"      # FL FR LFE FC RL RR [SL SR] — DVD/AC3 order (FC↔LFE swapped)
CHANNEL_MAP_CUSTOM = "custom"  # User-specified index pairs

# Per-layout channel pair index maps for each preset.
# Each entry maps PA sink name suffix -> (left_ch_index, right_ch_index)
# in the interleaved PCM stream delivered by MA.
_PAIRS_FLAC_71 = {
    "front_stereo": (0, 1),   # FL, FR
    "center_sub":   (2, 3),   # FC, LFE
    "rear_stereo":  (4, 5),   # RL, RR
    "side_stereo":  (6, 7),   # SL, SR
}
_PAIRS_FLAC_51 = {
    "front_stereo": (0, 1),   # FL, FR
    "center_sub":   (2, 3),   # FC, LFE
    "rear_stereo":  (4, 5),   # RL, RR
}
_PAIRS_DVD_71 = {
    "front_stereo": (0, 1),   # FL, FR
    "center_sub":   (3, 2),   # FC(idx3), LFE(idx2) — DVD swaps FC↔LFE vs FLAC
    "rear_stereo":  (4, 5),   # RL, RR
    "side_stereo":  (6, 7),   # SL, SR
}
_PAIRS_DVD_51 = {
    "front_stereo": (0, 1),   # FL, FR
    "center_sub":   (3, 2),   # FC(idx3), LFE(idx2)
    "rear_stereo":  (4, 5),   # RL, RR
}

PAIR_INDICES_BY_MAP: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
    CHANNEL_MAP_FLAC: {
        MULTICHANNEL_LAYOUT_71: _PAIRS_FLAC_71,
        MULTICHANNEL_LAYOUT_51: _PAIRS_FLAC_51,
    },
    CHANNEL_MAP_DVD: {
        MULTICHANNEL_LAYOUT_71: _PAIRS_DVD_71,
        MULTICHANNEL_LAYOUT_51: _PAIRS_DVD_51,
    },
}

# Config entry keys
CONF_PA_SINK_NAME = "pa_sink_name"
CONF_CARD_NAME = "card_name"
CONF_MULTICHANNEL_LAYOUT = "multichannel_layout"
CONF_CHANNEL_MAP = "channel_map"
CONF_CUSTOM_CHANNEL_MAP = "custom_channel_map"

# Defaults
DEFAULT_PLAYER_VOLUME = 25
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BIT_DEPTH = 24
DEFAULT_HARDWARE_VOLUME_CEILING = 85