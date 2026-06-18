"""Constants for the S/PDIF Audio (AC3 IEC61937 passthrough) provider."""

from __future__ import annotations

import uuid

# Provider-level configuration key — the target PulseAudio sink name
# (the optical/coax S/PDIF output, e.g. an HDA card's iec958 profile sink).
CONF_PA_SINK_NAME = "pa_sink_name"

# Optional AC3 encode bitrate override. 640k is the practical ceiling most
# receivers/soundbars support for 5.1 Dolby Digital (matches DVD/Blu-ray norms).
CONF_AC3_BITRATE = "ac3_bitrate"
DEFAULT_AC3_BITRATE = "640k"

# S/PDIF passthrough is fundamentally a 2-channel digital carrier — the AC3
# bitstream is "disguised" as 16-bit stereo PCM via IEC 61937 framing. 48kHz
# is the universal standard rate; AC3 does not support 96kHz/24-32bit.
SPDIF_CARRIER_CHANNELS = 2
SPDIF_SAMPLE_RATE = 48000
SPDIF_BIT_DEPTH = 16

# Maximum real content channels this player declares to MA (classic Dolby
# Digital 5.1). Used to anchor MA's flow encoder via supported_sample_rates,
# independent of the PA sink's own native 2ch carrier profile.
SPDIF_MAX_CONTENT_CHANNELS = 6

# Below this many real source channels, content goes out as plain PCM
# (no AC3 encoding needed — stereo travels over optical natively).
SPDIF_PASSTHROUGH_THRESHOLD_CHANNELS = 2

DEFAULT_PLAYER_VOLUME = 100  # Locked — see player.py docstring; no VOLUME_SET feature.

CACHE_CATEGORY_PREV_STATE = "spdif_audio_prev_state"

DEVICE_UUID_NAMESPACE = uuid.UUID("3f7a1c9d-2e4b-4a6f-8d1c-9b5e7f0a2c3d")
