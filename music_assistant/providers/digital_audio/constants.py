"""Constants for the S/PDIF Audio Out provider."""

from __future__ import annotations

import uuid

DOMAIN = "spdif_audio"

# Config entry keys
CONF_PA_SINK_NAME = "pa_sink_name"
CONF_ENCODING_FORMAT = "encoding_format"

# Encoding format values
ENCODING_AC3 = "ac3"
ENCODING_DTS = "dts"
ENCODING_EAC3 = "eac3"

# Bitrates (bps) for each format
ENCODING_BITRATES: dict[str, int] = {
    ENCODING_AC3: 640_000,
    ENCODING_DTS: 1_536_000,
    ENCODING_EAC3: 1_536_000,
}

# Maximum channels each format supports over S/PDIF IEC 61937
ENCODING_MAX_CHANNELS: dict[str, int] = {
    ENCODING_AC3: 6,  # 5.1
    ENCODING_DTS: 6,  # 5.1 (DTS 96/24 also 5.1)
    ENCODING_EAC3: 8,  # 7.1 via HDMI
}

# UUID namespace for stable player IDs
UUID_NAMESPACE = uuid.UUID("b3e2f1a0-4c5d-4e6f-8a9b-0c1d2e3f4a5b")
