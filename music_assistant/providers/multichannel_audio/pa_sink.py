"""PulseAudio helpers for multichannel sink enumeration and streaming."""

from __future__ import annotations

from typing import Any

from .constants import (
    MULTICHANNEL_LAYOUT_51,
    MULTICHANNEL_LAYOUT_71,
    PA_CHANNEL_MAP_51,
    PA_CHANNEL_MAP_71,
)

_MIN_MULTICHANNEL_CHANNELS = 6


def enumerate_surround_sinks() -> list[dict[str, Any]]:
    """
    Filter the full PA sink list for sinks with 6 or more channels.

    Reuses enumerate_pa_sinks() from pa_simple so we get the same
    pactl-based native sample_rate/bit_depth accuracy.

    :returns: List of dicts with keys: name, pa_sink_name, channels,
              layout, channel_map, sample_rate, bit_depth.
    """
    from music_assistant.providers.local_audio.pa_simple import enumerate_pa_sinks  # noqa: PLC0415

    results: list[dict[str, Any]] = []
    for sink in enumerate_pa_sinks():
        ch_count: int = sink.get("max_output_channels", 0)
        if ch_count < _MIN_MULTICHANNEL_CHANNELS:
            continue

        layout = MULTICHANNEL_LAYOUT_71 if ch_count >= 8 else MULTICHANNEL_LAYOUT_51
        channel_map = PA_CHANNEL_MAP_71 if ch_count >= 8 else PA_CHANNEL_MAP_51

        results.append(
            {
                "name": sink["name"],
                "pa_sink_name": sink["pa_sink_name"],
                "channels": ch_count,
                "layout": layout,
                "channel_map": channel_map,
                "sample_rate": sink["sample_rate"],
                "bit_depth": sink["bit_depth"],
                "is_remap": sink.get("is_remap", False),
            }
        )
    return results
