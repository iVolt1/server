"""Multichannel Audio player provider for Music Assistant."""

from __future__ import annotations

import subprocess
import json
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.mass import MusicAssistant

from .constants import (
    CHANNEL_MAP_CUSTOM,
    CHANNEL_MAP_DVD,
    CHANNEL_MAP_FLAC,
    CONF_CHANNEL_MAP,
    CONF_CUSTOM_CHANNEL_MAP,
    CONF_MULTICHANNEL_LAYOUT,
    CONF_PA_SINK_NAME,
    MULTICHANNEL_LAYOUT_51,
    MULTICHANNEL_LAYOUT_71,
)
from .provider import MultiChannelAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


def _get_pa_sink_options() -> list[ConfigValueOption]:
    """Return available PA sinks as config options by running pactl."""
    options: list[ConfigValueOption] = []
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for sink in json.loads(result.stdout):
                name: str = sink.get("name", "")
                desc: str = sink.get("description", name)
                if name:
                    options.append(ConfigValueOption(title=desc, value=name))
    except Exception:
        pass
    return options or [ConfigValueOption(title="(enter sink name manually)", value="")]


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # ruff: noqa: ARG001
    sink_options = await mass.loop.run_in_executor(None, _get_pa_sink_options)
    selected_map = str((values or {}).get(CONF_CHANNEL_MAP, CHANNEL_MAP_FLAC))
    show_custom = selected_map == CHANNEL_MAP_CUSTOM
    return (
        ConfigEntry(
            key=CONF_PA_SINK_NAME,
            type=ConfigEntryType.STRING,
            label="PulseAudio sink",
            options=sink_options,
            default_value="",
            description=(
                "Select the PulseAudio surround sink to use for multichannel output. "
                "This should be a 5.1 or 7.1 profile sink on your sound card."
            ),
            required=True,
        ),
        ConfigEntry(
            key=CONF_MULTICHANNEL_LAYOUT,
            type=ConfigEntryType.STRING,
            label="Channel layout",
            options=[
                ConfigValueOption(title="5.1 Surround (6 channels)", value=MULTICHANNEL_LAYOUT_51),
                ConfigValueOption(title="7.1 Surround (8 channels)", value=MULTICHANNEL_LAYOUT_71),
            ],
            default_value=MULTICHANNEL_LAYOUT_51,
            description=(
                "Select the surround layout matching the active profile on your sound card. "
                "Must match the PulseAudio sink configuration."
            ),
        ),
        ConfigEntry(
            key=CONF_CHANNEL_MAP,
            type=ConfigEntryType.STRING,
            label="Channel map",
            options=[
                ConfigValueOption(
                    title="FLAC / PCM standard (default)",
                    value=CHANNEL_MAP_FLAC,
                ),
                ConfigValueOption(
                    title="DVD / AC3 (FC\u2194LFE swapped)",
                    value=CHANNEL_MAP_DVD,
                ),
                ConfigValueOption(
                    title="Custom (enter index pairs below)",
                    value=CHANNEL_MAP_CUSTOM,
                ),
            ],
            default_value=CHANNEL_MAP_FLAC,
            description=(
                "Channel ordering of the PCM stream delivered by Music Assistant. "
                "FLAC/PCM standard is correct for the vast majority of sources. "
                "Use DVD/AC3 if center and LFE channels are swapped on playback. "
                "Custom allows manual specification as a comma-separated flat index list."
            ),
        ),
        ConfigEntry(
            key=CONF_CUSTOM_CHANNEL_MAP,
            type=ConfigEntryType.STRING,
            label="Custom channel map (index pairs)",
            default_value="",
            required=False,
            description=(
                "Only used when 'Custom' is selected above. "
                "Flat comma-separated indices for each sink pair in order: "
                "front_stereo, center_sub, rear_stereo[, side_stereo]. "
                "Example FLAC 5.1: 0,1,2,3,4,5  —  DVD 5.1: 0,1,3,2,4,5"
            ),
        ),


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MultiChannelAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
