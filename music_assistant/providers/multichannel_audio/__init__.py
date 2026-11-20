"""Multichannel Audio player provider for Music Assistant."""

from __future__ import annotations

import json
import logging
import subprocess
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
    CONF_SINK_CENTER,
    CONF_SINK_FRONT,
    CONF_SINK_REAR,
    CONF_SINK_SIDE,
    MULTICHANNEL_LAYOUT_51,
    MULTICHANNEL_LAYOUT_71,
)
from .provider import MultiChannelAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.models import ProviderInstanceType

LOGGER = logging.getLogger(__name__)

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


def _get_pa_sink_options() -> list[ConfigValueOption]:
    """Return available PA sinks as config options by running pactl."""
    options: list[ConfigValueOption] = []
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            sinks = [s for s in json.loads(result.stdout) if s.get("name")]
            descs = [s.get("description", s["name"]) for s in sinks]
            for sink, desc in zip(sinks, descs, strict=True):
                name: str = sink["name"]
                # remap sinks of one card all share the card's description; only their
                # (user-chosen) sink name tells them apart
                title = name if descs.count(desc) > 1 else desc
                options.append(ConfigValueOption(title=title, value=name))
    except Exception as err:  # noqa: BLE001 - listing sinks must never break the settings page
        LOGGER.debug("Could not list PulseAudio sinks: %s", err)
    return options or [ConfigValueOption("", translation_key="manual")]


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # ruff: noqa: ARG001
    sink_options = await mass.loop.run_in_executor(None, _get_pa_sink_options)
    pair_options = [ConfigValueOption("", translation_key="auto")] + [
        o for o in sink_options if o.value
    ]
    return (
        ConfigEntry(
            key=CONF_PA_SINK_NAME,
            type=ConfigEntryType.STRING,
            options=sink_options,
            default_value="",
            required=True,
        ),
        ConfigEntry(
            key=CONF_MULTICHANNEL_LAYOUT,
            type=ConfigEntryType.STRING,
            options=[
                ConfigValueOption(MULTICHANNEL_LAYOUT_51, translation_key="surround_51"),
                ConfigValueOption(MULTICHANNEL_LAYOUT_71, translation_key="surround_71"),
            ],
            default_value=MULTICHANNEL_LAYOUT_51,
        ),
        ConfigEntry(
            key=CONF_CHANNEL_MAP,
            type=ConfigEntryType.STRING,
            options=[
                ConfigValueOption(CHANNEL_MAP_FLAC),
                ConfigValueOption(CHANNEL_MAP_DVD),
                ConfigValueOption(CHANNEL_MAP_CUSTOM),
            ],
            default_value=CHANNEL_MAP_FLAC,
        ),
        ConfigEntry(
            key=CONF_CUSTOM_CHANNEL_MAP,
            type=ConfigEntryType.STRING,
            default_value="",
            required=False,
        ),
        *(
            ConfigEntry(
                key=key,
                type=ConfigEntryType.STRING,
                options=pair_options,
                default_value="",
                required=False,
            )
            for key in (CONF_SINK_FRONT, CONF_SINK_CENTER, CONF_SINK_REAR, CONF_SINK_SIDE)
        ),
    )


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MultiChannelAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
