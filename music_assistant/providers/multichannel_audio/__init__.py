"""Multichannel Audio player provider for Music Assistant."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.mass import MusicAssistant

from .constants import (
    CONF_MULTICHANNEL_LAYOUT,
    CONF_VOLUME_CONTROL,
    MULTICHANNEL_LAYOUT_51,
    MULTICHANNEL_LAYOUT_71,
    VOLUME_CONTROL_DISABLED,
    VOLUME_CONTROL_HARDWARE,
    VOLUME_CONTROL_SOFTWARE,
)
from .provider import MultiChannelAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # ruff: noqa: ARG001
    entries: list[ConfigEntry] = [
        ConfigEntry(
            key=CONF_MULTICHANNEL_LAYOUT,
            type=ConfigEntryType.STRING,
            label="Multichannel layout",
            options=[
                ConfigValueOption(title="5.1 Surround", value=MULTICHANNEL_LAYOUT_51),
                ConfigValueOption(title="7.1 Surround", value=MULTICHANNEL_LAYOUT_71),
            ],
            default_value=MULTICHANNEL_LAYOUT_51,
            description=(
                "Select the surround channel layout matching your PulseAudio sink configuration. "
                "5.1 uses FL, FR, FC, LFE, RL, RR (6 channels). "
                "7.1 adds side channels SL, SR (8 channels)."
            ),
        ),
        ConfigEntry(
            key=CONF_VOLUME_CONTROL,
            type=ConfigEntryType.STRING,
            label="Volume control mode",
            options=[
                ConfigValueOption(title="Hardware (preferred)", value=VOLUME_CONTROL_HARDWARE),
                ConfigValueOption(title="Software", value=VOLUME_CONTROL_SOFTWARE),
                ConfigValueOption(title="Disabled", value=VOLUME_CONTROL_DISABLED),
            ],
            default_value=VOLUME_CONTROL_HARDWARE,
            description=(
                "Hardware uses PulseAudio sink-input volume control. "
                "Software applies volume scaling to the PCM stream. "
                "Disabled passes audio at full volume."
            ),
        ),
    ]
    return tuple(entries)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MultiChannelAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
