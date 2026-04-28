"""S/PDIF Audio Out player provider for Music Assistant."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.mass import MusicAssistant

from .constants import (
    CONF_ENCODING_FORMAT,
    CONF_PA_SINK_NAME,
    ENCODING_AC3,
    ENCODING_DTS,
    ENCODING_EAC3,
)
from .provider import SPDIFAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.models import ProviderInstanceType

LOGGER = logging.getLogger(__name__)

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


async def _enumerate_iec958_sinks() -> list[str]:
    """
    Return a list of PulseAudio sink names that appear to support IEC 61937
    (S/PDIF or HDMI passthrough).

    Strategy: run ``pactl list sinks short`` and return any sink whose name
    contains ``iec958`` or ``hdmi``.  Falls back to all sinks if none match,
    so the user can still pick manually.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            "list",
            "sinks",
            "short",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("pactl list sinks short failed: %s", exc)
        return []

    sinks: list[str] = []
    all_sinks: list[str] = []
    for line in stdout.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        all_sinks.append(name)
        if "iec958" in name.lower() or "hdmi" in name.lower():
            sinks.append(name)

    # Fall back to all sinks so the dropdown is never empty
    return sinks if sinks else all_sinks


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    sink_names = await _enumerate_iec958_sinks()

    if sink_names:
        sink_entry = ConfigEntry(
            key=CONF_PA_SINK_NAME,
            type=ConfigEntryType.STRING,
            label="PulseAudio S/PDIF sink",
            options=[ConfigValueOption(title=s, value=s) for s in sink_names],
            default_value=sink_names[0],
            description=(
                "PulseAudio sink to write IEC 61937-encoded audio to. "
                "Sinks with 'iec958' or 'hdmi' in their name are listed first."
            ),
        )
    else:
        # pactl unavailable at config time — free-text fallback
        sink_entry = ConfigEntry(
            key=CONF_PA_SINK_NAME,
            type=ConfigEntryType.STRING,
            label="PulseAudio S/PDIF sink",
            default_value="",
            description=(
                "Enter the PulseAudio sink name manually "
                "(e.g. HD_Audio_Generic_iec958_stereo). "
                "Run 'pactl list sinks short' to find available sinks."
            ),
        )

    encoding_entry = ConfigEntry(
        key=CONF_ENCODING_FORMAT,
        type=ConfigEntryType.STRING,
        label="Encoding format",
        options=[
            ConfigValueOption(
                title="AC3 / Dolby Digital 5.1 (640 kbps)", value=ENCODING_AC3
            ),
            ConfigValueOption(title="DTS 5.1 (1536 kbps)", value=ENCODING_DTS),
            ConfigValueOption(
                title="E-AC3 / Dolby Digital Plus (1536 kbps, HDMI only)",
                value=ENCODING_EAC3,
            ),
        ],
        default_value=ENCODING_AC3,
        description=(
            "Encoding format to use for S/PDIF IEC 61937 passthrough. "
            "AC3 is the most compatible option for optical S/PDIF. "
            "DTS requires a DTS-capable receiver. "
            "E-AC3 requires HDMI — do not use with optical S/PDIF."
        ),
    )

    return (sink_entry, encoding_entry)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider instance with given configuration."""
    return SPDIFAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
