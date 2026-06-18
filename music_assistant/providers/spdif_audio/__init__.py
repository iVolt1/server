"""S/PDIF Audio provider for Music Assistant — AC3/IEC61937 passthrough output."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.mass import MusicAssistant

from .constants import CONF_AC3_BITRATE, CONF_PA_SINK_NAME, DEFAULT_AC3_BITRATE
from .provider import SpdifAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


def _get_pa_sink_options() -> list[ConfigValueOption]:
    """Return available PA sinks as config options by running pactl.

    Unlike digital_audio, this is not annotated with channel count — the PA
    sink itself reports as a plain 2-channel digital profile regardless of
    how many real content channels are encoded inside the AC3 bitstream it
    carries, so a channel-count label here would be misleading.
    """
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
    return (
        ConfigEntry(
            key=CONF_PA_SINK_NAME,
            type=ConfigEntryType.STRING,
            label="PulseAudio sink (S/PDIF output)",
            options=sink_options,
            default_value="",
            description=(
                "Select the optical or coax S/PDIF sink connected to your receiver "
                "or soundbar. Stereo content plays as plain PCM; 5.1 surround "
                "content is encoded to Dolby Digital and sent as a passthrough "
                "bitstream for the receiver to decode."
            ),
            required=True,
        ),
        ConfigEntry(
            key=CONF_AC3_BITRATE,
            type=ConfigEntryType.STRING,
            label="Dolby Digital encode bitrate",
            options=[
                ConfigValueOption(title="384 kbps", value="384k"),
                ConfigValueOption(title="448 kbps (DVD standard)", value="448k"),
                ConfigValueOption(title="640 kbps (maximum quality)", value="640k"),
            ],
            default_value=DEFAULT_AC3_BITRATE,
            required=False,
            description=(
                "Bitrate used when encoding 5.1 surround content to Dolby Digital "
                "for passthrough. Only applies to multichannel sources — stereo "
                "content is never re-encoded. 640 kbps matches Blu-ray quality and "
                "is supported by virtually all receivers/soundbars."
            ),
        ),
    )


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SpdifAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
