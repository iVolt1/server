"""Digital Audio provider for Music Assistant — direct PulseAudio multichannel output."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from music_assistant.mass import MusicAssistant

from .constants import CONF_PA_SINK_NAME
from .provider import DigitalAudioProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES = {
    ProviderFeature.SYNC_PLAYERS,
}


def _get_pa_sink_options() -> list[ConfigValueOption]:
    """Return available PA sinks as config options by running pactl.

    Appends the actual channel count to each label (parsed from
    sample_specification, e.g. "s32le 8ch 96000Hz") so multichannel-capable
    sinks are obvious at a glance without filtering out stereo-only sinks —
    this provider works generically for any channel count, 2ch included.
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
                if not name:
                    continue
                channels_label = ""
                spec_str: str = sink.get("sample_specification", "")
                parts = spec_str.split()
                if len(parts) >= 2 and parts[1].endswith("ch"):
                    channels_label = f" — {parts[1]}"
                options.append(
                    ConfigValueOption(title=f"{desc}{channels_label}", value=name)
                )
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
            label="PulseAudio sink",
            options=sink_options,
            default_value="",
            description=(
                "Select the target PulseAudio sink for direct multichannel output "
                "(analog surround, HDMI audio, or USB multichannel device). "
                "Channel count, sample rate, and bit depth are auto-detected from "
                "the sink — no separate layout configuration needed."
            ),
            required=True,
        ),
    )


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return DigitalAudioProvider(mass, manifest, config, SUPPORTED_FEATURES)
