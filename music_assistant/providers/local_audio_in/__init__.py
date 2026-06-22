"""
Local Audio In provider for Music Assistant.

Exposes PulseAudio audio sources (hardware inputs and optionally sink monitors)
as live audio streams in Music Assistant, modelled as radio stations.

Each qualifying PulseAudio source becomes a browsable Radio item.
When played, audio is captured via ``ffmpeg -f pulse`` and streamed
through MA's custom audio pipeline using StreamType.CUSTOM.

Sample rate, bit depth, and channel count are auto-detected from the PA
source's native format; the config entries act as overrides only.

Requires ffmpeg and pactl (pulseaudio-utils) in the container/system PATH.
PipeWire with the PulseAudio compatibility layer is fully supported.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import (
    AudioFormat,
    BrowseFolder,
    MediaItemType,
    ProviderMapping,
    Radio,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# ---------------------------------------------------------------------------
# Config entry keys (must match strings.json config keys)
# ---------------------------------------------------------------------------
CONF_PA_SERVER = "pa_server"
CONF_SOURCE_NAME = "source_name"
CONF_DISPLAY_NAME = "display_name"
CONF_SAMPLE_RATE = "sample_rate"
CONF_BIT_DEPTH = "bit_depth"
CONF_CHANNELS = "channels"
CONF_INCLUDE_MONITORS = "include_monitors"

# ---------------------------------------------------------------------------
# Sentinel: 0 means "auto-detect from source"
# ---------------------------------------------------------------------------
AUTO = 0

# Fallback values used only when pactl is unavailable
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_BIT_DEPTH = 16
_DEFAULT_CHANNELS = 2

# Well-known PA socket paths to probe when no server is explicitly configured.
_PA_SOCKET_CANDIDATES: tuple[str, ...] = (
    "/run/audio/pulse.sock",  # HAOS Music Assistant addon
    "/run/pulse/native",  # Debian/Ubuntu system-wide daemon
)

# ffmpeg chunk size (bytes). 4 KiB keeps first-audio latency low.
_READ_CHUNK_BYTES = 4096

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
}


# ---------------------------------------------------------------------------
# Module-level entry points
# ---------------------------------------------------------------------------


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialise and return a LocalAudioInProvider instance."""
    return LocalAudioInProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """
    Return config entries to set up this provider.

    Source name and display name dropdowns are dynamically populated from
    the live PulseAudio server.  Format options (sample rate, bit depth,
    channels) include an Auto entry that reads the source's native format.
    """
    current_pa_server = str(values.get(CONF_PA_SERVER, "") if values else "") or ""

    # --- pa_server options: detected socket paths ---
    pa_server_options: list[ConfigValueOption] = [
        ConfigValueOption("", "(auto-detect)"),
    ]
    for path in _PA_SOCKET_CANDIDATES:
        if os.path.exists(path):
            uri = f"unix:{path}"
            pa_server_options.append(ConfigValueOption(uri, uri))

    # --- Enumerate all PA sources (hardware + monitors) for dropdowns ---
    all_sources = await _enumerate_pa_sources(current_pa_server, include_monitors=True) or []

    source_name_options: list[ConfigValueOption] = [
        ConfigValueOption("", "(all hardware inputs)"),
    ]
    display_name_options: list[ConfigValueOption] = [
        ConfigValueOption("", "(use PA source description)"),
    ]
    for source in all_sources:
        source_name_options.append(ConfigValueOption(source.name, source.display_label))
        if source.description and source.description != source.name:
            display_name_options.append(ConfigValueOption(source.description, source.description))

    return (
        ConfigEntry(
            key=CONF_PA_SERVER,
            type=ConfigEntryType.STRING,
            label=CONF_PA_SERVER,
            required=False,
            default_value="",
            options=pa_server_options,
        ),
        ConfigEntry(
            key=CONF_SOURCE_NAME,
            type=ConfigEntryType.STRING,
            label=CONF_SOURCE_NAME,
            required=False,
            default_value="",
            options=source_name_options,
        ),
        ConfigEntry(
            key=CONF_DISPLAY_NAME,
            type=ConfigEntryType.STRING,
            label=CONF_DISPLAY_NAME,
            required=False,
            default_value="",
            options=display_name_options,
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label=CONF_SAMPLE_RATE,
            required=False,
            default_value=AUTO,
            options=[
                ConfigValueOption(AUTO, "Auto (detect from source)"),
                ConfigValueOption(44100, "44100 Hz (CD)"),
                ConfigValueOption(48000, "48000 Hz (HDMI / S/PDIF)"),
                ConfigValueOption(88200, "88200 Hz"),
                ConfigValueOption(96000, "96000 Hz (High-res)"),
                ConfigValueOption(176400, "176400 Hz"),
                ConfigValueOption(192000, "192000 Hz (High-res)"),
                ConfigValueOption(352800, "352800 Hz"),
                ConfigValueOption(384000, "384000 Hz (Ultra high-res)"),
            ],
        ),
        ConfigEntry(
            key=CONF_BIT_DEPTH,
            type=ConfigEntryType.INTEGER,
            label=CONF_BIT_DEPTH,
            required=False,
            default_value=AUTO,
            options=[
                ConfigValueOption(AUTO, "Auto (detect from source)"),
                ConfigValueOption(16, "16-bit"),
                ConfigValueOption(24, "24-bit"),
                ConfigValueOption(32, "32-bit"),
            ],
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label=CONF_CHANNELS,
            required=False,
            default_value=AUTO,
            options=[
                ConfigValueOption(AUTO, "Auto (detect from source)"),
                ConfigValueOption(1, "1 (Mono)"),
                ConfigValueOption(2, "2 (Stereo)"),
            ],
        ),
        ConfigEntry(
            key=CONF_INCLUDE_MONITORS,
            type=ConfigEntryType.BOOLEAN,
            label=CONF_INCLUDE_MONITORS,
            required=False,
            default_value=False,
        ),
    )


# ---------------------------------------------------------------------------
# Internal data class for a discovered PA source
# ---------------------------------------------------------------------------
@dataclass
class _PASource:
    name: str
    description: str
    sample_rate: int
    bit_depth: int
    channels: int
    is_monitor: bool = False

    @property
    def display_label(self) -> str:
        """Return the description if it differs meaningfully from the raw name."""
        if self.description and self.description != self.name:
            return self.description
        return self.name


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalAudioInProvider(MusicProvider):
    """Music provider that streams PulseAudio sources into MA."""

    async def handle_async_init(self) -> None:
        """Initialise the provider: resolve PA server address and verify connectivity."""
        self._pa_server: str = cast("str", self.config.get_value(CONF_PA_SERVER)) or ""
        self._source_filter: str = cast("str", self.config.get_value(CONF_SOURCE_NAME)) or ""
        self._override_display_name: str = (
            cast("str", self.config.get_value(CONF_DISPLAY_NAME)) or ""
        )
        # 0 = Auto: use source's native value
        self._override_sample_rate: int = (
            cast("int", self.config.get_value(CONF_SAMPLE_RATE)) or AUTO
        )
        self._override_bit_depth: int = cast("int", self.config.get_value(CONF_BIT_DEPTH)) or AUTO
        self._override_channels: int = cast("int", self.config.get_value(CONF_CHANNELS)) or AUTO
        self._include_monitors: bool = bool(self.config.get_value(CONF_INCLUDE_MONITORS))

        # Active ffmpeg capture subprocesses keyed by PA source name.
        self._capture_procs: dict[str, asyncio.subprocess.Process] = {}

        # Resolve PA server: explicit config > PULSE_SERVER env > socket probe
        if not self._pa_server:
            self._pa_server = os.environ.get("PULSE_SERVER", "")
        if not self._pa_server:
            self._pa_server = _probe_pa_socket()

        self.logger.debug(
            "PulseAudio server: %s",
            self._pa_server if self._pa_server else "(system default)",
        )

        sources = await self._list_pa_sources()
        if sources is None:
            self.logger.warning(
                "Cannot reach PulseAudio server (%s).",
                self._pa_server or "default",
            )
        elif not sources:
            self.logger.info(
                "No sources found (filter: %r, monitors: %s).",
                self._source_filter or "none",
                self._include_monitors,
            )
        else:
            self.logger.info(
                "Discovered %d source(s): %s",
                len(sources),
                [s.name for s in sources],
            )

    async def unload(self, is_removed: bool = False) -> None:
        """Unload provider; terminate any active capture subprocesses."""
        for source_name, proc in list(self._capture_procs.items()):
            self.logger.debug("Stopping capture subprocess for '%s'", source_name)
            await _terminate_proc(proc)
        self._capture_procs.clear()

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------

    async def browse(self, path: str) -> Sequence[MediaItemType | BrowseFolder]:
        """Return available input sources as Radio items."""
        sources = await self._list_pa_sources()
        if not sources:
            return []
        return [self._source_to_radio(s) for s in sources]

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Return a single Radio item by its provider ID (PA source name)."""
        sources = await self._list_pa_sources()
        for source in sources or []:
            if source.name == prov_radio_id:
                return self._source_to_radio(source)
        raise MediaNotFoundError(f"PA source not found: {prov_radio_id}")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Return StreamDetails for the given PA source name (item_id).

        Format (sample rate, bit depth, channels) is taken from the source's
        native values unless an explicit override is configured.
        """
        sources = await self._list_pa_sources()
        source = next((s for s in (sources or []) if s.name == item_id), None)
        if source is None:
            raise MediaNotFoundError(f"PA source not found: {item_id}")

        sample_rate = self._override_sample_rate or source.sample_rate or _DEFAULT_SAMPLE_RATE
        bit_depth = self._override_bit_depth or source.bit_depth or _DEFAULT_BIT_DEPTH
        channels = self._override_channels or source.channels or _DEFAULT_CHANNELS

        # FLAC is limited to 24-bit; use PCM_S32LE for 32-bit sources so the
        # signal chain reflects the true source format without truncation.
        if bit_depth == 32:
            content_type = ContentType.PCM_S32LE
        elif bit_depth == 24:
            content_type = ContentType.PCM_S24LE
        else:
            content_type = ContentType.FLAC

        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=content_type,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
            ),
            stream_type=StreamType.CUSTOM,
            media_type=MediaType.RADIO,
            can_seek=False,
            duration=0,
        )

    async def get_audio_stream(
        self,
        streamdetails: StreamDetails,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """
        Capture audio from the PA source and yield encoded bytes.

        For 32-bit sources: raw PCM (s32le) — no encode overhead, exact format.
        For 24-bit sources: raw PCM (s24le packed as s32le).
        For 16-bit sources: FLAC with compression_level 0.
        """
        source_name = streamdetails.item_id
        env = self._build_pa_env()
        fmt = streamdetails.audio_format
        bit_depth = fmt.bit_depth
        sample_rate = fmt.sample_rate
        channels = fmt.channels

        if fmt.content_type in (ContentType.PCM_S32LE, ContentType.PCM_S24LE):
            # Raw PCM passthrough — lowest latency, exact bit depth
            out_fmt = "s32le"
            sample_fmt = "s32"
            codec_args: list[str] = ["-f", out_fmt]
        else:
            # FLAC for 16-bit sources
            sample_fmt = "s16"
            codec_args = ["-c:a", "flac", "-compression_level", "0", "-f", "flac"]

        cmd: list[str] = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "pulse",
            "-i",
            source_name,
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            sample_fmt,
            *codec_args,
            "pipe:1",
        ]

        self.logger.debug("Starting capture: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._capture_procs[source_name] = proc

        try:
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(_READ_CHUNK_BYTES)
                if not chunk:
                    stderr_out = b""
                    if proc.stderr:
                        with contextlib.suppress(TimeoutError):
                            stderr_out = await asyncio.wait_for(proc.stderr.read(), timeout=1.0)
                    if stderr_out:
                        self.logger.warning(
                            "ffmpeg capture ended for '%s': %s",
                            source_name,
                            stderr_out.decode(errors="replace").strip(),
                        )
                    break
                yield chunk

        except asyncio.CancelledError, GeneratorExit:
            self.logger.debug("Capture cancelled for '%s'", source_name)

        finally:
            self._capture_procs.pop(source_name, None)
            await _terminate_proc(proc)
            self.logger.debug("Capture subprocess cleaned up for '%s'", source_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pa_env(self) -> dict[str, str]:
        """Return os.environ with PULSE_SERVER injected if configured."""
        env = dict(os.environ)
        if self._pa_server:
            env["PULSE_SERVER"] = self._pa_server
        return env

    def _source_to_radio(self, source: _PASource) -> Radio:
        """Convert a _PASource to an MA Radio item."""
        display_name = self._override_display_name or source.display_label
        return Radio(
            item_id=source.name,
            provider=self.instance_id,
            name=display_name,
            provider_mappings={
                ProviderMapping(
                    item_id=source.name,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )

    async def _list_pa_sources(self) -> list[_PASource] | None:
        """
        Return filtered PA sources, or None if PA is unreachable.

        When source_filter is set, the named source is returned regardless
        of whether it is a monitor.  When source_filter is empty, only
        hardware inputs are returned unless include_monitors is True.
        """
        all_sources = await _enumerate_pa_sources(self._pa_server, include_monitors=True)
        if all_sources is None:
            return None

        if self._source_filter:
            # Explicit filter: honour it for both hardware and monitor sources
            return [s for s in all_sources if s.name == self._source_filter]

        # No filter: apply monitor gate
        if self._include_monitors:
            return all_sources
        return [s for s in all_sources if not s.is_monitor]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _probe_pa_socket() -> str:
    """Return the first resolvable PulseAudio socket path, or empty string."""
    for path in _PA_SOCKET_CANDIDATES:
        if os.path.exists(path):
            return f"unix:{path}"
    return ""


async def _enumerate_pa_sources(
    pa_server: str = "",
    include_monitors: bool = False,
) -> list[_PASource] | None:
    """
    Enumerate PulseAudio sources via ``pactl list sources``.

    Returns all hardware inputs (and optionally monitor sources).
    Returns None if PA is unreachable.
    """
    env = dict(os.environ)
    resolved = pa_server or os.environ.get("PULSE_SERVER", "") or _probe_pa_socket()
    if resolved:
        env["PULSE_SERVER"] = resolved

    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            "list",
            "sources",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except FileNotFoundError:
        return None
    except TimeoutError:
        return None

    if proc.returncode != 0:
        return None

    sources = _parse_pactl_sources(stdout.decode(errors="replace"))
    if not include_monitors:
        sources = [s for s in sources if not s.is_monitor]
    return sources


def _parse_pactl_sources(output: str) -> list[_PASource]:
    """
    Parse ``pactl list sources`` output into a list of _PASource objects.

    Extracts name, description, sample rate, bit depth, and channel count
    from each Source block.  The is_monitor flag is set for sources whose
    name ends in ``.monitor`` or whose description starts with "Monitor of".
    """
    sources: list[_PASource] = []
    current_name = ""
    current_desc = ""
    current_rate = _DEFAULT_SAMPLE_RATE
    current_bit_depth = _DEFAULT_BIT_DEPTH
    current_ch = _DEFAULT_CHANNELS
    is_monitor = False
    in_source_block = False

    def _flush() -> None:
        if in_source_block and current_name:
            sources.append(
                _PASource(
                    name=current_name,
                    description=current_desc,
                    sample_rate=current_rate,
                    bit_depth=current_bit_depth,
                    channels=current_ch,
                    is_monitor=is_monitor,
                )
            )

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if raw_line.startswith("Source #"):
            _flush()
            current_name = ""
            current_desc = ""
            current_rate = _DEFAULT_SAMPLE_RATE
            current_bit_depth = _DEFAULT_BIT_DEPTH
            current_ch = _DEFAULT_CHANNELS
            is_monitor = False
            in_source_block = True
            continue

        if not in_source_block:
            continue

        if line.startswith("Name:"):
            current_name = line.split(":", 1)[1].strip()
            if current_name.endswith(".monitor"):
                is_monitor = True

        elif line.startswith("Description:"):
            current_desc = line.split(":", 1)[1].strip()
            if current_desc.startswith("Monitor of"):
                is_monitor = True

        elif line.startswith("Sample Specification:"):
            # e.g. "s32le 2ch 96000Hz" or "s16le 2ch 44100Hz"
            spec = line.split(":", 1)[1].strip()
            tokens = spec.split()
            if tokens:
                # Parse bit depth from format token (s32le → 32, s16le → 16)
                fmt = tokens[0].lower()
                for bits in (32, 24, 16, 8):
                    if str(bits) in fmt:
                        current_bit_depth = bits
                        break
            for token in tokens:
                if token.endswith("Hz"):
                    with contextlib.suppress(ValueError):
                        current_rate = int(token[:-2])
                elif token.endswith("ch"):
                    with contextlib.suppress(ValueError):
                        current_ch = int(token[:-2])

    _flush()
    return sources


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """Gracefully terminate a subprocess, escalating to SIGKILL if needed."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    except ProcessLookupError:
        pass
