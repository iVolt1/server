"""Local Audio In provider for Music Assistant.

Exposes PulseAudio hardware input sources (line-in, S/PDIF, HDMI-in)
as live audio streams in Music Assistant, modelled as radio stations.

Each non-monitor PulseAudio source becomes a browsable Radio item.
When played, audio is captured via ``ffmpeg -f pulse`` and streamed
through MA's custom audio pipeline using StreamType.CUSTOM.

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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 16
DEFAULT_CHANNELS = 2

# Well-known PA socket paths to probe when no server is explicitly configured.
# Listed in priority order for the HAOS / addon environment.
_PA_SOCKET_CANDIDATES: tuple[str, ...] = (
    "/run/audio/pulse.sock",  # HAOS Music Assistant addon
    "/run/pulse/native",      # Debian/Ubuntu system-wide daemon
)

# ffmpeg chunk size for streaming (bytes). 4 KiB keeps latency low while
# avoiding excessive syscall overhead.
_READ_CHUNK_BYTES = 4096

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.LIBRARY_RADIOS,
}


# ---------------------------------------------------------------------------
# Module-level entry points required by MA
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
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Return config entries to set up this provider."""
    return (
        ConfigEntry(
            key=CONF_PA_SERVER,
            type=ConfigEntryType.STRING,
            label=CONF_PA_SERVER,
            required=False,
            default_value="",
        ),
        ConfigEntry(
            key=CONF_SOURCE_NAME,
            type=ConfigEntryType.STRING,
            label=CONF_SOURCE_NAME,
            required=False,
            default_value="",
        ),
        ConfigEntry(
            key=CONF_DISPLAY_NAME,
            type=ConfigEntryType.STRING,
            label=CONF_DISPLAY_NAME,
            required=False,
            default_value="",
        ),
        ConfigEntry(
            key=CONF_SAMPLE_RATE,
            type=ConfigEntryType.INTEGER,
            label=CONF_SAMPLE_RATE,
            required=False,
            default_value=DEFAULT_SAMPLE_RATE,
            options=[
                ConfigValueOption("44100 Hz (CD)", 44100),
                ConfigValueOption("48000 Hz (HDMI / S/PDIF)", 48000),
                ConfigValueOption("88200 Hz (High-res)", 88200),
                ConfigValueOption("96000 Hz (High-res)", 96000),
            ],
        ),
        ConfigEntry(
            key=CONF_BIT_DEPTH,
            type=ConfigEntryType.INTEGER,
            label=CONF_BIT_DEPTH,
            required=False,
            default_value=DEFAULT_BIT_DEPTH,
            options=[
                ConfigValueOption("16-bit", 16),
                ConfigValueOption("24-bit", 24),
            ],
        ),
        ConfigEntry(
            key=CONF_CHANNELS,
            type=ConfigEntryType.INTEGER,
            label=CONF_CHANNELS,
            required=False,
            default_value=DEFAULT_CHANNELS,
            options=[
                ConfigValueOption("1 (Mono)", 1),
                ConfigValueOption("2 (Stereo)", 2),
            ],
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
    channels: int

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
    """Music provider that streams PulseAudio hardware inputs into MA."""

    async def handle_async_init(self) -> None:
        """Initialise the provider: resolve PA server address and verify connectivity."""
        self._pa_server: str = cast(str, self.config.get_value(CONF_PA_SERVER)) or ""
        self._source_filter: str = cast(str, self.config.get_value(CONF_SOURCE_NAME)) or ""
        self._override_display_name: str = (
            cast(str, self.config.get_value(CONF_DISPLAY_NAME)) or ""
        )
        self._sample_rate: int = (
            cast(int, self.config.get_value(CONF_SAMPLE_RATE)) or DEFAULT_SAMPLE_RATE
        )
        self._bit_depth: int = (
            cast(int, self.config.get_value(CONF_BIT_DEPTH)) or DEFAULT_BIT_DEPTH
        )
        self._channels: int = (
            cast(int, self.config.get_value(CONF_CHANNELS)) or DEFAULT_CHANNELS
        )

        # Active ffmpeg capture subprocesses keyed by PA source name.
        # Populated by get_audio_stream(); cleaned up in unload().
        self._capture_procs: dict[str, asyncio.subprocess.Process] = {}

        # Resolve PA server address: explicit config > PULSE_SERVER env > socket probe
        if not self._pa_server:
            self._pa_server = os.environ.get("PULSE_SERVER", "")
        if not self._pa_server:
            self._pa_server = _probe_pa_socket()

        self.logger.debug(
            "PulseAudio server: %s",
            self._pa_server if self._pa_server else "(system default)",
        )

        # Verify connectivity and log discovered sources
        sources = await self._list_pa_sources()
        if sources is None:
            self.logger.warning(
                "Cannot reach PulseAudio server (%s). "
                "Hardware input sources will not be available until PA is reachable.",
                self._pa_server or "default",
            )
        elif not sources:
            self.logger.info(
                "PulseAudio is reachable but no hardware input sources were found "
                "(source filter: %r).",
                self._source_filter or "none",
            )
        else:
            self.logger.info(
                "Discovered %d hardware input source(s): %s",
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
    # Browse / Library
    # ------------------------------------------------------------------

    async def browse(self, path: str) -> Sequence[MediaItemType | BrowseFolder]:
        """Return available input sources as Radio items."""
        sources = await self._list_pa_sources()
        if not sources:
            return []
        return [self._source_to_radio(s) for s in sources]

    async def get_library_radios(self) -> AsyncGenerator[Radio, None]:
        """Yield all available input sources as Radio items for the MA library."""
        sources = await self._list_pa_sources()
        if not sources:
            return
        for source in sources:
            yield self._source_to_radio(source)

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
        """Return StreamDetails for the given PA source name (item_id)."""
        sources = await self._list_pa_sources()
        if not sources or not any(s.name == item_id for s in sources):
            raise MediaNotFoundError(f"PA source not found: {item_id}")

        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.FLAC,
                sample_rate=self._sample_rate,
                bit_depth=self._bit_depth,
                channels=self._channels,
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
    ) -> AsyncGenerator[bytes, None]:
        """Capture audio from the PA source and yield FLAC-encoded bytes.

        Spawns an ``ffmpeg -f pulse`` subprocess per stream request.
        The subprocess is terminated when the caller stops consuming
        (CancelledError / GeneratorExit) or when the stream ends.

        ``compression_level 0`` gives lossless FLAC with minimal encode
        latency; the 4 KiB read chunk keeps the first-audio delay low.
        """
        source_name = streamdetails.item_id
        env = self._build_pa_env()

        # Select the correct ffmpeg sample format for the configured bit depth
        sample_fmt = "s16" if self._bit_depth <= 16 else "s32"

        cmd: list[str] = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            # PulseAudio input
            "-f", "pulse",
            "-i", source_name,
            # Output format
            "-ac", str(self._channels),
            "-ar", str(self._sample_rate),
            "-sample_fmt", sample_fmt,
            "-c:a", "flac",
            "-compression_level", "0",  # lossless, fastest encode
            "-f", "flac",
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
            assert proc.stdout is not None  # noqa: S101
            while True:
                chunk = await proc.stdout.read(_READ_CHUNK_BYTES)
                if not chunk:
                    # ffmpeg exited (source disconnected, etc.)
                    stderr_out = b""
                    if proc.stderr:
                        with contextlib.suppress(TimeoutError):
                            stderr_out = await asyncio.wait_for(
                                proc.stderr.read(), timeout=1.0
                            )
                    if stderr_out:
                        self.logger.warning(
                            "ffmpeg capture ended for '%s': %s",
                            source_name,
                            stderr_out.decode(errors="replace").strip(),
                        )
                    break
                yield chunk

        except (asyncio.CancelledError, GeneratorExit):
            # Normal stop: MA stopped the player or switched tracks
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
        radio = Radio(
            item_id=source.name,
            provider=self.instance_id,
            name=display_name,
        )
        radio.provider_mappings = {
            ProviderMapping(
                item_id=source.name,
                provider_domain=self.domain,
                provider_instance=self.instance_id,
            )
        }
        radio.metadata.description = (
            f"Live capture from PulseAudio source: {source.name}\n"
            f"{source.sample_rate} Hz / {source.channels} ch"
        )
        return radio

    async def _list_pa_sources(self) -> list[_PASource] | None:
        """Enumerate non-monitor PulseAudio sources via ``pactl list sources``.

        Returns a list of _PASource instances, or None if PA is unreachable.
        When ``source_filter`` is set, only the matching source is returned.
        """
        env = self._build_pa_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl", "list", "sources",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except FileNotFoundError:
            self.logger.debug("'pactl' not found in PATH")
            return None
        except asyncio.TimeoutError:
            self.logger.debug("pactl timed out")
            return None

        if proc.returncode != 0:
            self.logger.debug(
                "pactl list sources returned %d: %s",
                proc.returncode,
                stderr.decode(errors="replace").strip(),
            )
            return None

        sources = _parse_pactl_sources(stdout.decode(errors="replace"))

        if self._source_filter:
            sources = [s for s in sources if s.name == self._source_filter]

        return sources


# ---------------------------------------------------------------------------
# Module-level helpers (no provider state needed)
# ---------------------------------------------------------------------------

def _probe_pa_socket() -> str:
    """Return the first resolvable PulseAudio socket path, or empty string."""
    for path in _PA_SOCKET_CANDIDATES:
        if os.path.exists(path):
            return f"unix:{path}"
    return ""


def _parse_pactl_sources(output: str) -> list[_PASource]:
    """Parse ``pactl list sources`` output into a list of _PASource objects.

    Skips monitor sources (loopbacks of output sinks).  Both PulseAudio and
    PipeWire (with PA compatibility) are supported; monitor sources are
    identified by either:
    - Name ending in ``.monitor``
    - Description starting with "Monitor of"
    """
    sources: list[_PASource] = []
    current_name = ""
    current_desc = ""
    current_rate = DEFAULT_SAMPLE_RATE
    current_ch = DEFAULT_CHANNELS
    is_monitor = False
    in_source_block = False

    def _flush() -> None:
        if in_source_block and current_name and not is_monitor:
            sources.append(
                _PASource(
                    name=current_name,
                    description=current_desc,
                    sample_rate=current_rate,
                    channels=current_ch,
                )
            )

    for raw_line in output.splitlines():
        line = raw_line.strip()

        # New source block starts at column 0: "Source #N"
        if raw_line.startswith("Source #"):
            _flush()
            current_name = ""
            current_desc = ""
            current_rate = DEFAULT_SAMPLE_RATE
            current_ch = DEFAULT_CHANNELS
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
            # Format examples: "s16le 2ch 44100Hz"  "s32le 2ch 96000Hz"
            spec = line.split(":", 1)[1].strip()
            for token in spec.split():
                if token.endswith("Hz"):
                    with contextlib.suppress(ValueError):
                        current_rate = int(token[:-2])
                elif token.endswith("ch"):
                    with contextlib.suppress(ValueError):
                        current_ch = int(token[:-2])

    _flush()  # emit the last source block
    return sources


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """Gracefully terminate a subprocess, escalating to SIGKILL if needed."""
    if proc.returncode is not None:
        return  # already exited
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    except ProcessLookupError:
        pass
