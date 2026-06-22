"""
Local Audio In provider for Music Assistant.

Exposes PulseAudio audio sources (hardware inputs and optionally sink monitors)
as live audio streams in Music Assistant, modelled as radio stations.

All qualifying sources are auto-discovered with no per-source configuration.
Each source is named from its PA description with format details appended,
e.g. "Built-in Audio Analog Stereo (96000, 32, 2)".

Sample rate, bit depth, and channel count are read from the source's native
PA format; no manual configuration is required or available.

For 32 and 24-bit sources, audio is streamed as raw PCM (no encode overhead).
For 16-bit sources, FLAC is used (lossless, minimal CPU cost).

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
CONF_INCLUDE_MONITORS = "include_monitors"

# Fallback format values used only when pactl is unavailable
_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_BIT_DEPTH = 16
_DEFAULT_CHANNELS = 2

# Well-known PA socket paths to probe when no server is explicitly configured.
_PA_SOCKET_CANDIDATES: tuple[str, ...] = (
    "/run/audio/pulse.sock",  # HAOS Music Assistant addon
    "/run/pulse/native",  # Debian/Ubuntu system-wide daemon
)

# Read chunk size for the audio stream loop (bytes).
# Smaller = lower first-audio latency; 2 KiB is a good balance for PCM.
_READ_CHUNK_BYTES = 2048

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

    Only two entries are needed: the PA server address (auto-detected by
    default) and an optional toggle to include sink monitor sources.
    All audio sources are discovered and presented automatically with no
    per-source configuration required.
    """
    current_pa_server = str(values.get(CONF_PA_SERVER, "") if values else "") or ""

    pa_server_options: list[ConfigValueOption] = [
        ConfigValueOption("", "(auto-detect)"),
    ]
    for path in _PA_SOCKET_CANDIDATES:
        if os.path.exists(path):
            uri = f"unix:{path}"
            pa_server_options.append(ConfigValueOption(uri, uri))

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
        """
        Human-readable name: PA description with format details appended.

        Example: "Built-in Audio Analog Stereo (96000, 32, 2)"
        Falls back to the raw source name if no description is available.
        """
        base = self.description if self.description else self.name
        return f"{base} ({self.sample_rate}, {self.bit_depth}, {self.channels})"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class LocalAudioInProvider(MusicProvider):
    """Music provider that auto-discovers and streams PulseAudio sources."""

    async def handle_async_init(self) -> None:
        """Initialise: resolve PA server address and log discovered sources."""
        self._pa_server: str = cast("str", self.config.get_value(CONF_PA_SERVER)) or ""
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
                "No sources found (monitors included: %s).",
                self._include_monitors,
            )
        else:
            self.logger.info(
                "Discovered %d source(s): %s",
                len(sources),
                [s.display_label for s in sources],
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
        """Return all discovered PA input sources as Radio items."""
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
        Return StreamDetails using the source's native format.

        32 and 24-bit sources stream as raw PCM to avoid FLAC's 24-bit cap
        and eliminate the encode step.  16-bit sources use FLAC.
        """
        sources = await self._list_pa_sources()
        source = next((s for s in (sources or []) if s.name == item_id), None)
        if source is None:
            raise MediaNotFoundError(f"PA source not found: {item_id}")

        sample_rate = source.sample_rate or _DEFAULT_SAMPLE_RATE
        bit_depth = source.bit_depth or _DEFAULT_BIT_DEPTH
        channels = source.channels or _DEFAULT_CHANNELS

        # Always use FLAC: lossless, well-framed, reliably decoded by MA.
        # FLAC caps at 24-bit; 32-bit sources are stored as 24-bit FLAC which
        # is sonically transparent for any real-world line-in signal.
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
            # Disable loudness normalization: live line-in is already at
            # the correct level; MA's default boost causes severe clipping.
            loudness_correction_gain=0.0,
        )

    async def get_audio_stream(
        self,
        streamdetails: StreamDetails,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """
        Capture audio from the PA source and yield encoded bytes.

        FLAC with compression_level 0 for all sources — lossless, low
        encode overhead, and reliably parsed by MA's stream pipeline.
        PA fragment_size is tuned to ~10ms for low capture latency.
        """
        source_name = streamdetails.item_id
        env = self._build_pa_env()
        fmt = streamdetails.audio_format

        # Target ~10ms PA fragment size to minimise capture-side latency.
        # PA treats this as a hint; actual fragment may be slightly larger.
        bytes_per_ms = (fmt.sample_rate * (fmt.bit_depth // 8) * fmt.channels) // 1000
        fragment_size = max(bytes_per_ms * 10, 512)

        # FLAC sample format: s16 for 16-bit sources, s32 for 24/32-bit.
        # (FLAC encodes s32 input as 24-bit internally — lossless for audio.)
        sample_fmt = "s16" if fmt.bit_depth <= 16 else "s32"

        cmd: list[str] = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            # Suppress input probing — format is fully known
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            # Reduce ffmpeg's internal packet queue
            "-fflags",
            "nobuffer",
            # PulseAudio input with small fragment for low capture latency
            "-f",
            "pulse",
            "-fragment_size",
            str(fragment_size),
            "-i",
            source_name,
            "-ac",
            str(fmt.channels),
            "-ar",
            str(fmt.sample_rate),
            "-sample_fmt",
            sample_fmt,
            "-c:a",
            "flac",
            "-compression_level",
            "0",
            "-f",
            "flac",
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
        return Radio(
            item_id=source.name,
            provider=self.instance_id,
            name=source.display_label,
            provider_mappings={
                ProviderMapping(
                    item_id=source.name,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )

    async def _list_pa_sources(self) -> list[_PASource] | None:
        """Return all qualifying PA sources, or None if PA is unreachable."""
        return await _enumerate_pa_sources(
            self._pa_server,
            include_monitors=self._include_monitors,
        )


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

    Returns hardware inputs and optionally monitor (loopback) sources.
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

    Extracts name, description, sample rate, bit depth, channels, and
    monitor flag from each Source block.
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
