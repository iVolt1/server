"""
Local Audio In provider for Music Assistant.

Exposes PulseAudio audio sources (hardware inputs and optionally sink monitors)
as live AudioSource streams in Music Assistant, listed under the global
'Live Inputs' browse node.

All qualifying sources are auto-discovered at startup with no per-source
configuration.  Each source is named from its PA description with format
details appended, e.g. "Built-in Audio Analog Stereo (96000, 32, 2)".

Sample rate, bit depth, and channel count are read from the source's PA
server rate (not the raw hardware rate); using the server rate keeps source
and output sink rates aligned, avoiding MA-side resampling through the lower-
quality swr fallback when libsoxr is unavailable.

For 32 and 24-bit sources, audio is streamed as raw PCM (no encode overhead).
For 16-bit sources, FLAC is used (lossless, minimal CPU cost).

Requires ffmpeg and pactl (pulseaudio-utils) in the container/system PATH.
PipeWire with the PulseAudio compatibility layer is fully supported.

NOTE — Favorites / Shortcuts:
AudioSource items are surfaced under the global 'Live Inputs' browse node but
are NOT favoritable or library-backed in MA core today.  This is a current
MA-wide limitation on the AudioSource type, not specific to this provider.
Until the core adds favorites support for AudioSource, sources are reached
through Home → Live Inputs rather than a favorites shortcut.  The original
MusicProvider/Radio approach DID support favorites; this is the only
user-facing regression of the PluginProvider rebase.  If MA adds AudioSource
favorites in a future release this provider will gain them automatically with
no code change.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncGenerator
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
    AudioSource,
    ProviderMapping,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.models.plugin import PluginProvider

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

# PluginProvider with AUDIO_SOURCE: sources appear under the global
# 'Live Inputs' browse node.  BROWSE is not needed — the core handles it.
SUPPORTED_FEATURES = {
    ProviderFeature.AUDIO_SOURCE,
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
class LocalAudioInProvider(PluginProvider):
    """Plugin provider that auto-discovers and streams PulseAudio sources."""

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
    # AudioSource exposure
    # ------------------------------------------------------------------

    async def get_audio_sources(self) -> list[AudioSource]:
        """
        Return all discovered PA input sources as AudioSource items.

        Sources appear under the global 'Live Inputs' browse node in MA.

        NOTE — Favorites gap: AudioSource items are NOT favoritable or
        library-backed in MA core as of this writing.  This is a known
        MA-wide limitation (see module docstring for full context).  The
        previous MusicProvider/Radio implementation DID allow favorites;
        if this matters for your workflow, track upstream issue progress
        and this provider will benefit automatically when core support
        lands.
        """
        sources = await self._list_pa_sources()
        if not sources:
            return []
        return [self._source_to_audio_source(s) for s in sources]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def get_stream_details(self, source_id: str, queue_id: str) -> StreamDetails:
        """
        Return StreamDetails using the source's native format.

        32 and 24-bit sources stream as raw PCM to avoid FLAC's 24-bit cap
        and eliminate the encode step.  16-bit sources use FLAC.

        volume_normalization_mode is intentionally omitted: the MA core
        auto-disables normalization for MediaType.AUDIO_SOURCE
        (helpers/audio.py get_normalization_mode() returns DISABLED
        unconditionally — "live/realtime: upstream producer owns loudness").
        Setting it explicitly here would be redundant.
        """
        sources = await self._list_pa_sources()
        source = next((s for s in (sources or []) if s.name == source_id), None)
        if source is None:
            raise MediaNotFoundError(f"PA source not found: {source_id}")

        sample_rate = source.sample_rate or _DEFAULT_SAMPLE_RATE
        bit_depth = source.bit_depth or _DEFAULT_BIT_DEPTH
        channels = source.channels or _DEFAULT_CHANNELS

        # Use PCM for 24/32-bit sources: no encode step, exact bit depth
        # preserved in the signal chain. FLAC for 16-bit (lossless, framed).
        content_type = ContentType.PCM_S32LE if bit_depth >= 24 else ContentType.FLAC

        return StreamDetails(
            provider=self.instance_id,
            item_id=source_id,
            audio_format=AudioFormat(
                content_type=content_type,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
            ),
            stream_type=StreamType.CUSTOM,
            media_type=MediaType.AUDIO_SOURCE,
        )

    async def get_audio_stream(
        self,
        streamdetails: StreamDetails,
        seek_position: int = 0,
    ) -> AsyncGenerator[bytes]:
        """
        Capture audio from the PA source and yield encoded bytes.

        For PCM_S32LE: raw 32-bit little-endian PCM, no encode overhead.
        For FLAC: compression_level 0, lossless, minimal CPU cost.

        The MA core wraps this generator (StreamType.CUSTOM +
        MediaType.AUDIO_SOURCE) with a silence-keepalive: if the generator
        stops yielding (e.g. ffmpeg exits or the source goes silent), the
        core inserts silence frames at the declared PCM format and keeps
        the downstream player connected.
        """
        source_name = streamdetails.item_id
        env = self._build_pa_env()
        fmt = streamdetails.audio_format

        # Target ~10ms PA fragment size to reduce capture-side latency.
        bytes_per_ms = (fmt.sample_rate * (fmt.bit_depth // 8) * fmt.channels) // 1000
        fragment_size = max(bytes_per_ms * 10, 512)

        if fmt.content_type == ContentType.PCM_S32LE:
            # Raw PCM: no encode step, exact native bit depth, lowest latency.
            codec_args: list[str] = ["-sample_fmt", "s32", "-f", "s32le"]
        else:
            # FLAC for 16-bit sources: lossless and self-framing.
            codec_args = [
                "-sample_fmt",
                "s16",
                "-c:a",
                "flac",
                "-compression_level",
                "0",
                "-f",
                "flac",
            ]

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

        except (asyncio.CancelledError, GeneratorExit):  # fmt: skip
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

    def _source_to_audio_source(self, source: _PASource) -> AudioSource:
        """Convert a _PASource to an MA AudioSource item."""
        bit_depth = source.bit_depth or _DEFAULT_BIT_DEPTH
        content_type = ContentType.PCM_S32LE if bit_depth >= 24 else ContentType.FLAC

        return AudioSource(
            item_id=source.name,
            provider=self.instance_id,
            name=source.display_label,
            provider_mappings={
                ProviderMapping(
                    item_id=source.name,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    audio_format=AudioFormat(
                        content_type=content_type,
                        sample_rate=source.sample_rate or _DEFAULT_SAMPLE_RATE,
                        bit_depth=bit_depth,
                        channels=source.channels or _DEFAULT_CHANNELS,
                    ),
                )
            },
            can_play_pause=False,
            can_seek=False,
            can_next_previous=False,
            # MA can always initiate capture on demand: ffmpeg reads from the PA
            # source whenever the user selects it from Live Inputs.  This is
            # distinct from passive receivers (AirPlay, Spotify Connect) where
            # an external device must initiate the session first.
            can_initiate=True,
            # PA sources support multiple concurrent readers (ffmpeg can open
            # the same source from several consumers simultaneously).
            exclusive=False,
            allow_external_trigger=False,
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

    Full output (not ``pactl list sources short``) is used to obtain both
    human-readable descriptions and the PA server sample rate.  Using the
    server rate rather than the raw hardware rate ensures source and output
    sink rates match, avoiding MA-side resampling through the swr fallback
    when libsoxr is unavailable.
    """
    env = dict(os.environ)
    resolved = pa_server or os.environ.get("PULSE_SERVER", "") or _probe_pa_socket()
    if resolved:
        env["PULSE_SERVER"] = resolved

    full_out = await _pactl_output(["pactl", "list", "sources"], env)
    if full_out is None:
        return None
    sources = _parse_pactl_sources_full(full_out)
    if not include_monitors:
        sources = [s for s in sources if not s.is_monitor]
    return sources


def _parse_pactl_sources_full(output: str) -> list[_PASource]:
    """
    Parse ``pactl list sources`` full output into _PASource objects.

    Uses the PA server sample rate (which may be higher than the hardware
    rate due to the server clock being driven by the primary output device).
    This keeps source and output sink rates aligned, avoiding MA-side
    resampling through the swr fallback when libsoxr is unavailable.
    Also extracts human-readable descriptions for display names.
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


async def _pactl_output(
    cmd: list[str],
    env: dict[str, str],
) -> str | None:
    """Run a pactl command and return stdout as a string, or None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
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
    return stdout.decode(errors="replace")


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
