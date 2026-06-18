"""S/PDIF Audio Player — plain PCM for stereo, AC3/IEC61937 passthrough for 5.1."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING

from music_assistant_models.enums import (
    ContentType,
    IdentifierType,
    PlayerFeature,
    PlayerType,
    PlaybackState,
)
from music_assistant_models.player import DeviceInfo

from music_assistant.models.player import Player, PlayerMedia

from .constants import (
    CACHE_CATEGORY_PREV_STATE,
    DEFAULT_PLAYER_VOLUME,
    DEVICE_UUID_NAMESPACE,
    SPDIF_BIT_DEPTH,
    SPDIF_CARRIER_CHANNELS,
    SPDIF_MAX_CONTENT_CHANNELS,
    SPDIF_PASSTHROUGH_THRESHOLD_CHANNELS,
    SPDIF_SAMPLE_RATE,
)

if TYPE_CHECKING:
    from .provider import SpdifAudioProvider


def get_player_uuid(pa_sink_name: str) -> str:
    """
    Generate a stable UUID for an S/PDIF player from its PA sink name.

    :param pa_sink_name: The PulseAudio sink name.
    """
    return str(uuid.uuid5(DEVICE_UUID_NAMESPACE, pa_sink_name))


class SpdifAudioPlayer(Player):
    """
    Player for an optical/coax S/PDIF output via PulseAudio.

    Standard S/PDIF is a 2-channel digital carrier — it cannot carry discrete
    multichannel PCM. Two distinct paths are used depending on real source
    channel count, decided per-track from streamdetails:

    - Stereo (<=2ch): sent as plain 16-bit PCM via ffmpeg -f pulse, identical
      in spirit to digital_audio's direct-sink approach.
    - Surround (>2ch, up to 5.1): encoded to Dolby Digital (AC3) and wrapped
      in IEC 61937 framing, which disguises the compressed bitstream as plain
      16-bit stereo PCM at the carrier rate. The receiver's optical input
      recognizes the IEC 61937 sync pattern and decodes it back to 5.1
      internally — to PulseAudio, it is just an opaque 2-channel PCM stream.

    Volume control
    --------------
    Applied via ffmpeg's `-af volume=` filter on the decoded PCM, BEFORE
    either output path's encode step — safe on both paths, including AC3
    passthrough, since the filter only ever touches raw samples. AC3 encoding
    doesn't care whether its input was already gain-adjusted; the resulting
    bitstream is just as valid either way. What would NOT be safe is scaling
    AFTER encoding — touching the compressed bytes themselves corrupts the
    IEC 61937 sync pattern.

    A static ffmpeg filter value can't be changed on a running process, so a
    volume change kills the active segment and relaunches it with the new
    filter value and a `-ss` seek to the position it was at — see
    _playback_loop/_current_position/volume_set. This causes a brief
    (sub-second) audio gap on each volume change, similar to the click some
    standalone AVRs have on their own digital volume steps.

    NOTE: this relies on MA's resolve_stream_url output actually honoring
    -ss seeking. If it's a live single-pass flow transcode without range
    support, -ss may silently fail and restart from 0 instead — verify this
    in practice before relying on it.

    The PA sink itself is kept locked at 100% (see apply_restored_volume) so
    there is exactly one place gain is ever applied — the pre-encode filter —
    rather than risking double-attenuation between the sink and the filter.

    VOLUME_MUTE is handled completely separately, via pactl sink mute — a
    hard gate, not a scale, and instant (no restart needed).
    """

    def __init__(
        self,
        provider: SpdifAudioProvider,
        player_id: str,
        sink_name: str,
        display_name: str,
        sample_rate: int = SPDIF_SAMPLE_RATE,
        ac3_bitrate: str = "640k",
    ) -> None:
        """
        Initialize the S/PDIF Audio player.

        :param provider: The S/PDIF Audio provider instance.
        :param player_id: Stable player ID (UUID5 derived from sink name).
        :param sink_name: PulseAudio sink name for the S/PDIF output.
        :param display_name: Human-readable name shown in the MA UI.
        :param sample_rate: Carrier sample rate, always 48000 in practice —
            AC3 does not support 96kHz and standard S/PDIF locks to 48kHz.
        :param ac3_bitrate: Encode bitrate string for ffmpeg's -b:a, e.g. '640k'.
        """
        super().__init__(provider, player_id)
        self._attr_type = PlayerType.PLAYER
        self._attr_name = display_name
        self._attr_available = True
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.PAUSE,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
        }
        self._attr_device_info = DeviceInfo(
            model=display_name,
            manufacturer="S/PDIF Audio",
        )
        self._attr_device_info.add_identifier(IdentifierType.UUID, player_id)
        self._attr_can_group_with = set()
        self._attr_volume_level = DEFAULT_PLAYER_VOLUME

        self._sink_name = sink_name
        self.sample_rate = sample_rate
        self.bit_depth = SPDIF_BIT_DEPTH
        self.channels = SPDIF_MAX_CONTENT_CHANNELS
        self._ac3_bitrate = ac3_bitrate

        self._playback_task: asyncio.Task[None] | None = None
        self._paused = False

        # --- Volume-triggered restart-with-seek state ---
        # See _playback_loop, _current_position, and volume_set for how these
        # combine to let a volume change apply without losing track position.
        self._active_procs: list[asyncio.subprocess.Process] = []
        self._current_url: str | None = None
        self._current_source_channels: int = 0
        self._restart_requested: bool = False
        self._segment_start_offset: float = 0.0  # seconds into the track at segment start
        self._segment_start_monotonic: float = 0.0  # wall clock when segment's proc spawned
        self._paused_accum: float = 0.0  # total paused duration within the current segment
        self._pause_started_monotonic: float | None = None  # set while currently paused

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs to be polled for state updates."""
        return False

    @property
    def supported_sample_rates(self) -> list[tuple[int, int]]:
        """
        Declare only the S/PDIF carrier rate this player supports.

        Anchors MA's flow encoder to 48kHz/16bit regardless of source —
        matches both AC3's rate ceiling and the carrier's fixed format.
        """
        return [(self.sample_rate, self.bit_depth)]

    # --- MA mandatory player interface ---

    async def play_media(self, media: PlayerMedia) -> None:
        """Handle PLAY_MEDIA command."""
        await self._stop_playback()
        url = await self._provider.mass.streams.resolve_stream_url(self.player_id, media)
        self.logger.info("Starting S/PDIF playback from %s", url)

        # Real per-track channel count decides plain-PCM vs AC3-passthrough path.
        source_channels = self.channels
        try:
            queue = self.mass.player_queues.get_active_queue(self.player_id)
            if queue and queue.current_item and queue.current_item.streamdetails:
                sd = queue.current_item.streamdetails
                self.logger.debug(
                    "streamdetails: channels=%d sample_rate=%d uri=%s",
                    sd.audio_format.channels, sd.audio_format.sample_rate, sd.uri,
                )
                if sd.audio_format.channels > 0:
                    source_channels = sd.audio_format.channels
        except Exception as err:
            self.logger.debug("Could not read streamdetails: %s", err)

        self._attr_current_media = media
        self._attr_playback_state = PlaybackState.PLAYING
        self._paused = False
        self.update_state()
        self._playback_task = self.mass.create_task(
            self._playback_loop(url, source_channels)
        )

    async def stop(self) -> None:
        """Handle STOP command."""
        await self._stop_playback()
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_current_media = None
        self.update_state()

    async def pause(self) -> None:
        """Handle PAUSE command."""
        self._paused = True
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def play(self) -> None:
        """Handle PLAY/resume command."""
        self._paused = False
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    # --- Playback dispatch with volume-triggered restart support ---

    async def _playback_loop(self, url: str, source_channels: int) -> None:
        """
        Run playback segments, relaunching with a seek when volume changes mid-track.

        Each iteration is one ffmpeg/pacat "segment". volume_set() kills the
        active segment's process(es) and sets _restart_requested; this loop
        then computes the elapsed position via _current_position(), advances
        _segment_start_offset to it, and immediately relaunches with the new
        volume baked into a fresh -af filter and a -ss seek — so the change
        applies without losing playback position (seek support permitting —
        see the class docstring's caveat about MA's stream URL).
        """
        self._current_url = url
        self._current_source_channels = source_channels
        self._segment_start_offset = 0.0
        try:
            while True:
                self._restart_requested = False
                self._paused_accum = 0.0
                self._pause_started_monotonic = None
                self._segment_start_monotonic = time.monotonic()
                if source_channels <= SPDIF_PASSTHROUGH_THRESHOLD_CHANNELS:
                    await self._play_direct_pcm(url, self._segment_start_offset)
                else:
                    await self._play_ac3_passthrough(
                        url, source_channels, self._segment_start_offset
                    )
                if self._restart_requested:
                    self._segment_start_offset = self._current_position()
                    continue
                break
        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.error("Playback error: %s", err, exc_info=True)
        finally:
            self._active_procs = []
            self._attr_playback_state = PlaybackState.IDLE
            self._attr_current_media = None
            self._paused = False
            self.update_state()
            if self._playback_task is asyncio.current_task():
                self._playback_task = None

    def _current_position(self) -> float:
        """Return current elapsed audio position in seconds, excluding paused time.

        Used to compute the -ss seek offset when a volume change restarts the
        active segment, so playback resumes from where it left off.
        """
        if self._segment_start_monotonic == 0.0:
            return self._segment_start_offset
        now = time.monotonic()
        paused_total = self._paused_accum
        if self._pause_started_monotonic is not None:
            paused_total += now - self._pause_started_monotonic
        elapsed = now - self._segment_start_monotonic - paused_total
        return self._segment_start_offset + max(0.0, elapsed)

    def _volume_factor(self) -> float:
        """Return current volume as a linear 0.0-1.0 factor for ffmpeg's volume filter."""
        level = self._attr_volume_level if self._attr_volume_level is not None else (
            DEFAULT_PLAYER_VOLUME
        )
        return max(0.0, min(level, 100)) / 100.0

    # --- Path 1: plain stereo PCM (no encoding) ---

    async def _play_direct_pcm(self, url: str, offset: float) -> None:
        """Send stereo content directly as 16-bit PCM — standard optical, no encoding.

        Volume applied via -af volume on the decoded PCM. offset seeks the
        input to resume mid-track after a volume-triggered restart.
        """
        url_fmt_str = url.rsplit(".", 1)[-1] if "." in url.rsplit("/", 1)[-1] else ""
        url_content_type = ContentType.try_parse(url_fmt_str)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        seek_args = ["-ss", f"{offset:.3f}"] if offset > 0 else []

        if url_content_type.is_pcm():
            input_args = [
                "-thread_queue_size", "4096",
                "-f", str(url_content_type.value),
                "-ar", str(self.sample_rate),
                "-ac", str(SPDIF_CARRIER_CHANNELS),
                *seek_args,
                "-i", url,
            ]
        else:
            input_args = ["-thread_queue_size", "4096", *seek_args, "-i", url]

        cmd = [
            ffmpeg_bin, "-hide_banner", "-nostdin",
            *input_args,
            "-af", f"volume={self._volume_factor():.4f}",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(SPDIF_CARRIER_CHANNELS),
            "-f", "pulse",
            "-buffer_duration", "500",
            "-name", f"music-assistant-{self._sink_name}",
            self._sink_name,
        ]
        self.logger.debug("ffmpeg (direct PCM) cmd: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        self._active_procs = [proc]
        self.logger.info(
            "S/PDIF direct PCM playback started: sink=%s pid=%d offset=%.1fs volume=%.2f",
            self._sink_name, proc.pid, offset, self._volume_factor(),
        )
        stderr_task = self.mass.create_task(self._drain_stderr(proc, "pcm"))
        try:
            await self._monitor_single_process(proc)
        finally:
            await self._cleanup_process(proc, stderr_task)

    # --- Path 2: AC3 encode + IEC 61937 wrap + raw passthrough write ---

    async def _play_ac3_passthrough(
        self, url: str, source_channels: int, offset: float
    ) -> None:
        """
        Encode surround content to Dolby Digital and write it as a passthrough
        bitstream disguised as 2-channel 16-bit PCM.

        Two processes connected by a raw OS pipe (zero-copy, kernel-mediated):
          1. ffmpeg: decode source -> [volume filter, PCM] -> encode AC3 ->
             wrap as IEC 61937 -> stdout
          2. pacat: read raw bytes from stdin -> write bit-perfect to PA sink

        Volume is applied via -af volume BEFORE the AC3 encoder (-c:a ac3) —
        safe, since it only ever touches raw PCM. pacat never sees anything
        but the final compressed bitstream bytes and never modifies them.

        offset seeks the input to resume mid-track after a volume-triggered
        restart. Pause only needs to SIGSTOP the pacat consumer — once it
        stops draining, the OS pipe buffer fills and ffmpeg blocks naturally
        on its next write.
        """
        url_fmt_str = url.rsplit(".", 1)[-1] if "." in url.rsplit("/", 1)[-1] else ""
        url_content_type = ContentType.try_parse(url_fmt_str)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        pacat_bin = shutil.which("pacat") or "pacat"
        seek_args = ["-ss", f"{offset:.3f}"] if offset > 0 else []

        if url_content_type.is_pcm():
            input_args = [
                "-thread_queue_size", "4096",
                "-f", str(url_content_type.value),
                "-ar", str(self.sample_rate),
                "-ac", str(source_channels),
                *seek_args,
                "-i", url,
            ]
        else:
            input_args = ["-thread_queue_size", "4096", *seek_args, "-i", url]

        cmd_encode = [
            ffmpeg_bin, "-hide_banner", "-nostdin",
            *input_args,
            "-af", f"volume={self._volume_factor():.4f}",
            "-c:a", "ac3",
            "-b:a", self._ac3_bitrate,
            "-ar", str(SPDIF_SAMPLE_RATE),
            "-ac", str(SPDIF_MAX_CONTENT_CHANNELS),
            "-f", "spdif",
            "pipe:1",
        ]
        cmd_play = [
            pacat_bin,
            "--playback",
            f"--device={self._sink_name}",
            f"--rate={SPDIF_SAMPLE_RATE}",
            f"--channels={SPDIF_CARRIER_CHANNELS}",
            "--format=s16le",
            "--raw",
            "--latency-msec=500",
            f"--client-name=music-assistant-{self._sink_name}",
        ]
        self.logger.debug(
            "ffmpeg (AC3 encode) cmd: %s  |  pacat cmd: %s",
            " ".join(cmd_encode), " ".join(cmd_play),
        )

        read_fd, write_fd = os.pipe()
        proc_encode: asyncio.subprocess.Process | None = None
        proc_play: asyncio.subprocess.Process | None = None
        try:
            proc_encode = await asyncio.create_subprocess_exec(
                *cmd_encode, stdout=write_fd, stderr=asyncio.subprocess.PIPE,
            )
            os.close(write_fd)
            proc_play = await asyncio.create_subprocess_exec(
                *cmd_play, stdin=read_fd, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            os.close(read_fd)
            self._active_procs = [proc_encode, proc_play]

            self.logger.info(
                "S/PDIF AC3 passthrough started: sink=%s %dch source, bitrate=%s, "
                "offset=%.1fs volume=%.2f encode_pid=%d play_pid=%d",
                self._sink_name, source_channels, self._ac3_bitrate,
                offset, self._volume_factor(), proc_encode.pid, proc_play.pid,
            )

            stderr_tasks = [
                self.mass.create_task(self._drain_stderr(proc_encode, "encode")),
                self.mass.create_task(self._drain_stderr(proc_play, "spdif")),
            ]
            try:
                await self._monitor_passthrough_pair(proc_encode, proc_play)
            finally:
                for t in stderr_tasks:
                    if not t.done():
                        t.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await t
        finally:
            for proc in (proc_play, proc_encode):
                if proc is not None and proc.returncode is None:
                    with suppress(ProcessLookupError, OSError):
                        proc.kill()
                    with suppress(Exception):
                        await proc.wait()
            for fd in (read_fd, write_fd):
                with suppress(OSError):
                    os.close(fd)

    async def _monitor_passthrough_pair(
        self,
        proc_encode: asyncio.subprocess.Process,
        proc_play: asyncio.subprocess.Process,
    ) -> None:
        """Poll both processes, applying pause via SIGSTOP/SIGCONT to pacat only,
        and tracking paused duration for accurate _current_position()."""
        _was_paused = False
        if self._paused:
            with suppress(ProcessLookupError, OSError):
                proc_play.send_signal(signal.SIGSTOP)
            _was_paused = True
            self._pause_started_monotonic = time.monotonic()
        while proc_play.returncode is None:
            if self._paused and not _was_paused:
                with suppress(ProcessLookupError, OSError):
                    proc_play.send_signal(signal.SIGSTOP)
                _was_paused = True
                self._pause_started_monotonic = time.monotonic()
            elif not self._paused and _was_paused:
                with suppress(ProcessLookupError, OSError):
                    proc_play.send_signal(signal.SIGCONT)
                _was_paused = False
                if self._pause_started_monotonic is not None:
                    self._paused_accum += time.monotonic() - self._pause_started_monotonic
                    self._pause_started_monotonic = None
            await asyncio.sleep(0.1)
        if _was_paused:
            with suppress(ProcessLookupError, OSError):
                proc_play.send_signal(signal.SIGCONT)
            if self._pause_started_monotonic is not None:
                self._paused_accum += time.monotonic() - self._pause_started_monotonic
                self._pause_started_monotonic = None
        if proc_play.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
            self.logger.warning(
                "pacat exited unexpectedly: returncode=%s sink=%s",
                proc_play.returncode, self._sink_name,
            )

    # --- Shared process helpers ---

    async def _monitor_single_process(self, proc: asyncio.subprocess.Process) -> None:
        """Poll a single ffmpeg process, applying pause via SIGSTOP/SIGCONT,
        and tracking paused duration for accurate _current_position()."""
        _was_paused = False
        if self._paused:
            with suppress(ProcessLookupError, OSError):
                proc.send_signal(signal.SIGSTOP)
            _was_paused = True
            self._pause_started_monotonic = time.monotonic()
        while proc.returncode is None:
            if self._paused and not _was_paused:
                with suppress(ProcessLookupError, OSError):
                    proc.send_signal(signal.SIGSTOP)
                _was_paused = True
                self._pause_started_monotonic = time.monotonic()
            elif not self._paused and _was_paused:
                with suppress(ProcessLookupError, OSError):
                    proc.send_signal(signal.SIGCONT)
                _was_paused = False
                if self._pause_started_monotonic is not None:
                    self._paused_accum += time.monotonic() - self._pause_started_monotonic
                    self._pause_started_monotonic = None
            await asyncio.sleep(0.1)
        if _was_paused:
            with suppress(ProcessLookupError, OSError):
                proc.send_signal(signal.SIGCONT)
            if self._pause_started_monotonic is not None:
                self._paused_accum += time.monotonic() - self._pause_started_monotonic
                self._pause_started_monotonic = None
        if proc.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
            self.logger.warning(
                "ffmpeg exited unexpectedly: returncode=%s sink=%s",
                proc.returncode, self._sink_name,
            )

    async def _cleanup_process(
        self, proc: asyncio.subprocess.Process, stderr_task: asyncio.Task[None]
    ) -> None:
        """Cancel stderr drain and ensure the process is fully terminated."""
        if not stderr_task.done():
            stderr_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await stderr_task
        if proc.returncode is None:
            with suppress(ProcessLookupError, OSError):
                proc.kill()
            with suppress(Exception):
                await proc.wait()

    async def _drain_stderr(self, proc: asyncio.subprocess.Process, label: str) -> None:
        """Consume a process's stderr to prevent pipe buffer saturation; log at DEBUG."""
        if proc.stderr is None:
            return
        try:
            async for line_bytes in proc.stderr:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        "%s: %s", label, line_bytes.decode(errors="replace").rstrip()
                    )
        except asyncio.CancelledError:
            pass
        except Exception as err:
            self.logger.debug("%s stderr drain: %s", label, err)

    async def _stop_playback(self) -> None:
        """Cancel and await the playback task if running."""
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._playback_task
        self._playback_task = None

    async def stop_stream(self) -> None:
        """Stop streaming — alias for stop() used by provider unload."""
        await self._stop_playback()

    # --- Volume control (restart-with-seek) and mute control (instant gate) ---

    async def volume_set(self, volume_level: int) -> None:
        """Set volume via a pre-encode ffmpeg filter.

        Restarts the active segment with a seek to the current position so
        the new gain applies without losing playback position. Safe even for
        the AC3 passthrough path — see class docstring. There is a small
        (well under a second) audio gap during the restart.
        """
        self._attr_volume_level = volume_level
        await self._save_state()
        if self._playback_task and not self._playback_task.done() and self._active_procs:
            self._restart_requested = True
            for proc in list(self._active_procs):
                if proc.returncode is None:
                    with suppress(ProcessLookupError, OSError):
                        proc.kill()
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Mute/unmute via pactl sink mute — a hard gate, never sample scaling.

        Unlike volume_set, this is instant and needs no restart.
        """
        self._attr_volume_muted = muted
        await self.mass.loop.run_in_executor(None, self._pactl_set_mute, muted)
        await self._save_state()
        self.update_state()

    def _pactl_set_mute(self, muted: bool) -> None:
        """Blocking: mute or unmute the sink via pactl."""
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["pactl", "set-sink-mute", self._sink_name, "1" if muted else "0"],
            timeout=3, check=False, capture_output=True,
        )

    def _pactl_force_unity_volume(self) -> None:
        """Blocking: force sink volume to 100% — called once at registration.

        The sink stays fixed at unity; actual user-facing volume lives
        entirely in the pre-encode ffmpeg filter (_volume_factor), applied
        fresh at the start of every playback segment. This keeps gain
        applied in exactly one place rather than risking double-attenuation
        between the sink and the filter.
        """
        import subprocess  # noqa: PLC0415

        subprocess.run(
            ["pactl", "set-sink-volume", self._sink_name, "100%"],
            timeout=3, check=False, capture_output=True,
        )

    async def apply_restored_volume(self) -> None:
        """Force unity sink volume and apply restored mute state on startup."""
        await self.mass.loop.run_in_executor(None, self._pactl_force_unity_volume)
        if self._attr_volume_muted:
            await self.mass.loop.run_in_executor(None, self._pactl_set_mute, True)
        self.logger.debug(
            "Sink volume forced to 100%% (gain handled via encode filter), "
            "restored level=%d%% muted=%s on sink %s",
            self._attr_volume_level, self._attr_volume_muted, self._sink_name,
        )

    # --- State persistence ---

    async def restore_state(self) -> None:
        """Restore cached volume/mute state from a previous session."""
        if last_state := await self.mass.cache.get(
            key=self.player_id,
            provider=self._provider.instance_id,
            category=CACHE_CATEGORY_PREV_STATE,
        ):
            self._attr_volume_muted = last_state[0]
            self._attr_volume_level = last_state[1]
        else:
            self._attr_volume_muted = False
            self._attr_volume_level = DEFAULT_PLAYER_VOLUME

    async def _save_state(self) -> None:
        """Persist current volume/mute state to cache."""
        await self.mass.cache.set(
            key=self.player_id,
            data=[self._attr_volume_muted, self._attr_volume_level],
            provider=self._provider.instance_id,
            category=CACHE_CATEGORY_PREV_STATE,
        )
