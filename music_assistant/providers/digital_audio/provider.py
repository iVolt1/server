"""S/PDIF Audio Out — provider registration."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from music_assistant_models.enums import PlayerFeature, PlayerState, PlayerType
from music_assistant_models.player import DeviceInfo, Player

from music_assistant.models.player_provider import PlayerProvider

from .constants import (
    CONF_ENCODING_FORMAT,
    CONF_PA_SINK_NAME,
    ENCODING_MAX_CHANNELS,
    UUID_NAMESPACE,
)
from .player import SPDIFPlayer

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant_models.player_queue import PlayerQueue

    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)

# IEC 61937 streams are always delivered as 2-channel stereo to the PA sink
# (the encoded bitstream is multiplexed into a stereo PCM frame).
SPDIF_STREAM_CHANNELS = 2
SPDIF_STREAM_SAMPLE_RATE = 48000  # standard IEC 61937 rate
SPDIF_STREAM_BIT_DEPTH = 16  # IEC 61937 uses 16-bit containers


class SPDIFAudioProvider(PlayerProvider):
    """Player provider that delivers IEC 61937 encoded audio to a PA IEC958 sink."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set,
    ) -> None:
        super().__init__(mass, manifest, config)
        self._supported_features = supported_features
        self._player_id: str | None = None
        self._player: SPDIFPlayer = SPDIFPlayer(self)

    # ------------------------------------------------------------------
    # PlayerProvider interface
    # ------------------------------------------------------------------

    async def handle_async_init(self) -> None:
        """Async initialisation — register the player."""
        await self._register_player()

    async def unload(self, is_removed: bool = False) -> None:
        """Unload provider."""
        await self._player.stop()
        if self._player_id and self._player_id in self.mass.players:
            await self.mass.players.remove(self._player_id, is_removed)

    # ------------------------------------------------------------------
    # MA player command callbacks
    # ------------------------------------------------------------------

    async def cmd_play_media(
        self,
        player_id: str,
        queue: "PlayerQueue",  # noqa: F821
    ) -> None:
        """MA calls this when a queue item should start playing."""
        await self._player.play_media(player_id, queue)
        self.mass.players.update_state(player_id, PlayerState.PLAYING)

    async def cmd_stop(self, player_id: str) -> None:
        """MA calls this to stop playback."""
        await self._player.stop()
        self.mass.players.update_state(player_id, PlayerState.IDLE)

    async def cmd_pause(self, player_id: str) -> None:
        """MA calls this to pause playback."""
        await self._player.pause()
        self.mass.players.update_state(player_id, PlayerState.PAUSED)

    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """
        Set volume via PulseAudio sink volume.

        *volume_level* is 0–100; PA scale is 0–65536 (PA_VOLUME_NORM).
        We set the sink volume so all streams on it are affected equally.
        This is consistent with ``multichannel_audio``'s volume approach.
        """
        sink_name: str = self.config.get_value(CONF_PA_SINK_NAME)
        pa_vol = int(volume_level / 100 * 65536)
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl",
                "set-sink-volume",
                sink_name,
                str(pa_vol),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=3)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("pactl set-sink-volume failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _query_sink_sample_rate(self, sink_name: str) -> int:
        """
        Query the native sample rate of *sink_name* via ``pactl``.

        Returns the detected rate, or ``SPDIF_STREAM_SAMPLE_RATE`` as fallback.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pactl",
                "list",
                "sinks",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("pactl list sinks failed: %s", exc)
            return SPDIF_STREAM_SAMPLE_RATE

        in_target = False
        for line in stdout.decode(errors="replace").splitlines():
            if f"Name: {sink_name}" in line:
                in_target = True
            if in_target and "Sample Specification:" in line:
                # e.g.  "Sample Specification: s16le 2ch 48000Hz"
                for token in line.split():
                    if token.endswith("Hz"):
                        try:
                            return int(token[:-2])
                        except ValueError:
                            pass
                break

        LOGGER.debug(
            "Could not parse sample rate for sink %s, using default %d Hz",
            sink_name,
            SPDIF_STREAM_SAMPLE_RATE,
        )
        return SPDIF_STREAM_SAMPLE_RATE

    async def _register_player(self) -> None:
        """Create and register the MA player for the configured S/PDIF sink."""
        sink_name: str = self.config.get_value(CONF_PA_SINK_NAME)
        encoding: str = self.config.get_value(CONF_ENCODING_FORMAT)

        if not sink_name:
            LOGGER.warning("No PA sink configured — skipping player registration")
            return

        sample_rate = await self._query_sink_sample_rate(sink_name)
        max_channels = ENCODING_MAX_CHANNELS.get(encoding, 6)

        player_id = str(uuid.uuid5(UUID_NAMESPACE, sink_name))
        self._player_id = player_id

        player = Player(
            player_id=player_id,
            provider=self.domain,
            type=PlayerType.PLAYER,
            name=f"S/PDIF {sink_name}",
            available=True,
            powered=False,
            device_info=DeviceInfo(model="S/PDIF IEC 61937", manufacturer="PulseAudio"),
            supported_features={
                PlayerFeature.POWER,
                PlayerFeature.VOLUME_SET,
                PlayerFeature.PAUSE,
            },
            # IEC 61937 streams arrive at the sink as 2ch stereo containers,
            # but MA sees the logical channel count of the encoded format.
            channel_count=max_channels,
            sample_rate=sample_rate,
            bit_depth=SPDIF_STREAM_BIT_DEPTH,
        )

        await self.mass.players.register(player)
        LOGGER.info(
            "Registered S/PDIF player '%s' — encoding=%s max_ch=%d sample_rate=%d",
            sink_name,
            encoding,
            max_channels,
            sample_rate,
        )
