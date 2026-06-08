"""Multichannel Audio player provider for Music Assistant."""

from __future__ import annotations

import ctypes
from contextlib import suppress
from typing import TYPE_CHECKING

from music_assistant.models.player_provider import PlayerProvider

from .constants import (
    CONF_CHANNEL_MAP,
    CONF_CUSTOM_CHANNEL_MAP,
    CONF_MULTICHANNEL_LAYOUT,
    CONF_PA_SINK_NAME,
    MULTICHANNEL_CHANNELS,
    MULTICHANNEL_LAYOUT_51,
)
from .player import MultiChannelPlayer, get_player_uuid

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant


class MultiChannelAudioProvider(PlayerProvider):
    """Player provider that streams multichannel audio via stereo PA remap sink pairs."""

    _player: MultiChannelPlayer | None

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        try:
            ctypes.CDLL("libpulse-simple.so.0")
        except OSError as err:
            raise RuntimeError(
                "libpulse-simple.so.0 not found — is PulseAudio installed?"
            ) from err
        self._player = None

    async def loaded_in_mass(self) -> None:
        """Handle provider fully loaded in Music Assistant."""
        await self._register_player()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/removal of the provider."""
        if self._player:
            with suppress(Exception):
                await self._player.stop_stream()
            self._player = None

    async def _register_player(self) -> None:
        """Register a single multichannel player from provider configuration."""
        sink_name = str(self.config.get_value(CONF_PA_SINK_NAME) or "")
        if not sink_name:
            self.logger.warning("No PA sink configured — skipping player registration")
            return

        layout = str(self.config.get_value(CONF_MULTICHANNEL_LAYOUT) or MULTICHANNEL_LAYOUT_51)
        channels = MULTICHANNEL_CHANNELS[layout]
        player_id = get_player_uuid(sink_name)
        channel_map = str(self.config.get_value(CONF_CHANNEL_MAP) or "flac")
        custom_map = str(self.config.get_value(CONF_CUSTOM_CHANNEL_MAP) or "")

        # Query native format from the surround sink via pactl
        sample_rate, bit_depth, _ = await self.mass.loop.run_in_executor(
            None, _query_sink_format, sink_name
        )

        # Derive card name from sink name for stereo pair lookup.
        # e.g. "Creative_X_Fi_surround" -> "Creative_X_Fi"
        card_name = sink_name.replace("_surround", "")

        self._player = MultiChannelPlayer(
            provider=self,
            player_id=player_id,
            card_name=card_name,
            display_name=f"Multichannel Audio ({layout})",
            channels=channels,
            layout=layout,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channel_map=channel_map,
            custom_channel_map=custom_map,
        )
        await self._player.restore_state()
        await self._player.apply_restored_volume()
        await self.mass.players.register_or_update(self._player)
        self.logger.info(
            "Registered multichannel player: %s (%s, %dch, %dHz, %dbit) -> pairs: %s",
            sink_name,
            layout,
            channels,
            sample_rate,
            bit_depth,
            list(self._player._pair_sinks.keys()),
        )

    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Set volume level (0-100) for the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.volume_set(volume_level)

    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Mute/unmute the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.volume_mute(muted)

    async def cmd_stop(self, player_id: str) -> None:
        """Send stop command to the player."""
        if self._player and self._player.player_id == player_id:
            await self._player.stop_stream()


def _query_sink_format(sink_name: str) -> tuple[int, int, int]:
    """
    Query native sample rate, bit depth, and channel count for a PA sink via pactl.

    :param sink_name: The PulseAudio sink name.
    :returns: Tuple of (sample_rate, bit_depth, channels). Falls back to (48000, 16, 0).
    """
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

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
                if sink.get("name") == sink_name:
                    spec_str: str = sink.get("sample_specification", "")
                    parts = spec_str.split()
                    fmt = parts[0]
                    channels = int(parts[1].replace("ch", ""))
                    sample_rate = int(parts[2].replace("Hz", ""))
                    bit_depth = int(
                        "".join(filter(str.isdigit, fmt.split("le")[0].split("be")[0]))
                    )
                    return sample_rate, bit_depth, channels
    except Exception:
        pass
    return 48000, 16, 0
