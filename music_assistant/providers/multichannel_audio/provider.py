"""Multichannel Audio player provider for Music Assistant."""

from __future__ import annotations

import ctypes
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from music_assistant.models.player_provider import PlayerProvider

from .constants import (
    MULTICHANNEL_CHANNELS,
    MULTICHANNEL_LAYOUT_51,
)
from .player import MultiChannelPlayer, get_player_uuid

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest
    from music_assistant.mass import MusicAssistant


class MultiChannelAudioProvider(PlayerProvider):
    """Player provider that streams 5.1/7.1 surround PCM to a PulseAudio surround sink."""

    _players: dict[str, MultiChannelPlayer]

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        try:
            ctypes.CDLL("libpulse-simple.so.0")
        except OSError as err:
            raise RuntimeError(
                "libpulse-simple.so.0 not found — is PulseAudio installed?"
            ) from err
        self._players = {}

    async def loaded_in_mass(self) -> None:
        """Handle provider fully loaded in Music Assistant."""
        await self._discover_and_register()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/removal of the provider."""
        for player in list(self._players.values()):
            with suppress(Exception):
                await player.stop_stream()
        self._players.clear()

    async def _discover_and_register(self) -> None:
        """Enumerate PulseAudio surround sinks and register players."""
        from .pa_sink import enumerate_surround_sinks  # noqa: PLC0415

        try:
            sinks: list[dict[str, Any]] = await self.mass.loop.run_in_executor(
                None, enumerate_surround_sinks
            )
        except Exception as err:
            self.logger.warning("Failed to enumerate surround sinks: %s", err)
            return

        if not sinks:
            self.logger.info("No multichannel PulseAudio sinks found")
            return

        self.logger.info("Found %d multichannel sink(s)", len(sinks))

        for sink in sinks:
            sink_name: str = sink["pa_sink_name"]
            player_id = get_player_uuid(sink_name)

            if player_id in self._players:
                continue

            layout: str = sink.get("layout", MULTICHANNEL_LAYOUT_51)
            channels: int = MULTICHANNEL_CHANNELS[layout]

            player = MultiChannelPlayer(
                provider=self,
                player_id=player_id,
                sink_name=sink_name,
                display_name=sink.get("name", sink_name),
                channels=channels,
                layout=layout,
                channel_map=sink["channel_map"],
                sample_rate=sink["sample_rate"],
                bit_depth=sink["bit_depth"],
                is_remap=sink.get("is_remap", False),
            )
            await player.restore_state()
            await player.apply_hardware_ceiling()
            await self.mass.players.register_or_update(player)
            self._players[player_id] = player
            self.logger.info(
                "Registered multichannel player: %s (%s, %dch, %dHz, %dbit)",
                sink_name,
                layout,
                channels,
                sink["sample_rate"],
                sink["bit_depth"],
            )

    async def cmd_volume_set(self, player_id: str, volume_level: int) -> None:
        """Set volume level (0-100) for the given player."""
        if player := self._players.get(player_id):
            await player.volume_set(volume_level)

    async def cmd_volume_mute(self, player_id: str, muted: bool) -> None:
        """Mute/unmute the given player."""
        if player := self._players.get(player_id):
            await player.volume_mute(muted)

    async def cmd_stop(self, player_id: str) -> None:
        """Send stop command to the given player."""
        if player := self._players.get(player_id):
            await player.stop_stream()
