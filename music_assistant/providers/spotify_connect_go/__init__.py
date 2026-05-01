"""
Spotify Connect Go plugin for Music Assistant.

This plugin uses go-librespot with its web interface for better control capabilities.
We tie a single player to a single Spotify Connect daemon.
The provider has multi instance support,
so multiple players can be linked to multiple Spotify Connect daemons.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import yaml
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, cast

import aiohttp
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    EventType,
    MediaType,
    PlayerFeature,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import PlayerMedia

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.helpers.process import AsyncProcess, check_output
from music_assistant.models.plugin import PluginProvider, PluginSource

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_MASS_PLAYER_ID = "mass_player_id"
CONF_SERVER_PORT = "server_port"
CONF_EXTERNAL_VOLUME = "external_volume"
CONNECT_ITEM_ID = "spotify_connect_go"

# Default server port for go-librespot web interface
DEFAULT_SERVER_PORT = 3678

SUPPORTED_FEATURES = {ProviderFeature.AUDIO_SOURCE}

async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SpotifyConnectGoProvider(mass, manifest, config)

async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    return (
        CONF_ENTRY_WARN_PREVIEW,
        ConfigEntry(
            key=CONF_MASS_PLAYER_ID,
            type=ConfigEntryType.STRING,
            label="Connected Music Assistant Player",
            description="Select the player for which you want to enable Spotify Connect Go.",
            multi_value=False,
            options=[ConfigValueOption(x.display_name, x.player_id) for x in mass.players],
            required=True,
        ),
        ConfigEntry(
            key=CONF_SERVER_PORT,
            type=ConfigEntryType.INTEGER,
            label="Web Interface Port",
            description="Port for the go-librespot web interface (default: 3678)",
            default_value=DEFAULT_SERVER_PORT,
            required=False,
        ),
        ConfigEntry(
            key=CONF_EXTERNAL_VOLUME,
            type=ConfigEntryType.BOOLEAN,
            label="External Volume Control",
            description=(
                "When enabled, volume is controlled by Music Assistant at the player level "
                "(recommended for sync groups). When disabled, the Spotify app volume slider "
                "controls playback volume."
            ),
            default_value=True,
            required=False,
        ),
        ConfigEntry(
            key="metadata_delay",
            type=ConfigEntryType.FLOAT,
            label="Metadata Delay (seconds)",
            description="Delay metadata updates to sync with audio playback (0-10 seconds). Adjust based on your buffer chain latency.",
            default_value=3.5,
            required=False,
            range=(0, 10),
        ),
    )


class SpotifyConnectGoProvider(PluginProvider):
    """Implementation of a Spotify Connect Go Plugin."""

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize MusicProvider."""
        super().__init__(mass, manifest, config)
        self.mass_player_id = cast("str", self.config.get_value(CONF_MASS_PLAYER_ID))
        self.server_port = cast(
            "int", self.config.get_value(CONF_SERVER_PORT) or DEFAULT_SERVER_PORT
        )
        self.external_volume = cast(
            "bool",
            self.config.get_value(CONF_EXTERNAL_VOLUME)
            if self.config.get_value(CONF_EXTERNAL_VOLUME) is not None
            else True,
        )
        self.cache_dir = os.path.join(self.mass.cache_path, self.instance_id)
        self.config_dir = os.path.join(self.cache_dir, "config")
        self._go_librespot_bin = "/media/bin/go-librespot"
        self._stop_called: bool = False
        self._runner_task: asyncio.Task | None = None
        self._websocket_task: asyncio.Task | None = None
        self._position_poll_task: asyncio.Task | None = None
        self._go_librespot_proc: AsyncProcess | None = None
        self._go_librespot_started = asyncio.Event()
        self.named_pipe = f"/tmp/{self.instance_id}"  # noqa: S108
        self._api_base_url = f"http://localhost:{self.server_port}"
        self._ws_url = f"ws://localhost:{self.server_port}/events"
        self._ws_session: aiohttp.ClientSession | None = None
        self._ws_connection = None
        self._pipe_fd: int | None = None
        self._active_player_id: str | None = None
        self._current_track_uri: str | None = None
        self._last_seek_time: float = 0.0
        self._metadata_update_task: asyncio.Task | None = None

        # Create the source details
        self._source_details = PluginSource(
            id=self.instance_id,
            name=self.manifest.name,
            passive=False,
            can_play_pause=True,
            can_seek=True,
            can_next_previous=True,
            audio_format=AudioFormat(
                content_type=ContentType.PCM_S16LE,
                codec_type=ContentType.PCM_S16LE,
                sample_rate=44100,
                bit_depth=16,
                channels=2,
            ),
            stream_type=StreamType.NAMED_PIPE,
            path=self.named_pipe,
            on_play=self._on_play_callback,
            on_pause=self._on_pause_callback,
            on_next=self._on_next_callback,
            on_previous=self._on_previous_callback,
            on_seek=self._on_seek_callback,
            on_volume=self._on_volume_callback,
            on_select=self._on_source_selected,
        )
        self._on_unload_callbacks: list[Callable[..., None]] = [
            self.mass.subscribe(
                self._on_mass_player_event,
                (EventType.PLAYER_ADDED, EventType.PLAYER_REMOVED),
                id_filter=self.mass_player_id,
            ),
        ]

    @property
    def supported_features(self) -> set[ProviderFeature]:
        """Return the features supported by this Provider."""
        return {ProviderFeature.AUDIO_SOURCE}

    def _add_seek_to_player(self, player_id: str) -> None:
        """Add PlayerFeature.SEEK to a player and invalidate its state cache."""
        player = self.mass.players.get_player(player_id)
        if player and PlayerFeature.SEEK not in player._attr_supported_features:
            player._attr_supported_features.add(PlayerFeature.SEEK)
            player.update_state()
            self.logger.debug("Added PlayerFeature.SEEK to player %s", player_id)

    def _trigger_update(self) -> None:
        """Trigger player update on the correct player — the one with in_use_by set."""
        player_id = self._source_details.in_use_by or self._active_player_id
        if player_id:
            self.mass.players.trigger_player_update(player_id)

    def _trigger_update(self) -> None:
        """Trigger player update on the correct player — the one with in_use_by set."""
        player_id = self._source_details.in_use_by or self._active_player_id
        if player_id and self._source_details.metadata:
            player = self.mass.players.get_player(player_id)
            if player and self._source_details.metadata.elapsed_time is not None:
                elapsed = self._source_details.metadata.elapsed_time
                updated = self._source_details.metadata.elapsed_time_last_updated
                player._attr_elapsed_time = elapsed
                player._attr_elapsed_time_last_updated = updated
            self.mass.players.trigger_player_update(player_id)

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        if not os.path.exists(self._go_librespot_bin):
            raise FileNotFoundError(
                f"go-librespot binary not found at {self._go_librespot_bin}"
            )
        os.makedirs(self.config_dir, exist_ok=True)
        self.player = self.mass.players.get_player(self.mass_player_id)
        if self.player:
            self._add_seek_to_player(self.mass_player_id)
            if group_id := getattr(self.player, "active_group", None):
                self._add_seek_to_player(group_id)
            self._setup_player_daemon()

    async def unload(self, is_removed: bool = False) -> None:
        """Handle close/cleanup of the provider."""
        self._stop_called = True
        if self._position_poll_task and not self._position_poll_task.done():
            self._position_poll_task.cancel()
            self._position_poll_task = None
        if self._ws_connection:
            await self._ws_connection.close()
        if self._ws_session:
            await self._ws_session.close()
        if self._go_librespot_proc:
            await self._go_librespot_proc.close()
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner_task
        if self._websocket_task and not self._websocket_task.done():
            self._websocket_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._websocket_task
        for callback in self._on_unload_callbacks:
            callback()

    def get_source(self) -> PluginSource:
        """Get (audio)source details for this plugin."""
        return self._source_details

    # ---------------------------------------------------------------------------
    # PluginSource callbacks — called by MA when user triggers controls
    # ---------------------------------------------------------------------------

    async def _on_source_selected(self) -> None:
        """Handle callback when this source is selected on a player."""
        new_player_id = self._source_details.in_use_by
        if not new_player_id:
            return
        if self._active_player_id and self._active_player_id != new_player_id:
            self.logger.info(
                "Source selected on player %s, stopping playback on %s",
                new_player_id,
                self._active_player_id,
            )
            try:
                await self.mass.players.cmd_stop(self._active_player_id)
            except Exception as err:
                self.logger.debug(
                    "Failed to stop previous player %s: %s", self._active_player_id, err
                )
        self._active_player_id = new_player_id
        self.logger.info("Active player set to: %s", self._active_player_id)
        self._add_seek_to_player(new_player_id)
        # If player is in an active group, set in_use_by to the group player
        # so __final_active_source finds our plugin source for the displayed player
        player = self.mass.players.get_player(new_player_id)
        if player and player.state.active_group:
            group_id = player.state.active_group
            self._source_details.in_use_by = group_id
            self._add_seek_to_player(group_id)
            self.logger.debug("Updated in_use_by to group player: %s", group_id)
        # Start position polling for accurate progress bar
        if self._position_poll_task and not self._position_poll_task.done():
            self._position_poll_task.cancel()
        self._position_poll_task = self.mass.create_task(self._position_poll_loop())
        player = self.mass.players.get_player(new_player_id)
        self.logger.info(
            "SOURCE SELECTED DEBUG: player=%s, state.active_group=%s",
            new_player_id,
            player.state.active_group if player else "NO PLAYER",
        )
    def _clear_active_player(self) -> None:
        """Clear the active player when playback ends."""
        prev_player_id = self._active_player_id
        self._active_player_id = None
        self._source_details.in_use_by = None
        self._source_details.metadata = None
        self._current_track_uri = None
        self._last_seek_time = 0.0
        if self._position_poll_task and not self._position_poll_task.done():
            self._position_poll_task.cancel()
            self._position_poll_task = None
        if prev_player_id:
            self.logger.debug(
                "Playback ended on player %s, clearing active player", prev_player_id
            )
            self.mass.players.trigger_player_update(prev_player_id)

    async def _position_poll_loop(self) -> None:
        """Poll go-librespot /status and correct position if it drifts significantly."""
        while not self._stop_called and self._active_player_id:
            try:
                # Skip polling for 3 seconds after a seek to avoid overwriting seek position
                if time.time() - self._last_seek_time < 3:
                    await asyncio.sleep(1)
                    continue
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self._api_base_url}/status") as response:
                        if response.status == 200:
                            data = await response.json()
                            if (
                                not data.get("stopped")
                                and not data.get("paused")
                                and (track := data.get("track"))
                                and self._source_details.metadata
                            ):
                                actual_position = track.get("position", 0) / 1000
                                meta = self._source_details.metadata
                                # Skip if position equals or exceeds duration (track ending)
                                if not (meta.duration and actual_position >= meta.duration):
                                    # Calculate what MA thinks the position is right now
                                    if meta.elapsed_time_last_updated is not None:
                                        expected_position = (
                                            (meta.elapsed_time or 0)
                                            + (time.time() - meta.elapsed_time_last_updated)
                                        )
                                    else:
                                        expected_position = meta.elapsed_time or 0
                                    # Only correct if drift exceeds 3 seconds
                                    if abs(actual_position - expected_position) > 3:
                                        self.logger.debug(
                                            "Position drift: expected=%.1f actual=%.1f, correcting",
                                            expected_position,
                                            actual_position,
                                        )
                                        meta.elapsed_time = actual_position
                                        meta.elapsed_time_last_updated = time.time()
                                        self._trigger_update()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug("Position poll error: %s", e)
            await asyncio.sleep(1)

    async def _on_play_callback(self) -> None:
        """Called by MA when play is requested."""
        await self._send_api_command("player/resume", method="POST")

    async def _on_pause_callback(self) -> None:
        """Called by MA when pause is requested."""
        await self._send_api_command("player/pause", method="POST")

    async def _on_next_callback(self) -> None:
        """Called by MA when next track is requested."""
        await self._send_api_command("player/next", method="POST")

    async def _on_previous_callback(self) -> None:
        """Called by MA when previous track is requested."""
        await self._send_api_command("player/prev", method="POST")

    async def _on_seek_callback(self, position: int) -> None:
        """Called by MA when seek is requested (position in seconds)."""
        self.logger.debug("Seek requested to position: %s seconds", position)
        position_ms = int(position * 1000)
        self._last_seek_time = time.time()
        await self._send_api_json("player/seek", {"position": position_ms})
        if self._source_details.metadata:
            self._source_details.metadata.elapsed_time = position
            self._source_details.metadata.elapsed_time_last_updated = time.time()
        self._force_update()

    async def _on_volume_callback(self, volume: int) -> None:
        """Called by MA when volume change is requested."""
        if not self.external_volume:
            await self._send_api_json("player/volume", {"volume": volume})

    # ---------------------------------------------------------------------------
    # go-librespot API
    # ---------------------------------------------------------------------------

    async def _send_api_command(self, endpoint: str, method: str = "POST") -> None:
        """Send a command to the go-librespot API."""
        url = f"{self._api_base_url}/{endpoint}"
        self.logger.debug("Sending %s request to %s", method, url)
        try:
            async with aiohttp.ClientSession() as session:
                if method == "POST":
                    async with session.post(url) as response:
                        response_text = await response.text()
                        self.logger.debug(
                            "API response (%s): %s - %s",
                            response.status,
                            endpoint,
                            response_text,
                        )
                        if response.status != 200:
                            self.logger.error(
                                "API command failed: %s - Status: %s - Response: %s",
                                endpoint,
                                response.status,
                                response_text,
                            )
                elif method == "PUT":
                    async with session.put(url) as response:
                        response_text = await response.text()
                        self.logger.debug(
                            "API response (%s): %s - %s",
                            response.status,
                            endpoint,
                            response_text,
                        )
                        if response.status != 200:
                            self.logger.error(
                                "API command failed: %s - Status: %s - Response: %s",
                                endpoint,
                                response.status,
                                response_text,
                            )
        except Exception as e:
            self.logger.error("Failed to send API command %s: %s", endpoint, e)

    async def _send_api_json(self, endpoint: str, payload: dict) -> None:
        """Send a JSON POST request to the go-librespot API."""
        url = f"{self._api_base_url}/{endpoint}"
        self.logger.debug("Sending JSON POST to %s: %s", url, payload)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response_text = await response.text()
                    self.logger.debug(
                        "API response (%s): %s - %s",
                        response.status,
                        endpoint,
                        response_text,
                    )
                    if response.status != 200:
                        self.logger.error(
                            "API command failed: %s - Status: %s - Response: %s",
                            endpoint,
                            response.status,
                            response_text,
                        )
        except Exception as e:
            self.logger.error("Failed to send API JSON command %s: %s", endpoint, e)

    # ---------------------------------------------------------------------------
    # go-librespot process management
    # ---------------------------------------------------------------------------

    def _create_config_file(self) -> str:
        """Create go-librespot config file and return its path."""
        config_path = os.path.join(self.config_dir, "config.yml")
        config = {
            "zeroconf_enabled": True,
            "zeroconf_port": 0,
            "credentials": {
                "type": "zeroconf",
                "zeroconf": {"persist_credentials": True},
            },
            "server": {
                "enabled": True,
                "address": "0.0.0.0",
                "port": self.server_port,
                "allow_origin": "*",
                "cert_file": "",
                "key_file": "",
            },
            "log_level": "info",
            "device_id": "",
            "device_name": self.name,
            "device_type": "computer",
            "audio_backend": "pipe",
            "audio_device": "",
            "audio_output_pipe": self.named_pipe,
            "audio_output_pipe_format": "s16le",
            "audio_buffer_time": 50000,
            "audio_period_count": 4,
            "bitrate": 320,
            "volume_steps": 100,
            "initial_volume": 100,
            "external_volume": self.external_volume,
            "disable_autoplay": False,
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return config_path

    async def _go_librespot_runner(self) -> None:
        """Run the spotify connect daemon in a background task."""
        self.logger.info("Starting Spotify Connect Go background daemon")

        # Create named pipe for audio
        await check_output("rm", "-f", self.named_pipe)
        await asyncio.sleep(0.1)
        await check_output("mkfifo", self.named_pipe)
        await check_output("chmod", "666", self.named_pipe)
        await asyncio.sleep(0.1)

        if os.path.exists(self.named_pipe):
            self.logger.info("Named pipe created successfully at %s", self.named_pipe)
        else:
            self.logger.error("Failed to create named pipe at %s", self.named_pipe)

        config_file = self._create_config_file()
        self.logger.debug("Created config file at: %s", config_file)

        # Open pipe for reading permanently so go-librespot can always open
        # its write end. MA will take over reading when select_source is called.
        try:
            self._pipe_fd = os.open(self.named_pipe, os.O_RDONLY | os.O_NONBLOCK)
            self.logger.debug("Pipe held open for reading at fd %s", self._pipe_fd)
        except OSError as e:
            self.logger.error("Failed to open pipe for reading: %s", e)

        try:
            args: list[str] = [
                self._go_librespot_bin,
                "--config_dir",
                self.config_dir,
            ]
            self.logger.debug("Starting go-librespot with args: %s", " ".join(args))
            self._go_librespot_proc = go_librespot = AsyncProcess(
                args, stdout=False, stderr=True, name=f"go-librespot[{self.name}]"
            )
            await go_librespot.start()

            # Give the server time to start
            await asyncio.sleep(3)

            # Check if server is responding
            max_retries = 5
            for i in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self._api_base_url}/status") as response:
                            if response.status == 200:
                                self._go_librespot_started.set()
                                self.logger.info("go-librespot web interface is ready")
                                break
                except Exception as e:
                    if i < max_retries - 1:
                        self.logger.debug(
                            "Waiting for go-librespot to start (attempt %d/%d)",
                            i + 1,
                            max_retries,
                        )
                        await asyncio.sleep(1)
                    else:
                        self.logger.error(
                            "Failed to connect to go-librespot web interface: %s", e
                        )

            if self._go_librespot_started.is_set():
                self._websocket_task = self.mass.create_task(self._websocket_listener())

            stderr_task = self.mass.create_task(self._read_stderr_output(go_librespot))
            return_code = await go_librespot.wait()
            self.logger.info(
                "go-librespot process exited with return code: %s", return_code
            )
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task

        except asyncio.CancelledError:
            self.logger.info("go-librespot runner cancelled")
        except Exception as e:
            self.logger.error("Error running go-librespot: %s", e)
        finally:
            if self._pipe_fd is not None:
                with suppress(OSError):
                    os.close(self._pipe_fd)
                self._pipe_fd = None
            if self._go_librespot_proc:
                await self._go_librespot_proc.close()
            # Cancel websocket task on crash/restart to prevent accumulation
            if self._websocket_task and not self._websocket_task.done():
                self._websocket_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._websocket_task
                self._websocket_task = None
            self._go_librespot_started.clear()
            self.logger.info(
                "Spotify Connect Go background daemon stopped for %s", self.name
            )
            await check_output("rm", "-f", self.named_pipe)

        if not self._go_librespot_started.is_set():
            self.unload_with_error("Unable to initialize go-librespot daemon.")
            return

        if not self._stop_called:
            self.logger.warning(
                "go-librespot exited unexpectedly, restarting in 5 seconds..."
            )
            await asyncio.sleep(5)
            self._setup_player_daemon()

    async def _read_stderr_output(self, process: AsyncProcess) -> None:
        """Read stderr output from go-librespot process."""
        try:
            async for line in process.iter_stderr():
                if "error" in line.lower():
                    self.logger.error("[go-librespot] %s", line)
                else:
                    self.logger.debug("[go-librespot] %s", line)
        except asyncio.CancelledError:
            pass

    # ---------------------------------------------------------------------------
    # WebSocket listener and event handling
    # ---------------------------------------------------------------------------

    async def _websocket_listener(self) -> None:
        """Listen to WebSocket events from go-librespot."""
        retry_count = 0
        max_retries = 10

        while not self._stop_called and retry_count < max_retries:
            try:
                self._ws_session = aiohttp.ClientSession()
                async with self._ws_session.ws_connect(self._ws_url) as ws:
                    self._ws_connection = ws
                    self.logger.info("Connected to go-librespot WebSocket")
                    retry_count = 0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_websocket_event(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.logger.error("WebSocket error: %s", ws.exception())
                            break
            except aiohttp.ClientError as e:
                retry_count += 1
                self.logger.warning(
                    "WebSocket connection failed (attempt %d/%d): %s",
                    retry_count,
                    max_retries,
                    e,
                )
                if retry_count < max_retries:
                    await asyncio.sleep(2)
            except Exception as e:
                self.logger.error("Unexpected WebSocket error: %s", e)
                break
            finally:
                if self._ws_session and not self._ws_session.closed:
                    await self._ws_session.close()
                self._ws_session = None

    async def _handle_websocket_event(self, event_data: dict) -> None:
        """Handle WebSocket event from go-librespot."""
        event_type = event_data.get("type")

        self.logger.debug(
            "WebSocket event: %s - Data keys: %s",
            event_type,
            list(event_data.get("data", {}).keys()) if event_data.get("data") else "None",
        )

        if event_type in ("metadata", "track_changed", "new_track"):
            await self._update_metadata(event_data.get("data", {}))

        elif event_type in ("will_play", "playback_started", "playing", "active"):
            self.logger.info("Playback event: %s", event_type)
            is_resume = False
            if data := event_data.get("data", {}):
                is_resume = data.get("resume", False)
                if not is_resume and (
                    "track" in data or "metadata" in data or "name" in data or "title" in data
                ):
                    await self._update_metadata(data)
                elif not is_resume and "uri" in data:
                    track_uri = data.get("uri", "")
                    if track_uri == self._current_track_uri:
                        self.logger.debug("Track restart detected - resetting position to 0")
                        if self._source_details.metadata:
                            self._source_details.metadata.elapsed_time = 0
                            self._source_details.metadata.elapsed_time_last_updated = time.time()

            if not self._source_details.in_use_by:
                self.logger.info("Selecting source on player %s", self.mass_player_id)
                await self.mass.players.select_source(self.mass_player_id, self.instance_id)
            elif self._active_player_id:
                # Check if player is now in an active group and update in_use_by
                # active_group is None at _on_source_selected time but populated once playing
                player = self.mass.players.get_player(self._active_player_id)
                if player and player.state.active_group:
                    group_id = player.state.active_group
                    if self._source_details.in_use_by != group_id:
                        self.logger.debug(
                            "Updating in_use_by from %s to group %s",
                            self._source_details.in_use_by,
                            group_id,
                        )
                        self._source_details.in_use_by = group_id
                        self._add_seek_to_player(group_id)
                        self._force_update()

        elif event_type in ("playback_paused", "paused", "inactive"):
            self.logger.debug("Playback paused")
            if self._source_details.metadata:
                if data := event_data.get("data", {}):
                    if "position" in data:
                        position_sec = data.get("position") / 1000
                        self._source_details.metadata.elapsed_time = position_sec
                # Freeze progress by clearing elapsed_time_last_updated
                self._source_details.metadata.elapsed_time_last_updated = None
                self._trigger_update()

        elif event_type in ("stopped", "session_disconnected"):
            self.logger.info("Playback stopped/disconnected event: %s", event_type)
            if event_type == "session_disconnected":
                self._clear_active_player()
            else:
                # Delay clearing on 'stopped' to handle Spotify Connect transfer sequences
                stopped_player_id = self._active_player_id

                async def _delayed_clear() -> None:
                    await asyncio.sleep(3)
                    if self._active_player_id != stopped_player_id:
                        self.logger.debug("New playback started during stop delay - not clearing")
                        return
                    if self._current_track_uri:
                        self.logger.debug("Track still active during stop delay - not clearing")
                        return
                    self.logger.debug("Clearing active player after stop delay")
                    self._clear_active_player()

                self.mass.create_task(_delayed_clear())

        elif event_type == "not_playing":
            self.logger.debug("Playback ended (not_playing)")
            self._clear_active_player()

        elif event_type == "volume":
            volume = event_data.get("data", {}).get("value", 0)
            self.logger.debug(
                "go-librespot volume event ignored (MA handles volume): %d", volume
            )

        elif event_type in ("seek", "seeked", "position_correction"):
            if data := event_data.get("data", {}):
                if "position" in data:
                    position_sec = data.get("position") / 1000
                    if self._source_details.metadata:
                        if (
                            self._source_details.metadata.duration
                            and position_sec >= self._source_details.metadata.duration
                        ):
                            return
                        self._source_details.metadata.elapsed_time = position_sec
                        self._source_details.metadata.elapsed_time_last_updated = time.time()
                    self._force_update()
                    self.logger.debug("Seek confirmed at position: %.1f seconds", position_sec)

        elif event_type == "end_of_track":
            self.logger.debug("Track ended")

        elif event_type in ("session_connected", "device_became_active"):
            self.logger.info("Device became active")

        elif event_type == "session_client_changed":
            if data := event_data.get("data", {}):
                self.logger.info(
                    "Control client changed to: %s", data.get("client_name", "Unknown")
                )

        elif event_type == "loading":
            self.logger.debug("Loading track...")

        else:
            self.logger.debug(
                "Unhandled WebSocket event type: %s with data: %s",
                event_type,
                event_data.get("data"),
            )

    async def _update_metadata(self, metadata: dict) -> None:
        """Update metadata from go-librespot events."""
        if not metadata:
            return

        track_info = metadata.get("track", metadata)
        track_uri = track_info.get("uri", "")
        is_new_track = track_uri != self._current_track_uri

        if is_new_track:
            self.logger.info("New track detected: %s", track_uri)
            self._current_track_uri = track_uri
            self._last_seek_time = 0.0

        title = track_info.get("name", "Unknown")
        artist = "Unknown"
        if artist_names := track_info.get("artist_names"):
            if isinstance(artist_names, list) and artist_names:
                artist = (
                    artist_names[0]
                    if isinstance(artist_names[0], str)
                    else str(artist_names[0])
                )
            elif isinstance(artist_names, str):
                artist = artist_names

        album_name = track_info.get("album_name", "Unknown")
        image_url = track_info.get("album_cover_url")

        self.logger.info(
            "Creating PlayerMedia: title=%s, artist=%s, album=%s", title, artist, album_name
        )

        media = PlayerMedia(
            uri=track_uri.replace("spotify:", "spotifyconnect:"),
            title=title,
            artist=artist,
            album=album_name,
            media_type=MediaType.TRACK,
        )

        if image_url:
            media.image_url = image_url

        # Duration comes in milliseconds from go-librespot
        if raw_duration := track_info.get("duration"):
            media.duration = raw_duration / 1000
            self.logger.debug("Track duration: %s seconds", media.duration)

        # Set elapsed time and start the progress clock
        if is_new_track:
            reported_position = (
                track_info.get("position", 0) / 1000 if "position" in track_info else 0
            )
            if reported_position > 5:
                media.elapsed_time = reported_position
                self.logger.info(
                    "Reconnecting to track at position: %s seconds", reported_position
                )
            else:
                media.elapsed_time = 0
                self.logger.info("New track - starting from 0")
        elif "position" in track_info:
            media.elapsed_time = track_info.get("position") / 1000
        else:
            media.elapsed_time = 0

        # Always set the timestamp so MA's progress bar starts advancing
        media.elapsed_time_last_updated = time.time()

        if track_number := track_info.get("track_number"):
            media.track_number = track_number
        if disc_number := track_info.get("disc_number"):
            media.disc_number = disc_number

        self._source_details.metadata = media
        self.logger.info(
            "Updated source metadata: %s - %s (uri: %s)", media.title, media.artist, media.uri
        )

        self._trigger_update()

    # ---------------------------------------------------------------------------
    # Player daemon management
    # ---------------------------------------------------------------------------

    def _setup_player_daemon(self) -> None:
        """Handle setup of the spotify connect daemon for a player."""
        self._go_librespot_started.clear()
        self._runner_task = self.mass.create_task(self._go_librespot_runner())

    def _on_mass_player_event(self, event: MassEvent) -> None:
        """Handle incoming event from linked player."""
        if event.object_id != self.mass_player_id:
            return
        if event.event == EventType.PLAYER_REMOVED:
            self._stop_called = True
            self.mass.create_task(self.unload())
            return
        if event.event == EventType.PLAYER_ADDED:
            self._setup_player_daemon()
            return
