"""AirPlay Multiroom Audio -- Music Assistant player provider.

Spawns one shairport-sync-pa (RAOP / AirPlay 1) subprocess per PulseAudio
remap-sink zone created by the separate Multiroom Audio addon (Chrisuthe),
and registers each directly as an MA player at spawn time -- skipping mDNS
discovery entirely, since MA already knows everything about the player it
just spawned. See the long debugging conversation this was built from for
why: MA's built-in AirPlayProvider discovers shairport-sync instances via
mDNS, which is slow (RFC 6762 conservative re-query backoff) and was the
root cause of MA taking a long time to pick up local AirPlay/SPS instances.

Design choices carried over deliberately from the standalone addon this
replaces (all confirmed working over a long debugging session):
  - output_backend = "pa" only. No ALSA support anywhere in this pipeline --
    not just unused, structurally absent (the shairport-sync-pa binary this
    spawns has no --with-alsa compiled in at all -- see binary_fetch.py's
    header and the Dockerfile stage it's built from). This means the class
    of ALSA-driver kernel hangs that can put a calling thread into an
    uninterruptible sleep cannot be reached through this provider, full
    stop -- not "we don't use it," structurally impossible.
  - audio_backend_buffer_desired_length_in_seconds = 0.5, confirmed via
    real-device testing to absorb WiFi jitter spikes up to ~47ms without
    audible dropouts; the shairport-sync default (~0.35s) was not enough.
  - output_format left unset (auto). A pinned format was tried once, had a
    casing typo, was never validated, and was reverted -- every session
    that was actually confirmed working ran with automatic format
    selection. Don't pin this without deliberately testing it first.
  - Per-instance volume control rides the RTSP layer shairport-sync already
    implements (dmcp-style AirPlay volume, same as talking to a real remote
    AirPlay receiver) -- NOT PA sink volume. This sidesteps a whole class of
    module-stream-restore-reasserts-stale-volume bugs hit earlier with a
    different provider's PA-volume approach, by design: normal playback
    volume changes never touch PA sink state under this design at all. See
    cmd_set_volume() below for what's actually confirmed vs. what needs
    verifying against current MA AirPlay-player volume-command code.

TODO before this file will actually run, none of which are guesses I'm
confident enough to have filled in blind -- confirm each against current
music-assistant/server source before relying on it (the whole point of
building it this way rather than fabricating a complete-looking file is
that guessing at MA-internal API surface wasted real hours tonight; better
to hand you an honestly partial file than a confidently wrong complete one):
  1. Import path / exact base class for PlayerProvider -- confirm against
     music_assistant/models/player_provider.py (or wherever it currently
     lives; this moved at least once already in the MA codebase history
     visible in this conversation).
  2. Player object construction -- exact required/optional fields, enums
     for PlayerType/DeviceInfo etc. Confirm against
     music_assistant/models/player.py or equivalent.
  3. self.mass.players.register(player) -- this ONE call is confirmed real,
     pulled directly from the current airplay/provider.py source during
     this conversation. Everything around how `player` gets constructed
     before that call is what needs verifying.
  4. get_config_entries() signature and ConfigEntry class -- confirmed to
     exist as a PlayerProvider instance method (not a module-level
     __init__.py function) as of the local_audio work referenced in this
     session's memory, but the exact current signature isn't verified here.
  5. Volume-set command routing -- whether MA's current AirPlay player
     volume commands go out over raw RTSP (as described earlier in this
     conversation) or via the newer cliairplay CLI binary discovered during
     this same session (MA's AirPlay provider now shells out to a unified
     `cliairplay` binary rather than doing raw RTSP in Python, per the
     airplay-cli repo found searching for this). If it's the latter, the
     cmd_set_volume() sketch below is very likely wrong for THIS provider's
     purposes anyway -- this provider spawns its own shairport-sync-pa
     process directly and can send it an RTSP SET_PARAMETER volume command
     itself, without needing MA's sender-side machinery at all. Confirm
     which one is actually simplest before implementing either.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# TODO: same issue as CACHE_DIR below -- /config is a HAOS/container
# convention, not guaranteed to exist or be writable when running MA
# directly as a regular user (confirmed: this failed with PermissionError
# the same way CACHE_DIR did). Should eventually come from MA's own
# provider-storage API if one exists; for now, same env-var-with-writable-
# default pattern as CACHE_DIR.
CONFIG_DIR = Path(
    os.environ.get(
        "AIRPLAY_MULTIROOM_CONFIG_DIR",
        Path.home() / ".cache" / "airplay-multiroom-audio" / "config",
    )
)
# TODO: /data is not guaranteed writable -- confirmed the hard way (this
# hardcoded path failed with PermissionError running MA directly, outside
# any container, as a regular user). This should come from whatever MA's
# own API for provider persistent-storage location is (something like
# mass.storage_path, if that exists -- unverified, same category as the
# other MA-internal TODOs in provider.py) rather than being hardcoded here.
# For now, points at a location under the current user's home so local
# testing can proceed; do not ship this default as-is.
CACHE_DIR = Path(os.environ.get("AIRPLAY_MULTIROOM_CACHE_DIR", Path.home() / ".cache" / "airplay-multiroom-audio-bin"))

PORT_BASE_DEFAULT = 5020
UDP_PORT_BASE_DEFAULT = 6001

# Same interface-detection problem the standalone addon hit: under
# host_network, "select interface(s) automatically" can pick the internal
# Docker/hassio bridge over the real LAN interface, producing RTSP-connects-
# no-audio. Override via env var per-deployment rather than hardcoding a
# single interface name here -- unlike the addon (single host, confirmed
# interface name), a provider install could run on arbitrary hardware.
AIRPLAY_INTERFACE = os.environ.get("AIRPLAY_INTERFACE")  # None = let shairport-sync guess


@dataclass
class SinkZone:
    """One PulseAudio remap-sink zone discovered from the Multiroom Audio addon."""

    sink_name: str
    port: int
    udp_port_base: int


async def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a subprocess with a hard timeout. Never block the event loop.

    Every external process call in this module goes through this helper --
    no bare subprocess.run/Popen anywhere. This is not a style preference:
    a naive blocking subprocess call on the event loop thread is exactly
    the class of bug this session's own local_audio work hit and fixed
    once already (the PAVolumeController._lock executor-stall bug,
    resolved by adding a bounded timeout). Don't reintroduce it here.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def discover_remap_sinks(retries: int = 10, delay: float = 2.0) -> list[str]:
    """List PulseAudio sink names, retrying if none are found yet.

    By default, only lists module-remap-sink sinks -- matching the
    original standalone addon's deliberate scoping (its own header
    comments: masters that don't get a remap sink from the Multiroom
    Audio addon get no AirPlay instance either, "by design"). This
    directly encodes a lesson from tonight's live debugging too: the
    standalone addon's generator script had a known, and eventually
    actually-hit, startup-order race against that addon -- its own header
    comments flagged the risk months ago ("if that ever proves to be a
    real problem... the fix is a short retry/wait loop"), and it did
    prove to be a real problem, repeatedly, during testing. Build the
    retry in from day one here rather than waiting to hit it.

    Set AIRPLAY_MULTIROOM_ALL_SINKS=1 to include every PulseAudio sink,
    not just remap-sink ones -- for dev/POC testing on a box that doesn't
    have the Multiroom Audio addon running at all (module-remap-sink sinks
    can never exist there, no matter how long the retry loop waits). Not
    the intended default for a real deployment: it changes which physical
    outputs get an AirPlay instance, silently, versus what the original
    addon's scoping decision intended.
    """
    include_all = os.environ.get("AIRPLAY_MULTIROOM_ALL_SINKS", "").lower() in ("1", "true", "yes")
    LOGGER.info(
        "AIRPLAY_MULTIROOM_ALL_SINKS raw value: %r -- include_all resolved to %s",
        os.environ.get("AIRPLAY_MULTIROOM_ALL_SINKS"),
        include_all,
    )
    for attempt in range(1, retries + 1):
        returncode, stdout, stderr = await _run(["pactl", "list", "sinks", "short"])
        if returncode != 0:
            LOGGER.warning(
                "pactl list sinks failed (attempt %d/%d): %s", attempt, retries, stderr.strip()
            )
        else:
            sinks = [
                line.split()[1]
                for line in stdout.splitlines()
                if len(line.split()) > 1 and (include_all or "module-remap-sink" in line)
            ]
            if sinks:
                return sinks
            LOGGER.info(
                "No %ssinks found yet (attempt %d/%d)%s",
                "" if include_all else "module-remap-sink ",
                attempt,
                retries,
                "" if include_all else " -- Multiroom Audio addon topology may not exist yet",
            )
        await asyncio.sleep(delay)
    raise RuntimeError(
        f"No PulseAudio remap-sink zones found after {retries} attempts "
        f"({retries * delay:.0f}s). Confirm the Multiroom Audio addon is "
        "running and has created its sink topology."
    )


def build_shairport_config(zone: SinkZone, config_path: Path) -> None:
    """Write a shairport-sync .conf for one zone.

    Content is carried over directly from generate_airplay_services.sh,
    already debugged and confirmed working over this session: the `pa`
    section name (not `pulseaudio` -- that was a real, previously-shipped
    bug that silently sent audio to PulseAudio's default sink instead of
    the intended zone), the 0.5s buffer, output_format left unset.
    """
    interface_line = (
        f'  interface = "{AIRPLAY_INTERFACE}";\n' if AIRPLAY_INTERFACE else ""
    )
    config_path.write_text(
        f"""general :
{{
  name = "{zone.sink_name}";
  port = {zone.port};
{interface_line}  output_backend = "pa";
  udp_port_base = {zone.udp_port_base};
  audio_backend_buffer_desired_length_in_seconds = 0.5;
}};
sessioncontrol :
{{
  allow_session_interruption = "yes";
}};
pa :
{{
  sink = "{zone.sink_name}";
  application_name = "Shairport Sync";
}};
"""
    )


class AirplayMultiroomProcess:
    """One running shairport-sync-pa subprocess for one sink zone."""

    def __init__(self, zone: SinkZone, binary_path: Path, config_path: Path) -> None:
        self.zone = zone
        self.binary_path = binary_path
        self.config_path = config_path
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "-a",
            self.zone.sink_name,
            "-p",
            str(self.zone.port),
            "-c",
            str(self.config_path),
            "-vv",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # TODO: confirm readiness before registering with MA (see module
        # docstring / earlier conversation) -- a bounded poll of the
        # assigned port being open is the cheap, already-discussed fix if
        # MA's player-registration path turns out to assume the player is
        # immediately ready. Not implemented here since it depends on #1.
        await asyncio.sleep(0.2)
        if self._proc.returncode is not None:
            # Process already exited -- read whatever it printed before
            # dying instead of discarding it. This is exactly the output
            # that would have been dumped straight to DEVNULL before;
            # don't repeat the "guess at the failure instead of reading
            # the actual error text" mistake from earlier tonight.
            output = b""
            if self._proc.stdout is not None:
                try:
                    output = await asyncio.wait_for(self._proc.stdout.read(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            raise RuntimeError(
                f"shairport-sync-pa for {self.zone.sink_name} exited immediately "
                f"with code {self._proc.returncode}:\n"
                f"{output.decode(errors='replace')}"
            )

    async def stop(self, timeout: float = 5.0) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            LOGGER.warning(
                "shairport-sync-pa for %s did not exit within %.1fs, killing",
                self.zone.sink_name,
                timeout,
            )
            self._proc.kill()
            await self._proc.wait()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None


# ---------------------------------------------------------------------------
# TODO: everything below this point is the part that needs grounding against
# current MA source (see module docstring items 1-5). What's sketched here
# is structurally what needs to happen -- discover zones, spawn processes,
# register players -- but the exact MA API calls are placeholders except
# where noted.
# ---------------------------------------------------------------------------


async def resolve_binary() -> Path:
    """Find a usable shairport-sync-pa binary, preferring a system install.

    Resolution order:
      1. AIRPLAY_MULTIROOM_SPS_BINARY env var, if set -- explicit override,
         highest priority, no further checks beyond existence+executable.
      2. `shairport-sync` on $PATH, if present -- covers exactly the
         Linux Mint POC case: you already compiled and installed one
         yourself with the right flags, no reason to also download a
         second copy just to satisfy this provider.
      3. Fall back to the download/verify/cache path (ensure_binary()),
         for environments (the actual MA dev-server / HAOS deployment)
         where there's no pre-installed binary to find.

    For paths 1 and 2, this does NOT re-verify against the pinned SHA256
    in binary_fetch.py -- that check only applies to what this module
    downloads itself. A system-installed binary is trusted as-is, same as
    trusting whatever you already built and ran `-V` on by hand. Worth
    remembering if this ever stops being "just me testing on my own Mint
    box" and becomes something other people run.
    """
    override = os.environ.get("AIRPLAY_MULTIROOM_SPS_BINARY")
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise RuntimeError(
                f"AIRPLAY_MULTIROOM_SPS_BINARY is set to {override_path}, "
                "but that path does not exist."
            )
        if not os.access(override_path, os.X_OK):
            raise RuntimeError(f"{override_path} exists but is not executable.")
        LOGGER.info("Using shairport-sync binary from AIRPLAY_MULTIROOM_SPS_BINARY: %s", override_path)
        return override_path

    system_binary = shutil.which("shairport-sync")
    if system_binary:
        binary_path = Path(system_binary)
        LOGGER.info(
            "Using system shairport-sync from PATH: %s -- assuming it was built "
            "with --with-pa (this provider does not verify that; confirm with "
            "'shairport-sync -V' yourself if unsure)",
            binary_path,
        )
        return binary_path

    LOGGER.info(
        "No system shairport-sync found (no AIRPLAY_MULTIROOM_SPS_BINARY set, "
        "none on PATH) -- falling back to downloading a verified binary"
    )
    try:
        from .binary_fetch import ensure_binary
    except ImportError as exc:
        raise RuntimeError(
            "No system shairport-sync found, and binary_fetch.py is not "
            "present to fall back to downloading one. Either install "
            "shairport-sync (with --with-pa) and ensure it's on PATH or set "
            "AIRPLAY_MULTIROOM_SPS_BINARY, or restore binary_fetch.py."
        ) from exc
    return await ensure_binary(CACHE_DIR)


class AirplayMultiroomProvider:  # TODO: subclass the real PlayerProvider base
    """Sketch of the provider's setup flow. Not a complete PlayerProvider."""

    def __init__(self, mass) -> None:  # TODO: real __init__ signature
        self.mass = mass
        self._processes: dict[str, AirplayMultiroomProcess] = {}

    async def setup(self) -> None:
        binary_path = await resolve_binary()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        sinks = await discover_remap_sinks()
        port = PORT_BASE_DEFAULT
        udp_base = UDP_PORT_BASE_DEFAULT
        for sink_name in sinks:
            zone = SinkZone(sink_name=sink_name, port=port, udp_port_base=udp_base)
            config_path = CONFIG_DIR / f"{sink_name}.conf"
            build_shairport_config(zone, config_path)

            process = AirplayMultiroomProcess(zone, binary_path, config_path)
            await process.start()
            self._processes[sink_name] = process

            # TODO (confirmed-real call, unconfirmed construction of `player`):
            # player = Player(...)  # needs real Player() fields
            # await self.mass.players.register(player)

            port += 1
            udp_base += 10

    async def unload(self) -> None:
        await asyncio.gather(*(p.stop() for p in self._processes.values()))