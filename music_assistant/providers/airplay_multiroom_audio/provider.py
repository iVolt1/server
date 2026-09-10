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

RESOLVED this session, against real source (not guessed):
  - PlayerProvider base class found and subclassed properly (see the
    import comment above class AirplayMultiroomProvider below). Its own
    base, Provider, was also found -- __init__ signature, the instance_id
    property, get_config_entries()'s real default implementation,
    unload()'s real signature (is_removed: bool = False), and the
    self.available flag (starts False, must be set True once actually
    ready) are all confirmed directly from that source, not inferred.
  - discover_players() confirmed as the correct override point for a
    provider that registers players directly instead of via mDNS.
  - self.mass.players.register(player) confirmed real (pulled directly
    from the built-in airplay provider's own source).

STILL OPEN, genuinely unverified -- don't assume these are right just
because the rest of the file now runs further than it used to:
  1. Player object construction -- exact required/optional fields, enums
     for PlayerType/DeviceInfo etc. Confirm against
     music_assistant/models/player.py or equivalent. This is the next
     concrete blocker: discover_players() spawns processes successfully
     but never actually registers a Player with MA yet.
  2. Volume-set command routing -- whether MA's current AirPlay player
     volume commands go out over raw RTSP (as described earlier in this
     conversation) or via the newer cliairplay CLI binary discovered during
     this same session (MA's AirPlay provider now shells out to a unified
     `cliairplay` binary rather than doing raw RTSP in Python, per the
     airplay-cli repo found searching for this). If it's the latter, a
     cmd_set_volume() implementation modeled on that would very likely be
     wrong for THIS provider's purposes anyway -- this provider spawns its
     own shairport-sync-pa process directly and can send it an RTSP
     SET_PARAMETER volume command itself, without needing MA's sender-side
     machinery at all. Confirm which one is actually simplest before
     implementing either.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Module path inferred, not independently confirmed via a file listing:
# PlayerProvider's own source used `from .provider import Provider` as a
# relative import, and the file containing Provider itself is titled
# "Model/base for a Provider implementation within Music Assistant" --
# strongly suggesting music_assistant/models/provider.py, with
# PlayerProvider as the sibling music_assistant/models/player_provider.py.
# If this import fails, that's the thing to double check first.
from music_assistant.models.player import Player
from music_assistant.models.player_provider import PlayerProvider

LOGGER = logging.getLogger(__name__)

# Hardcoded rather than imported from music_assistant.providers.airplay.constants,
# to avoid a hard dependency on another provider's internal module (not a
# guaranteed-stable public API). Confirmed exact values directly from that
# module's real source during this session.
RAOP_DISCOVERY_TYPE = "_raop._tcp.local."
AIRPLAY_DISCOVERY_TYPE = "_airplay._tcp.local."

# Deliberately minimal TXT record. get_model_info() (confirmed from real
# source) falls through to the "am" property when manufacturer/model are
# absent, producing ("AirPlay", "ShairportSync") -- confirmed non-Apple, so
# is_apple_device() returns False and _setup_player() takes the simpler
# GenericAirPlayPlayer path rather than AirPlayControlPlayer's Companion/MRP
# machinery. "features"/"ft" are deliberately omitted too, so
# supports_airplay2() reads False and cliairplay never attempts an AirPlay2
# negotiation these RAOP-only receivers can't do -- the exact failure mode
# hit earlier this session ("session SETUP -> 501") when a build genuinely
# did advertise AirPlay2 capability it couldn't back up.
SYNTHETIC_AIRPLAY_TXT = {"am": "ShairportSync", "txtvers": "1"}


async def announce_as_airplay_device(
    mass, zone: SinkZone, raop_wait_timeout: float = 5.0
):
    """Register a synthetic _airplay._tcp record so the built-in AirPlayProvider
    discovers this zone's shairport-sync-pa instance immediately instead of
    stalling ~10s per device.

    Returns the registered AsyncServiceInfo (so the caller can unregister it
    on unload -- see AirplayMultiroomProvider.unload()), or None if no RAOP
    record was found to announce against.

    Root cause this addresses, confirmed directly from real
    DiscoveryController/AirPlayProvider source pulled during this session:
    every shairport-sync-pa instance advertises RAOP (_raop._tcp) only --
    confirmed via avahi-browse against every instance in this whole project,
    both this provider's and the standalone addon's -- never a companion
    _airplay._tcp record. AirPlayProvider._setup_player() always tries to
    find that companion record for a newly-seen RAOP device
    (async_find_mdns_service(AIRPLAY_DISCOVERY_TYPE, ..., timeout=10.0)) and,
    since it structurally never exists for these receivers, always burns the
    full 10-second timeout. That lookup is also serialized behind a single
    per-provider asyncio.Lock in DiscoveryController, so N devices cost
    N x ~10s in strict sequence -- confirmed by a real-world capture showing
    11 devices registering at almost exactly 10.00s/device apart.

    Registering this record ourselves, right after learning the real mDNS
    name shairport-sync's embedded tinysvcmdns responder announced for this
    zone, gives that lookup something to find from local cache/self-response
    instead of exhausting the timeout -- letting _setup_player()'s existing,
    already-correct logic run fast and register a real, fully-working
    GenericAirPlayPlayer, same as it already does for a real AirPlay device.

    UNVERIFIED, the one real assumption this whole mechanism rests on: that
    a service registered on our own Zeroconf instance
    (mass.discovery.aiozc.async_register_service) shows up in that same
    instance's *inbound* cache (zeroconf.cache.cache) fast enough for
    async_find_mdns_service()'s cache-scan to find it -- python-zeroconf's
    register/cache internals were not part of what was pulled and confirmed
    this session, unlike everything else this function relies on. Verify
    directly: after this returns, watch whether _setup_player() actually
    registers a real player for this zone within a second or two (not ~10s)
    -- if it still takes the full timeout, this assumption was wrong and the
    mechanism needs rethinking, not just retrying.
    """
    # Dot-splitting is no longer a concern here: zone.announce_name has
    # dots stripped at the source (see SinkZone.announce_name), so
    # shairport-sync never has a literal dot to announce in the first
    # place -- fixing the root cause rather than matching around it, as
    # an earlier version of this code did.
    #
    # DNS-label-length truncation is still real and separate, confirmed
    # on real HAOS hardware: labels have a hard 63-byte wire-format limit
    # (RFC 1035). The advertised name is "<12-hex-char pseudo-MAC>@<name>"
    # -- 13 fixed bytes of prefix, leaving 50 for the name itself. A name
    # longer than that gets silently truncated by the mDNS stack before
    # it's ever announced. Confirmed exactly: a real HAOS sink name
    # ("HD_Audio_Generic_Digital_Surround_7_1_HDMI_2_fc_lfe", 51 chars,
    # 64 combined with the MAC prefix -- one byte over) consistently failed
    # this lookup, while announcing/registering fine as its own truncated
    # 50-char form (missing exactly the trailing "e") via the slower
    # built-in mDNS path. Truncating our own search target to the same
    # 50-byte budget is what makes the lookup match what's actually on
    # the wire.
    name_filter = zone.announce_name[:50]
    raop_info = await mass.discovery.async_find_mdns_service(
        RAOP_DISCOVERY_TYPE, name_filter=name_filter, timeout=raop_wait_timeout
    )
    if raop_info is None:
        LOGGER.warning(
            "Could not find RAOP mDNS record for %s within %.1fs -- "
            "skipping synthetic AirPlay announcement for this zone "
            "(built-in AirPlayProvider will still find it eventually, just "
            "slowly, via its own ~10s-per-device path)",
            zone.sink_name,
            raop_wait_timeout,
        )
        return

    # raop_info.name is "<pseudo-MAC>@<sink_name>._raop._tcp.local." -- the
    # pseudo-MAC prefix is generated internally by tinysvcmdns, not something
    # we control or can predict, so it has to be learned via lookup rather
    # than assembled ourselves. Reusing the identical "<MAC>@<sink_name>"
    # portion for the synthetic _airplay record is what makes
    # _setup_player()'s name parsing derive the same raw_id/display_name (and
    # therefore the same player_id) regardless of which record type it sees
    # first -- confirmed from real _setup_player() source: it splits on the
    # first "@" in whichever info.name it's given.
    base_name = raop_info.name.split(".", 1)[0]  # "<MAC>@<sink_name>"

    from zeroconf import NonUniqueNameException  # noqa: PLC0415
    from zeroconf.asyncio import AsyncServiceInfo  # noqa: PLC0415

    airplay_info = AsyncServiceInfo(
        AIRPLAY_DISCOVERY_TYPE,
        name=f"{base_name}.{AIRPLAY_DISCOVERY_TYPE}",
        addresses=raop_info.addresses,
        # Deliberately 0, not a real port: _setup_player() only uses this to
        # fire a background /info probe (probe_audio_formats) for 24-bit
        # capability detection -- an AirPlay2-only HTTP endpoint our RAOP-only
        # receivers don't serve. 0 is falsy, so that probe is skipped
        # entirely rather than left to fail/timeout harmlessly.
        port=0,
        properties=SYNTHETIC_AIRPLAY_TXT,
        server=raop_info.server,
    )
    aiozc = mass.discovery.aiozc
    try:
        await aiozc.async_register_service(airplay_info)
    except NonUniqueNameException:
        # Same reclaim pattern as AirPlayProvider._register_dacp_service()'s
        # real source (confirmed this session): our service name is
        # deterministic per sink, so a prior run's registration that was
        # never cleanly unregistered on unload collides with this one.
        # Flush the stale record and register again, rather than fail.
        LOGGER.debug(
            "Synthetic _airplay._tcp record %s already registered -- reclaiming",
            airplay_info.name,
        )
        await aiozc.async_unregister_service(airplay_info)
        await asyncio.sleep(1.0)  # matches DACP_RECLAIM_DELAY's real value
        try:
            await aiozc.async_register_service(airplay_info)
        except NonUniqueNameException:
            # A second collision, right after unregistering, means this
            # isn't a stale leftover from a prior run -- it's a LIVE,
            # currently-valid registration from a DIFFERENT zone that
            # happens to truncate to the identical name. Confirmed real
            # cause on real hardware: several distinct raw PA sinks whose
            # names all start "alsa_output." collapse to the identical
            # truncated device name everywhere in this whole pipeline
            # (our own lookup's truncation, and MA's own real
            # _setup_player() name parsing, confirmed from source) --
            # genuinely ambiguous, not something either side can
            # disambiguate after the fact. Log and skip this one zone
            # rather than letting the exception propagate and abort every
            # zone still left in discover_players()'s loop.
            LOGGER.warning(
                "Synthetic _airplay._tcp record %s collides with a "
                "currently-live registration from a different zone (not "
                "a stale leftover) -- skipping synthetic announcement for "
                "%s. This usually means multiple distinct sink names "
                "truncate to the same name once DNS label limits/dot "
                "splitting apply (confirmed cause: several raw "
                "'alsa_output.*'-style sink names collapsing to the "
                "identical 'alsa_output'). The built-in AirPlayProvider "
                "will still eventually find this zone via its own slower "
                "mDNS path, just without this speedup, and its identity "
                "may be ambiguous among the colliding sinks either way.",
                airplay_info.name,
                zone.sink_name,
            )
            return None
    LOGGER.debug(
        "Registered synthetic _airplay._tcp record for %s (as %s)",
        zone.sink_name,
        airplay_info.name,
    )
    return airplay_info


class AirplayMultiroomPlayer(Player):
    """Minimal Player for one spawned shairport-sync-pa instance.

    Deliberately declares NO supported_features. This provider's Python
    code never actually handles a play/stop/volume command -- the AirPlay
    protocol itself does, driven by whatever external sender (a phone, or
    MA's own built-in AirPlay provider via cliairplay) connects to the
    spawned shairport-sync-pa process. An empty feature set is a design
    choice reflecting that reality, not an unfinished stub.

    Construction order matters and is easy to get backwards (confirmed
    from the real Player.__init__ source): _attr_name must be set BEFORE
    calling super().__init__(), since it's read during that call for
    create_default_player_config() and is NOT one of the attributes the
    base class resets afterward. Everything else that base __init__ DOES
    unconditionally reset (_attr_supported_features, _attr_device_info,
    etc.) must be set AFTER calling super().__init__(), or the reset
    silently wipes it back to an empty default.

    OPEN RISK, not yet resolved by anything in this codebase -- confirmed
    from tonight's own logs, not a guess: MA's built-in AirPlay provider
    already creates its own separate "protocol" player for a shairport-
    sync instance purely from its own mDNS discovery, completely
    independent of what this provider does (`Player (type protocol)
    registered: ap70cd60aadede/alsa_output` appeared in the logs with
    zero involvement from this provider's code). If this class's players
    get registered AND the built-in provider's mDNS discovery also finds
    the same shairport-sync-pa process, that's very likely two separate
    entries for the same physical device once mDNS catches up -- their
    player_ids are scoped to different provider instances, so nothing
    here would deduplicate them automatically. The airplay_receiver
    plugin's own "skip same-host instances" filter is specific to THAT
    plugin's domain and has no reason to apply to this one. Test this
    directly before assuming either way: register one player, then watch
    whether a second one for the same zone appears once mDNS would have
    had time to catch up.
    """

    def __init__(self, provider: PlayerProvider, player_id: str, display_name: str) -> None:
        # Must happen before super().__init__() -- see class docstring.
        self._attr_name = display_name
        super().__init__(provider, player_id)
        # Must happen after super().__init__() -- the base class resets
        # these unconditionally during its own __init__, so setting them
        # any earlier would just get silently wiped.
        self._attr_supported_features = set()

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

    @property
    def announce_name(self) -> str:
        """The name to actually announce over mDNS -- sink_name with dots
        replaced, NOT the raw PA sink name itself.

        Root-cause fix, not a workaround: DNS labels split on literal dots,
        and PipeWire's own sink-naming convention always puts the generic
        class prefix first and the identifying hardware descriptor after
        the first dot (e.g. "alsa_output.pci-0000_00_1b.0.analog-stereo.2"
        or "alsa_output.usb-Generic_ELEGIANT_SR030..."). Confirmed on real
        hardware: every dotted sink name truncates to the same generic,
        non-identifying "alsa_output" once announced, colliding with every
        other dotted sink on the same host, in both our own mDNS lookup
        and MA's own real _setup_player() name parsing (neither of which
        can be changed from this side). Removing the dot at the source,
        before shairport-sync ever announces anything, means there's
        nothing left to truncate on -- the full identifying name survives
        intact everywhere downstream.

        sink_name itself is deliberately untouched and must stay that way:
        it's the real PulseAudio sink identifier used in the .conf's
        `sink = "..."` line, and has to match exactly for audio routing
        to actually work. Only the announced name changes.
        """
        return self.sink_name.replace(".", "_")


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


def _parse_pactl_sinks_verbose(output: str) -> list[dict[str, str]]:
    """Parse `pactl list sinks` (verbose) output into a list of per-sink dicts.

    Each dict has at least "name" and "driver"; "master_device" is present
    only when the sink's Properties block carries a device.master_device
    key. UNVERIFIED: this property name/format is inferred from this
    project's own local_audio provider history (its remap_topology.py used
    device.master_device for exactly this kind of relationship detection),
    not confirmed against a real `pactl list sinks` dump for this addon's
    actual remap sinks. Verify directly: `pactl list sinks | grep -A 40
    module-remap-sink` on the real system, and check the Properties block
    for whatever key actually links a remap sink back to its master --
    fix the property name below if it's different.
    """
    sinks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_properties = False
    for line in output.splitlines():
        if line.startswith("Sink #"):
            if current.get("name"):
                sinks.append(current)
            current = {}
            in_properties = False
            continue
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current["name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Driver:"):
            current["driver"] = stripped.split(":", 1)[1].strip()
        elif stripped == "Properties:":
            in_properties = True
        elif in_properties and stripped.startswith("device.master_device"):
            # Property lines look like: device.master_device = "some-value"
            value = stripped.split("=", 1)[1].strip().strip('"')
            current["master_device"] = value
        elif in_properties and stripped and not stripped.startswith(('"', "device.")):
            # A non-property-looking line (e.g. the next section header)
            # ends the Properties block for this sink.
            in_properties = False
    if current.get("name"):
        sinks.append(current)
    return sinks


async def discover_remap_sinks(retries: int = 10, delay: float = 2.0) -> list[str]:
    """List PulseAudio sink names to create AirPlay instances for, retrying
    if none are found yet.

    Selection rule: every module-remap-sink sink, PLUS every sink that has
    no remap-sink children of its own -- but NOT a multichannel card's bare
    master sink when it DOES have remap children (that card's real zones
    already cover it; a redundant raw-master AirPlay instance alongside
    them would just be confusing). This matches what the goal actually is:
    AirPlay coverage for every real audio destination, whether or not the
    Multiroom Audio addon happened to create a remap zone for it -- not
    "only what that addon explicitly created," which was the older,
    narrower scoping this replaces.

    Master/child relationship is detected via each remap sink's
    device.master_device property -- see _parse_pactl_sinks_verbose()'s
    docstring for why that's an inference, not a confirmed fact, and how to
    check it directly. If that detection fails to identify any relationship
    at all (property missing/unparseable on every remap sink found), this
    falls back to including everything rather than silently dropping a
    legitimate standalone sink -- the safer failure direction given the
    stated goal is broader coverage, not narrower.

    Set AIRPLAY_MULTIROOM_ALL_SINKS=1 to skip the master-exclusion logic
    entirely and include literally every sink, remap or not, master-with-
    children or not -- useful for debugging this selection logic itself,
    not the intended normal mode now that the rule above is the default.

    This also directly encodes a lesson from tonight's live debugging: the
    standalone addon's generator script had a known, and eventually
    actually-hit, startup-order race against the Multiroom Audio addon --
    build the retry in from day one here rather than waiting to hit it.
    """
    force_all = os.environ.get("AIRPLAY_MULTIROOM_ALL_SINKS", "").lower() in ("1", "true", "yes")
    for attempt in range(1, retries + 1):
        returncode, stdout, stderr = await _run(["pactl", "list", "sinks"])
        if returncode != 0:
            LOGGER.warning(
                "pactl list sinks failed (attempt %d/%d): %s", attempt, retries, stderr.strip()
            )
            await asyncio.sleep(delay)
            continue

        all_sinks = _parse_pactl_sinks_verbose(stdout)
        if not all_sinks:
            LOGGER.info("No PulseAudio sinks found yet (attempt %d/%d)", attempt, retries)
            await asyncio.sleep(delay)
            continue

        LOGGER.debug(
            "pactl list sinks found %d sink(s): %s",
            len(all_sinks),
            [s["name"] for s in all_sinks],
        )

        if force_all:
            return [s["name"] for s in all_sinks]

        remap_sinks = [s for s in all_sinks if "module-remap-sink" in s.get("driver", "")]
        masters_with_children = {
            s["master_device"] for s in remap_sinks if s.get("master_device")
        }
        LOGGER.debug(
            "%d remap sink(s): %s -- master_device values found: %s",
            len(remap_sinks),
            [s["name"] for s in remap_sinks],
            masters_with_children or "(none)",
        )
        if remap_sinks and not masters_with_children:
            LOGGER.warning(
                "Found %d remap sink(s) but could not determine any master/child "
                "relationship (device.master_device property missing or "
                "unparseable) -- falling back to including every sink rather "
                "than risk silently dropping a legitimate standalone one. "
                "Verify the property name in _parse_pactl_sinks_verbose() "
                "against real `pactl list sinks` output.",
                len(remap_sinks),
            )
            return [s["name"] for s in all_sinks]

        selected = [
            s["name"]
            for s in all_sinks
            if "module-remap-sink" in s.get("driver", "")
            or s["name"] not in masters_with_children
        ]
        excluded = [
            s["name"]
            for s in all_sinks
            if "module-remap-sink" not in s.get("driver", "")
            and s["name"] in masters_with_children
        ]
        if excluded:
            LOGGER.debug(
                "Excluded %d sink(s) as masters with remap children: %s -- "
                "if any of these should actually have gotten their own "
                "AirPlay instance, the master_device detection likely "
                "matched incorrectly; check against real `pactl list "
                "sinks` output for that specific sink.",
                len(excluded),
                excluded,
            )
        LOGGER.debug("Final selected sink list (%d): %s", len(selected), selected)
        if selected:
            return selected
        LOGGER.info("No eligible sinks found yet (attempt %d/%d)", attempt, retries)
        await asyncio.sleep(delay)
    raise RuntimeError(
        f"No PulseAudio sinks found after {retries} attempts "
        f"({retries * delay:.0f}s). Confirm PulseAudio/the Multiroom Audio "
        "addon is running."
    )


def build_shairport_config(zone: SinkZone, config_path: Path) -> None:
    """Write a shairport-sync .conf for one zone.

    Content is carried over directly from generate_airplay_services.sh,
    already debugged and confirmed working over this session against a
    4.3.7-class shairport-sync build: the `pa` section name (not
    `pulseaudio` -- that was a real, previously-shipped bug on THAT build
    that silently sent audio to PulseAudio's default sink instead of the
    intended zone), the 0.5s buffer, output_format left unset.

    output_backend / section name are overridable via
    AIRPLAY_MULTIROOM_SPS_BACKEND because they are NOT universal across
    shairport-sync versions -- confirmed directly: a newer 5.6-dev build
    rejected "pa" outright ("the audio backend selected: pa is not
    supported"), and its own -h output lists the real backend names as
    "pipewire" and "pulseaudio" instead. Default stays "pa" here because
    every other pinned build in this whole project (the standalone
    addon's Dockerfile, MA's own Dockerfile.base, the 5.5.1-classic image
    extracted from Docker Hub) is 4.3.7-class, where "pa" is the
    confirmed-correct value.

    IMPORTANT CAVEAT, not yet verified either way: this assumes the
    section header (`pa { ... }` vs `pulseaudio { ... }`) renamed in step
    with the output_backend selector string on the newer build. That's a
    plausible guess, not confirmed -- and if it's wrong, the failure mode
    is silent (libconfig ignores an unrecognized section rather than
    erroring, the exact same failure shape as the pa/pulseaudio
    section-name bug this session already hit once on the 4.3.7 build).
    Before trusting this on the 5.6-dev build: check that version's
    actual bundled sample shairport-sync.conf for the real current
    section name, rather than assuming it tracks the backend selector.
    """
    backend = os.environ.get("AIRPLAY_MULTIROOM_SPS_BACKEND", "pa")
    interface_line = (
        f'  interface = "{AIRPLAY_INTERFACE}";\n' if AIRPLAY_INTERFACE else ""
    )
    config_path.write_text(
        f"""general :
{{
  name = "{zone.announce_name}";
  port = {zone.port};
{interface_line}  output_backend = "{backend}";
  udp_port_base = {zone.udp_port_base};
  audio_backend_buffer_desired_length_in_seconds = 0.5;
}};
sessioncontrol :
{{
  allow_session_interruption = "yes";
}};
{backend} :
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
        # Explicit -o flag as defense-in-depth, not just belt-and-suspenders
        # for its own sake: confirmed directly (manual CLI test) that
        # shairport-sync's -o flag overrides whatever output_backend says
        # in the .conf file. Reusing the same env var build_shairport_config()
        # already uses, rather than a second hardcoded "pa", so the two
        # can never silently drift apart -- that class of "two things that
        # were supposed to agree, didn't, and nobody noticed" bug already
        # cost real time earlier this session.
        backend = os.environ.get("AIRPLAY_MULTIROOM_SPS_BACKEND", "pa")
        self._proc = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            "-a",
            self.zone.announce_name,
            "-p",
            str(self.zone.port),
            "-c",
            str(self.config_path),
            "-o",
            backend,
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


class AirplayMultiroomProvider(PlayerProvider):
    """AirPlay Multiroom Audio player provider.

    Properly subclasses PlayerProvider now that its real base (Provider)
    has been confirmed against actual source. Two stopgaps from earlier
    rounds are gone, not just fixed:
      - instance_id: turned out to be a read-only @property on Provider
        (`return self.config.instance_id`) -- the manual
        `self.instance_id = ...` assignment from before wouldn't just be
        redundant now, it would actively break (AttributeError: can't set
        attribute) once this class inherits the property. Removed
        entirely, not reimplemented.
      - get_config_entries(): Provider already defines a real default
        (`async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        return ()`) -- the stopgap's `list` return was even the wrong
        type. No override needed at all unless this provider grows actual
        user-configurable settings later.

    unload()'s signature also needed a real fix, not just a guess: the
    base class takes `is_removed: bool = False`, which the previous
    version didn't have at all.
    """

    def __init__(self, mass, manifest, config) -> None:
        super().__init__(mass, manifest, config)
        self._processes: dict[str, AirplayMultiroomProcess] = {}
        self._airplay_infos: dict[str, object] = {}  # AsyncServiceInfo, kept as
        # `object` to avoid importing zeroconf at module scope just for a type hint

    async def discover_players(self) -> None:
        """Discover and register players for this provider.

        Overriding this method (rather than mDNS-style auto-discovery) is
        confirmed correct for this use case: PlayerProvider's own
        docstring for this method says "For providers that support
        dynamic discovery of players via mdns, there is no need to
        implement this method" -- meaning it's specifically the intended
        override point for a provider that, like this one, deliberately
        skips mDNS and registers its players directly.

        Body wrapped in try/except purely for diagnostics: MA's own task
        wrapper logs "Exception in task ... target: <coroutine>:" with
        nothing after the colon when this fails, no traceback, no message
        -- not something this file controls. LOGGER.exception() here
        guarantees the real error actually gets logged somewhere, whatever
        MA's own wrapper does or doesn't show. Re-raises unchanged so
        MA's own error handling/state (provider load failure, etc.) still
        happens exactly as it would have.
        """
        try:
            await self._discover_players_impl()
        except Exception:
            LOGGER.exception(
                "discover_players() failed -- full traceback follows "
                "(MA's own task-exception log line for this doesn't "
                "include one)"
            )
            raise

    async def _discover_players_impl(self) -> None:
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

            # Was: registering our own AirplayMultiroomPlayer here directly.
            # Replaced with this: that class declares zero supported_features
            # (see its own docstring) -- it was never able to actually play
            # anything. The functional player has always been the one the
            # built-in AirPlayProvider registers via its own mDNS discovery
            # (confirmed: that's what "sound working on two players" was
            # actually playing through). This announces the record that lets
            # that discovery happen fast instead of ~10s/device, rather than
            # registering a second, non-functional, competing player entry.
            airplay_info = await announce_as_airplay_device(self.mass, zone)
            if airplay_info is not None:
                self._airplay_infos[sink_name] = airplay_info

            port += 1
            udp_base += 10

        # Newly-confirmed requirement (from the real Provider base class):
        # self.available starts False and doesn't flip automatically.
        # Setting it here, at the end of a successful discover_players(),
        # on the reasoning that "available" should mean "actually did the
        # setup work and has running processes" -- not yet confirmed
        # against how other providers decide exactly when to set this.
        self.available = True

    async def unload(self, is_removed: bool = False) -> None:
        await asyncio.gather(*(p.stop() for p in self._processes.values()))
        # No self._players to unregister anymore -- see discover_players()'s
        # comment on why native player registration was replaced with the
        # synthetic mDNS announcement. The built-in AirPlayProvider owns
        # unregistering its own players when their mDNS records disappear.
        #
        # This part closes a gap that was actually hit, not just a
        # theoretical one: without it, a synthetic _airplay._tcp record
        # (deterministically named per sink) survives across a
        # disable/enable cycle and collides with the next registration
        # attempt, raising zeroconf.NonUniqueNameException. The reclaim
        # logic in announce_as_airplay_device() recovers from that when it
        # happens, but not registering a stale record in the first place is
        # better than recovering from one every time.
        await asyncio.gather(
            *(
                self.mass.discovery.aiozc.async_unregister_service(info)
                for info in self._airplay_infos.values()
            ),
            return_exceptions=True,
        )
        self._airplay_infos.clear()
