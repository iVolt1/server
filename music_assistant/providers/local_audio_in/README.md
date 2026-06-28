# Local Audio In

A [Music Assistant](https://music-assistant.io) plugin provider that exposes PulseAudio hardware audio inputs — line-in, microphone, S/PDIF, and HDMI-in — as live audio sources playable on any MA player.

Sources appear automatically under **Home → Live Inputs** with no per-source configuration. Format details (sample rate, bit depth, channels) are shown in the source name.

---

## Requirements

- Music Assistant 2.x with plugin provider support
- PulseAudio or PipeWire (with PulseAudio compatibility layer)
- `ffmpeg` on the system PATH
- `pactl` (pulseaudio-utils) on the system PATH
- HAOS Music Assistant add-on satisfies all of the above out of the box

---

## Installation

Add the provider through **Settings → Providers → Add provider → Local Audio In**.

On HAOS the PulseAudio socket is detected automatically. No configuration is required to get started.

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| PulseAudio server address | *(auto-detect)* | Socket path or TCP address. Auto-detects `/run/audio/pulse.sock` (HAOS) or `/run/pulse/native`. Override only if PA is on a non-standard path. |
| Include monitor sources | Off | When enabled, sink monitor sources (loopbacks of audio outputs) appear in the source list alongside hardware inputs. |
| Source | *(all sources)* | Which PA source to expose. Leave empty to show all discovered sources. Set to a specific source to dedicate this instance to one input — see [Multiple instances](#multiple-instances) below. |

---

## Multiple instances

The provider can be added more than once, following the same pattern as the Spotify Connect provider. Each instance is independent with its own configuration.

The intended use is per-source settings: add one instance per physical input, set the **Source** dropdown to that input's PA source name, and configure the **Input gain** independently for each. This gives per-source gain control without any core MA changes.

Example setup for two inputs:

| Instance | Source |
|---|---|
| Local Audio In (X-Fi) | Creative X-Fi Analog Stereo |
| Local Audio In (HD Audio) | HD-Audio Generic Analog Stereo |

Each instance appears separately in the MA provider list and contributes its source to **Live Inputs**.

---

## How it works

At startup the provider calls `pactl list sources` to enumerate all available PulseAudio sources. Each qualifying source becomes an `AudioSource` item visible under **Home → Live Inputs**.

Source names include format details so you know exactly what you are selecting:

```
Creative X-Fi Analog Stereo (96000, 32, 2)
HD-Audio Generic Analog Stereo (48000, 32, 2)
```

Audio is captured via `ffmpeg` using the PulseAudio input backend:

- **24 and 32-bit sources** stream as raw PCM (`s32le`) — no encode step, exact bit depth preserved in the MA signal chain
- **16-bit sources** stream as FLAC at compression level 0 — lossless, self-framing, minimal CPU overhead

The provider uses the PA server sample rate rather than the raw hardware rate. This keeps source and output sink rates aligned, avoiding MA-side resampling through the lower-quality `swr` fallback when `libsoxr` is unavailable.

Volume normalization is automatically disabled by the MA core for `MediaType.AUDIO_SOURCE` streams. MA's dynamic loudness normalization is designed for music library content; applying it to a live input would measure quiet sources (idle line-in, low-level microphone) and apply severe boost, causing clipping at normal signal levels.

---

## Gain staging

ADC clipping happens before any software processing and cannot be corrected after the fact. If a source sounds distorted, reduce the analog output level on the source device first. The only controls that prevent clipping are the source device volume and the ALSA capture gain.

Check and set ALSA capture gain:

```bash
alsamixer   # F4 to switch to Capture view
```

Keep the ALSA Capture slider at 0 dB for line-level sources. If the captured signal is too quiet, increase the source device's output volume rather than the ALSA capture gain. Raising ALSA gain above 0 dB amplifies noise and risks clipping at the ADC.

Save ALSA state so it persists across reboots:

```bash
alsactl store
```

---

## PipeWire

PipeWire with the PulseAudio compatibility layer is fully supported. Source enumeration and capture work identically to native PulseAudio. Monitor source detection uses the `device.master_device` property rather than driver string matching for compatibility with PipeWire's source naming conventions.

---

## Known limitations

**Favorites and shortcuts** — `AudioSource` items are not favoritable or library-backed in MA core as of this writing. This is a current MA-wide limitation on the `AudioSource` type. Sources are reached through **Home → Live Inputs**. When the MA core adds favorites support for `AudioSource`, this provider will gain it automatically with no code change.

**Latency** — Capture-side latency is approximately 10–50 ms (PA fragment size). MA's internal stream pipeline adds a small additional buffer. The total end-to-end latency is suitable for monitoring but not for real-time performance applications.

---

## Related discussions

- [MA Discussion #1156](https://github.com/music-assistant/server/discussions/1156) — original local audio input feature request (62 upvotes)
- [MA Discussion #2343](https://github.com/music-assistant/server/discussions/2343) — Marcel's note on audio source architecture groundwork

---

## See also

- [local_audio](https://github.com/music-assistant/server/tree/dev/music_assistant/providers/local_audio) — the complementary output provider for PulseAudio sinks
- [MA PR #4384](https://github.com/music-assistant/server/pull/4384) — go-librespot Spotify Connect, the reference `PluginProvider`/`AudioSource` implementation this provider follows
