# Trailer Cut

`trailer_cut.py` assembles a short trailer from a JSON spec. The spec lists the beats (a clip, still
shots with a slow push, an end card), the on-screen line for each beat, the voice-over, and the audio
numbers. The tool cuts the picture, lays the lines over it, mixes the voice, ducks an optional music
bed under the voice, and writes an MP4 plus a thumbnail. `trailer_cards.py` draws the text and end card
PNGs for it.

```
python trailer_cut.py --spec trailer-specs/coop-comms-v1.json \
  --take "$V/takes/beat1-take-01-gen45-5s.mp4" \
  --shots <roomMic build-app/store-shots> \
  --brand <coop-comms-website assets/brand> \
  --voice "$V/vo" [--music "$V/music/music-1.wav" [--music-offset 0.4]] \
  "$V/coop-comms-demo-v1-vo.mp4"
```

## Inputs

- **`--spec`** (required): the JSON spec. See below.
- **`--take`**: the clip for beats of kind `clip`. Required when the spec has a clip beat. Any size; it is scaled to the spec's size.
- **`--shots`**: the folder the still beats' `image` names resolve against. Required when the spec has a still beat.
- **`--brand`** (required): the folder with the wordmark SVG and the tagline font. The spec names both, relative to it.
- **`--voice`**: the folder the beats' voice `file` names resolve against. Each line starts `at` seconds into its beat. The cut refuses a line that is too long. Without it the track is silent.
- **`--music`**: one instrumental file. Needs `--voice`, because the bed is ducked under the voice. It must be at least `min_seconds` long, after the offset.
- **`--music-offset`**: seconds. Overrides the spec's `music.offset`. Positive skips that far into the music. Negative delays the music's start by that much.
- **`--line-font`**: overrides the spec's `line_font`.
- **`out`**: the MP4 to write. Default is `output/<name>.mp4`. An explicit path is used as given.
- **`--overwrite`**: replace an existing output or thumbnail. Without it the tool stops.

## Spec format

Shown with the coop-comms-v1 values.

| Key | Example | Meaning |
|-----|---------|---------|
| `name` | `coop-comms-v1` | Names the default output file. |
| `size` | `[1920, 1080]` | Width and height of the cut and every card. |
| `fps` | `30` | Frame rate. |
| `line_font` | `fonts/AtkinsonHyperlegibleNext.ttf` | Font for the on-screen lines. Relative paths resolve against the vid-tools folder. |
| `cards.background` | `#121519` | Band and end card colour. |
| `cards.text_color` | `#e9edf2` | Text colour. |
| `cards.line_size`, `cards.line_weight` | `64`, `600` | Size and variable-font weight of the lines. |
| `cards.tagline_size`, `cards.tagline_weight` | `60`, `600` | The same for the end card tagline. |
| `brand.wordmark` | `wordmark.svg` | Wordmark SVG, relative to `--brand`. |
| `brand.wordmark_width` | `1240` | Width in pixels the SVG is rasterised to. |
| `brand.tagline_font` | `fonts/Orbitron-wght.ttf` | Tagline font, relative to `--brand`. |
| `line_fade.at`, `line_fade.seconds` | `0.2`, `0.3` | When each line starts to fade in, and how long the fade takes. |
| `beats` | list | The beats, in order. See below. |
| `voice.lufs`, `voice.true_peak`, `voice.lra` | `-16`, `-1.5`, `11` | Loudness target for the voice mix. |
| `music.lufs`, `music.true_peak`, `music.lra` | `-24`, `-3`, `11` | Loudness target for the bed before ducking. |
| `music.min_seconds` | `14.5` | The shortest music the cut accepts. |
| `music.offset` | `0.0` | Default seconds into the music. |
| `music.fade_in`, `music.fade_out` | `0.3`, `0.8` | Fades on the bed. |
| `music.duck` | `threshold 0.03, ratio 4, attack 20, release 400` | The sidechain compressor that ducks the bed under the voice. |
| `music.limit` | `0.84` | Final limiter level (about -1.5 dBFS). |
| `encode` | `preset slow`, `profile high`, `video_bitrate 20M`, `maxrate 24M`, `bufsize 40M`, `audio_bitrate 192k`, `sample_rate 48000` | Encode settings. |
| `thumbnail_at` | `8.4` | Second of the cut to save as the thumbnail. Must be before the end. |

Each beat has these keys:

| Key | Meaning |
|-----|---------|
| `kind` | `clip`, `still` or `end_card`. |
| `seconds` | Length of the beat. The cut is the sum of all beats. |
| `start` | Clip only. Where in the take the beat begins. Default 0. |
| `image` | Still only. File name under `--shots`. |
| `push` | Still only. A slow zoom toward a point: `zoom` (default 0.15), `x` and `y` as fractions of the frame. |
| `fade_in` | Optional. Seconds of fade in from black. |
| `line` | Optional. The on-screen line, drawn as a band near the bottom. |
| `tagline` | End card only. Optional text under the wordmark. |
| `voice` | Optional. `file`, `at` (seconds into the beat) and `max` (longest allowed, default `seconds - 2 * at`). |
| `rings` | Optional. The broadcast rings overlay (`trailer_rings.py`); see Rings. |

## Rings

A beat can carry a `rings` block: the website's broadcast rings (coop-comms-website `js/main.js`, `initRings`).
Full circles grow from a centre, the Host half teal and the Guest half blue, born in phrases like speech.
`trailer_rings.py` renders them as PNG frames; the cut lays them over the beat with an ffmpeg `blend`
(screen by default) in RGB, under the on-screen line and before the beat's fade-in, so they glow over dark
areas and only brighten faces. Preview one beat alone with `python trailer_rings.py --spec <spec> [--beat N] [-o dir]`.

Every key except `center` has the default shown.

| Key | Default | Meaning |
|-----|---------|---------|
| `center` | required | `[x, y]` in output pixels. Must be inside the frame. |
| `host_side` | `left` | Which half gets the Host colour: `left` (pixels with x < centre x) or `right`. |
| `colors` | `["#5bd6b0", "#93b4ff"]` | `[Host, Guest]` colours. |
| `line_width` | `3` | Stroke width in output pixels, anti-aliased. |
| `opacity` | `0.55` | Peak opacity of the line. |
| `falloff` | `1.5` | Exponent of the fade: `(1 - r / max_radius) ** falloff`. |
| `start_radius` | `48` | Radius a ring is born at, in pixels. |
| `speed` | `108` | Growth in pixels per second. |
| `max_radius` | `null` | A ring is dropped at this radius. `null` is the distance to the farthest frame corner. |
| `target` | `null` | A point `[x, y]`, such as the farther PC. Sets `speed = distance / reach` and `max_radius = fade_ratio * distance`. Giving it together with `speed` or `max_radius` is an error. |
| `reach` | `1.0` | Seconds a ring takes to reach `target`. |
| `fade_ratio` | `1.2` | The ring is gone at this multiple of the distance to `target`. |
| `phrase.rings` | `[2, 5]` | Rings per phrase, `[min, max]` inclusive. |
| `phrase.gap` | `[0.18, 0.34]` | Seconds between rings inside a phrase. |
| `phrase.pause` | `[0.9, 1.8]` | Seconds of pause after a phrase. |
| `seed` | `1` | Random seed, so a render repeats exactly. |
| `first` | `0.0` | Beat-local second of the first ring; negative starts with rings already in flight. |
| `start`, `end` | `0.0`, `null` | Beat-local seconds the layer shows. `null` end is the beat's length. |
| `fade` | `0.0` | Seconds of fade in after `start` and fade out before `end`. |
| `glow` | `{"width": 20, "opacity": 0.3}` | A soft radial Gaussian halo around each ring, `width` its full width at half maximum in pixels. `null` or opacity 0 is none. |
| `blend` | `screen` | The ffmpeg blend mode used to lay the rings over the beat. |

The video tuning came from coop-comms-website via coop-comms-ad-ep1 on 2026-10-21: the centre sits on the
mic capsule; the Host colour is on the Host side; a ring reaches the farther PC in about 1 s (`target` plus
`reach`) and fades out at 1.2 times that distance; the line is about 3 px with peak opacity 0.5 to 0.6, screen
blend, and a soft glow about 20 px wide at 0.3; phrases are 2 to 5 rings 180 to 340 ms apart with a 0.9 to
1.8 s pause, the first phrase on the first spoken word (`first` 0.2); the rings are flat screen-space half
circles, never a Wi-Fi wedge; there is no hero glow. The `start_radius` 48 and `speed` 108 defaults are the
site's 40 and 90 scaled by 1.2 for a 900 px hero shown at 1080p. The line is 3 px because 1.5 CSS px shows as
3 device px on a retina screen, and a thinner line does not survive Steam's re-encode and small player.

## The beats

| Beat | Seconds | Picture | Line | Voice-over |
|------|---------|---------|------|------------|
| 1 | 0.0-3.0 | the Runway take | Two people, two PCs, one room. | Two people, two PCs, one room. |
| 2 | 3.0-6.0 | store shot 01, a push toward the status | The PC with the mic presses Host. | The PC with the mic presses Host. |
| 3 | 6.0-9.0 | store shot 03, a push toward Join | The other PC joins free. Press Join. | The other PC joins free. Press Join. |
| 4 | 9.0-12.5 | store shot 02, a push toward the status | You both hear it, near instantly. | You both hear it, near instantly. |
| 5 | 12.5-15.0 | the wordmark, with the tagline in Orbitron 600 | Hear the person next to you. | CO-OP Comms. Hear the person next to you. (If over 2.1 s: Hear the person next to you.) |

## Audio

The voice lines are placed on a silent track and mixed to -16 LUFS. The music bed is brought to
-24 LUFS and ducked under each voice line by a sidechain compressor fed from the voice. It
fades in over 0.3 s and out over the last 0.8 s. The mix goes through a limiter at -1.5 dBFS.

How deep the duck goes depends on the bed. On 2026-10-21, `music-1-full.wav` at `--music-offset 2.5`
measured: voice -15.8 LUFS while speaking, bed -32.1 LUFS under the voice and -23.5 LUFS in the gaps
(a duck of about 8.6 dB, the voice 16.3 LU over the bed), and the whole mix -16.8 LUFS with a true peak
of -1.5 dBFS.

## Output

- H.264 high profile, 1920x1080, 30 fps, 20 Mbps.
- AAC stereo at 48 kHz (silent without `--voice`).
- A thumbnail, `<out>-thumbnail-1920x1080.png`, cut from the video at `thumbnail_at`, because Steam's thumbnail must come from the video.

## Requirements

`ffmpeg`, `ffprobe`, `rsvg-convert` (`brew install librsvg`) and Python 3 with Pillow. This ffmpeg has no
`drawtext`, so the text is drawn as PNGs and laid over the picture. The lines are set with Pillow's raqm layout (kerning), which needs `libfribidi`
(`brew install fribidi`; Homebrew's librsvg already brings it). When the Python running the tool cannot find it, as
with the python.org build in our venv, the tool re-runs itself with `DYLD_FALLBACK_LIBRARY_PATH` pointing at
Homebrew's lib folder; without fribidi it stops rather than set the lines without kerning.

## Where the spec lives

The spec belongs with the commercial. The CO-OP Comms commercial repo (coop-comms-ad-ep1) owns it, at
`script/coop-comms-v1.json` there. The copy here is the worked example and the fixture
`tests/test_trailer.py` loads.

## Origin

This came from roomMic's feature-trailer-cut (a1f566f) on 2026-10-21.
