# Remix Video for the iMessage / QuickTime Player

Old rips (AVI/DivX, MKV, WMV) won't play in iMessage, Messages.app, or QuickTime — wrong
container *and* wrong codecs. This is the process for making them play natively while
usually coming out **smaller** than the source.

## What Apple's player actually requires

| Layer | Required | Common failure |
|---|---|---|
| Container | `.mp4` / `.mov` | `.avi`, `.mkv`, `.webm` |
| Video | H.264 (`avc1`) or HEVC (`hvc1`) | MPEG-4 ASP (`DX50`/`XVID`), VP8/VP9 |
| Audio | AAC-LC (`mp4a`) | MP3, AC3, DTS, Vorbis, Opus |
| Pixel format | `yuv420p` | `yuv444p`, 10-bit |
| Layout | `moov` atom first (faststart) | `moov` at end → no streaming/scrub |

All five must be right. A file that satisfies four of them still won't open.

## Step 1 — Probe before you touch anything

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,codec_tag_string,\
profile,width,height,pix_fmt,r_frame_rate,channels,sample_rate,bit_rate \
  -of compact "$IN"
ffprobe -v error -show_entries format=format_name,duration,size,bit_rate -of compact "$IN"
```

**Decide from the probe, don't reflexively re-encode:**

- Already H.264 + AAC, just in the wrong container (or `moov` at the end)?
  → **Remux only.** Lossless and near-instant:
  ```bash
  ffmpeg -i "$IN" -c copy -movflags +faststart "$OUT"
  ```
- Wrong video or audio codec (the AVI/DivX case) → full re-encode, Step 2.
- Only the audio is wrong (e.g. H.264 video + AC3 audio) → re-encode audio, copy video:
  ```bash
  ffmpeg -i "$IN" -c:v copy -c:a aac -b:a 128k -movflags +faststart "$OUT"
  ```

Re-encoding a file that only needed a remux throws away quality for nothing.

## Step 2 — The re-encode

```bash
ffmpeg -y -i "$IN" \
  -map 0:v:0 -map 0:a:0 \
  -vf "hqdn3d=2:1:3:3" \
  -c:v libx264 -preset slower -tune animation -crf 25 \
  -profile:v high -level 4.0 -pix_fmt yuv420p \
  -c:a aac -b:a 112k -ac 2 -ar 48000 \
  -movflags +faststart \
  "$OUT"
```

Flag by flag:

- `-map 0:v:0 -map 0:a:0` — take exactly one video and one audio stream. Stops stray
  subtitle/attachment streams from failing the mux.
- `-vf hqdn3d=2:1:3:3` — **light** denoise. See "Why denoise" below. Omit for clean sources.
- `-preset slower` — better compression at the same CRF. Cheap at low resolution.
- `-tune animation` — only for cartoons/anime. Spends bits on line art rather than texture.
- `-crf 25` — quality target. Lower = bigger/better. See the CRF table below.
- `-profile:v high -level 4.0 -pix_fmt yuv420p` — the compatibility triple. Non-negotiable.
- `-b:a 112k -ac 2 -ar 48000` — stereo AAC; plenty for TV audio.
- `-movflags +faststart` — moves `moov` to the front so the file scrubs and streams.

### Why denoise makes it *both* smaller and better

Old DivX/XviD rips carry the original encoder's blocking and mosquito noise. That noise is
high-entropy, so a faithful re-encode spends a large share of its bitrate preserving
artifacts you don't want. A light `hqdn3d` strips them, which frees those bits for the
actual picture. The result is smaller *and* visually cleaner than the source.

**Do not denoise** a clean modern source (phone video, a render, a Blu-ray rip) — there you
are only destroying detail. Denoise is for cleaning up someone else's bad encode.

### Choosing CRF

| CRF | Use for |
|---|---|
| 20–21 | Live action you care about, high motion, film grain worth keeping |
| 22–23 | General-purpose default |
| 24–25 | Animation, low-resolution sources, flat color |
| 26–28 | Squeezing under a hard size limit; visible on detailed content |

### Content-specific flags

- **Animation** — `-tune animation`, CRF 24–25.
- **Live action** — drop `-tune animation`; CRF 21–23. Add `-tune film` for grainy film.
- **Downscaling** (only if the source is larger than you need for a phone):
  `-vf "scale=-2:1080"`. The `-2` keeps aspect ratio *and* an even height, which H.264
  requires. Odd dimensions are a hard encode failure.

## Step 3 — Verify (don't trust the exit code)

```bash
# Codec tags must read avc1 / mp4a
ffprobe -v error -show_entries stream=codec_name,codec_tag_string,profile,pix_fmt \
  -of compact "$OUT"

# Duration must match the source
ffprobe -v error -show_entries format=duration,size,bit_rate -of compact "$OUT"

# faststart check: moov must appear BEFORE mdat
xxd -l 100000 "$OUT" | grep -o -m1 -E 'moov|mdat'   # must print: moov
```

Visual A/B on a zoomed crop — the honest check that denoise didn't smear line art. Use
*output* seeking (`-i` before `-ss`) so both grabs land on the same frame:

```bash
T=00:07:30.5
ffmpeg -v error -y -i "$IN"  -ss $T -frames:v 1 a.png
ffmpeg -v error -y -i "$OUT" -ss $T -frames:v 1 b.png
ffmpeg -v error -y -i a.png -i b.png -filter_complex \
  "[0]crop=iw/2:ih/2:iw/4:ih/4,scale=iw*3:ih*3:flags=neighbor[a];\
   [1]crop=iw/2:ih/2:iw/4:ih/4,scale=iw*3:ih*3:flags=neighbor[b];[a][b]hstack" cmp.png
```

## Step 4 — Batch the season

Three concurrent jobs at `-threads 5` saturates a 16-core machine; one ffmpeg alone
leaves cores idle.

Encode to `$out.part` and rename only on success. A long batch *will* get interrupted, and
a plain `-f "$out"` skip check treats a half-written file as finished — so the resume
silently skips it and you ship a truncated episode. The rename makes the skip check mean
"complete", and makes the batch safe to re-run.

The `-f mp4` is **required** once you write to `.part`: ffmpeg picks its muxer from the
output file extension, and `.part` isn't one it recognises. Without it every job dies
instantly with `Unable to choose an output format`.

```bash
mkdir -p output
encode_one() {
  in="$1"; out="output/$(basename "${in%.*}").mp4"
  [ -f "$out" ] && { echo "SKIP $(basename "$out")"; return; }
  ffmpeg -y -hide_banner -loglevel error -threads 5 -i "$in" \
    -map 0:v:0 -map 0:a:0 -vf "hqdn3d=2:1:3:3" \
    -c:v libx264 -preset slower -tune animation -crf 25 \
    -profile:v high -level 4.0 -pix_fmt yuv420p \
    -c:a aac -b:a 112k -ac 2 -ar 48000 -movflags +faststart \
    -f mp4 "$out.part" && mv "$out.part" "$out" && echo "OK   $(basename "$out")" \
    || { rm -f "$out.part"; echo "FAIL $(basename "$in")"; }
}
export -f encode_one
ls srcs/*.avi | xargs -P 3 -I{} bash -c 'encode_one "$@"' _ {}
```

Throughput reference: ~7.5 hours of 512×384 source encoded in ~14 minutes wall-clock on a
16-core M-series machine, roughly 33× realtime at 3 concurrent jobs.

### Verify the whole batch

The check that matters is **duration**: a truncated encode is the one failure that still
produces a file that opens and plays. Compare every output against its source.

```bash
for f in output/*.mp4; do
  b=$(basename "$f" .mp4)
  sd=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "srcs/$b.avi")
  od=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  awk -v n="$b" -v a="$sd" -v o="$od" 'BEGIN{d=a-o; if(d<0)d=-d;
    print (o!="" && d<1.0 ? "OK      " : "TRUNCATED ") n "  src=" a " out=" o}'
done
```

And confirm the container/codec layer:

```bash
for f in output/*.mp4; do
  printf "%-16s %-6s %-5s %-5s %s\n" "$(basename $f)" \
    "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_tag_string -of csv=p=0 $f)" \
    "$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_tag_string -of csv=p=0 $f)" \
    "$(xxd -l 100000 $f | grep -o -m1 -E 'moov|mdat')" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 $f)"
done
```

## Gotchas

- **HEVC needs `-tag:v hvc1`.** If you encode with `libx265` for extra savings, QuickTime
  will refuse the file without `-tag:v hvc1` — ffmpeg defaults to `hev1`, which Apple won't
  open. H.264 has no such trap, which is why it's the default here.
- **iMessage attachment ceiling is ~100 MB.** Above that Messages punts to Mail Drop. Aim
  well under it; raise CRF before you drop resolution.
- **Odd dimensions fail.** Always scale with `-2` on one axis, never `-1`.
- **A non-standard output extension needs `-f`.** ffmpeg infers the container from the
  extension; any temp suffix like `.part` must be paired with an explicit `-f mp4`.
- **`xargs` swallows failures.** The batch exits 0 even when every job inside it failed,
  so a clean exit code proves nothing. Count the outputs and check durations instead.
- **Seeking for A/B frames:** `-ss` *before* `-i` is fast but snaps to a keyframe, so the
  two grabs land on different frames and the comparison is meaningless. Put `-ss` after
  `-i` for frame-accurate grabs.

## Reference result

Season 1, 41 episodes of 512×384 DivX (`DX50`) + MP3 in AVI — unplayable in iMessage:

| | Source | Output |
|---|---|---|
| Container | AVI | MP4, faststart |
| Video | MPEG-4 ASP `DX50` | H.264 `avc1` High@L4.0 |
| Audio | MP3 | AAC-LC `mp4a` |
| Bitrate | ~1262 kbps | ~630 kbps |
| Total size | 3.9 GB | ~1.9 GB |

Line art held up pixel-for-pixel at 3× zoom, and flat backgrounds came out cleaner than
the source because the DivX blocking was denoised away.
