#!/usr/bin/env python3
"""
Trailer Cut
===========
Assembles a short trailer from a JSON spec: clip and still beats, on-screen lines, an end card,
voice-over, and an optional music bed that ducks under the voice. The spec holds the beat timings,
lines, voice offsets and audio numbers; nothing about the cut is hardcoded here.

Usage:
    python trailer_cut.py --spec trailer-specs/coop-comms-v1.json \\
      --take "$V/takes/beat1-take-01-gen45-5s.mp4" \\
      --shots <roomMic build-app/store-shots> \\
      --brand <coop-comms-website assets/brand> \\
      --voice "$V/vo" [--music "$V/music/music-1.wav" [--music-offset 0.4]] \\
      "$V/coop-comms-demo-v1-vo.mp4"

Requirements:
    ffmpeg, ffprobe, rsvg-convert (brew install librsvg), Pillow
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from trailer_cards import ensure_raqm, render_cards, resolve_line_font


def fmt(x):
    """A number as it is written into an ffmpeg argument: 3.0 -> "3", 3.5 -> "3.5"."""
    return f"{x:g}"


def beat_starts(spec):
    """Start time of each beat, in beat order."""
    starts = []
    t = 0.0
    for beat in spec["beats"]:
        starts.append(t)
        t += beat["seconds"]
    return starts


def total_seconds(spec):
    return sum(beat["seconds"] for beat in spec["beats"])


def voice_max(beat):
    """The longest a beat's voice line may run."""
    voice = beat["voice"]
    if "max" in voice:
        return voice["max"]
    return round(beat["seconds"] - 2 * voice["at"], 6)


def check_voice_lengths(spec, durations):
    """Error messages for voice lines that run too long. durations maps beat number to seconds."""
    errors = []
    for n, beat in enumerate(spec["beats"], start=1):
        if "voice" not in beat or n not in durations:
            continue
        limit = voice_max(beat)
        if durations[n] > limit:
            errors.append(f"{beat['voice']['file']} is {fmt(durations[n])} s; beat {n} allows {fmt(limit)} s")
    return errors


def fade_suffix(beat):
    return f",fade=t=in:st=0:d={fmt(beat['fade_in'])}" if "fade_in" in beat else ""


def video_graph(spec, cards_texts):
    """The video filter segments, ending in the [v] label."""
    w, h = spec["size"]
    fps = spec["fps"]
    fade = spec["line_fade"]
    segments = []
    labels = []
    for n, beat in enumerate(spec["beats"], start=1):
        i = n - 1
        kind = beat["kind"]
        if kind == "clip":
            start = beat.get("start", 0)
            chain = (
                f"[{i}:v]trim={fmt(start)}:{fmt(start + beat['seconds'])},setpts=PTS-STARTPTS,fps={fps},"
                f"scale={w}:{h}:flags=lanczos,setsar=1"
            )
        elif kind == "still" and "push" in beat:
            push = beat["push"]
            frames = round(beat["seconds"] * fps)
            chain = (
                f"[{i}:v]zoompan=z='1+{fmt(push.get('zoom', 0.15))}*on/({frames}-1)'"
                f":x='(iw-iw/zoom)*{fmt(push['x'])}':y='(ih-ih/zoom)*{fmt(push['y'])}'"
                f":d=1:s={w}x{h}:fps={fps},setsar=1"
            )
        elif kind == "still":
            chain = f"[{i}:v]fps={fps},scale={w}:{h}:flags=lanczos,setsar=1"
        else:
            chain = f"[{i}:v]fps={fps},setsar=1"
        segments.append(f"{chain}{fade_suffix(beat)}[b{n}]")
        labels.append(f"[b{n}]")
    text_index = len(spec["beats"])
    for n, beat in enumerate(spec["beats"], start=1):
        if "line" not in beat:
            continue
        segments.append(
            f"[{text_index}:v]format=rgba,fade=t=in:st={fmt(fade['at'])}:d={fmt(fade['seconds'])}:alpha=1[t{n}]"
        )
        text_index += 1
    for n, beat in enumerate(spec["beats"], start=1):
        if "line" in beat:
            segments.append(f"[b{n}][t{n}]overlay=shortest=1[v{n}]")
            labels[n - 1] = f"[v{n}]"
    segments.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,format=yuv420p[v]")
    return segments


def audio_graph(spec, null_index, voice_files, music_index, music_offset):
    """The audio filter segments and the map target."""
    enc = spec["encode"]
    rate = enc["sample_rate"]
    total = total_seconds(spec)
    if not voice_files:
        return [], f"{null_index}:a"
    starts = beat_starts(spec)
    segments = []
    labels = ""
    for k, (n, _path) in enumerate(voice_files):
        j = null_index + 1 + k
        ms = round((starts[n - 1] + spec["beats"][n - 1]["voice"]["at"]) * 1000)
        segments.append(f"[{j}:a]aresample={rate},aformat=channel_layouts=stereo,adelay={ms}|{ms}[vo{n}]")
        labels += f"[vo{n}]"
    v = spec["voice"]
    mix = (
        f"[{null_index}:a]{labels}amix=inputs={len(voice_files) + 1}:duration=first:normalize=0,"
        f"loudnorm=I={fmt(v['lufs'])}:TP={fmt(v['true_peak'])}:LRA={fmt(v['lra'])},aresample={rate}"
    )
    if music_index is None:
        segments.append(f"{mix}[a]")
        return segments, "[a]"
    m = spec["music"]
    segments.append(f"{mix},asplit=2[vo][sc]")
    if music_offset > 0:
        offset = f"atrim=start={fmt(music_offset)},asetpts=PTS-STARTPTS,"
    elif music_offset < 0:
        ms = round(-music_offset * 1000)
        offset = f"adelay={ms}|{ms},"
    else:
        offset = ""
    fade_out_start = fmt(round(total - m["fade_out"], 6))
    segments.append(
        f"[{music_index}:a]aresample={rate},aformat=channel_layouts=stereo,{offset}apad,atrim=0:{fmt(total)},"
        f"asetpts=PTS-STARTPTS,loudnorm=I={fmt(m['lufs'])}:TP={fmt(m['true_peak'])}:LRA={fmt(m['lra'])},"
        f"aresample={rate},afade=t=in:st=0:d={fmt(m['fade_in'])},afade=t=out:st={fade_out_start}:d={fmt(m['fade_out'])}[mus]"
    )
    d = m["duck"]
    segments.append(
        f"[mus][sc]sidechaincompress=threshold={fmt(d['threshold'])}:ratio={fmt(d['ratio'])}"
        f":attack={fmt(d['attack'])}:release={fmt(d['release'])}[duck]"
    )
    segments.append(
        f"[vo][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit={fmt(m['limit'])}:level=0,aresample={rate}[a]"
    )
    return segments, "[a]"


def build_ffmpeg_args(spec, take, shots, cards_texts, cards_ends, voice_files, music, music_offset, out):
    """The full ffmpeg argv for the cut."""
    fps = spec["fps"]
    enc = spec["encode"]
    total = total_seconds(spec)
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    def looped(seconds, path):
        return ["-loop", "1", "-framerate", str(fps), "-t", fmt(seconds), "-i", str(path)]

    for n, beat in enumerate(spec["beats"], start=1):
        if beat["kind"] == "clip":
            args += ["-i", str(take)]
        elif beat["kind"] == "still":
            args += looped(beat["seconds"], Path(shots) / beat["image"])
        else:
            args += looped(beat["seconds"], cards_ends[n])
    for n, beat in enumerate(spec["beats"], start=1):
        if "line" in beat:
            args += looped(beat["seconds"], cards_texts[n])
    null_index = len(spec["beats"]) + sum(1 for beat in spec["beats"] if "line" in beat)
    args += ["-f", "lavfi", "-t", fmt(total), "-i", f"anullsrc=r={enc['sample_rate']}:cl=stereo"]
    for _n, path in voice_files:
        args += ["-i", str(path)]
    music_index = None
    if music is not None:
        music_index = null_index + 1 + len(voice_files)
        args += ["-i", str(music)]

    a_segments, a_map = audio_graph(spec, null_index, voice_files, music_index, music_offset)
    graph = ";\n".join(video_graph(spec, cards_texts) + a_segments)
    args += ["-filter_complex", graph, "-map", "[v]", "-map", a_map, "-t", fmt(total), "-r", str(fps)]
    args += ["-c:v", "libx264", "-preset", enc["preset"], "-profile:v", enc["profile"], "-pix_fmt", "yuv420p"]
    args += ["-b:v", enc["video_bitrate"], "-maxrate", enc["maxrate"], "-bufsize", enc["bufsize"]]
    args += ["-c:a", "aac", "-b:a", enc["audio_bitrate"], "-ar", str(enc["sample_rate"]), "-ac", "2"]
    args += ["-movflags", "+faststart", str(out)]
    return args


def probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def fail(message, code=1):
    print(message, file=sys.stderr)
    sys.exit(code)


def main():
    ensure_raqm()
    parser = argparse.ArgumentParser(description="Assemble a trailer from a JSON spec.")
    parser.add_argument("--spec", required=True, type=Path, help="the trailer JSON spec")
    parser.add_argument("--take", type=Path, help="the clip for beats of kind clip")
    parser.add_argument("--shots", type=Path, help="folder the still beats' image names resolve against")
    parser.add_argument("--brand", required=True, type=Path, help="folder with the wordmark and tagline font")
    parser.add_argument("--voice", type=Path, help="folder the beats' voice file names resolve against")
    parser.add_argument("--music", type=Path, help="music bed (needs --voice)")
    parser.add_argument("--music-offset", type=float, help="seconds into the music (negative delays its start)")
    parser.add_argument("--line-font", type=Path, help="font for the on-screen lines (overrides the spec)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output or thumbnail")
    parser.add_argument("out", nargs="?", type=Path, help="the .mp4 to write (default: output/<name>.mp4)")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    beats = spec["beats"]
    if any(b["kind"] == "clip" for b in beats) and args.take is None:
        parser.error("the spec has a clip beat: --take is required")
    if any(b["kind"] == "still" for b in beats) and args.shots is None:
        parser.error("the spec has a still beat: --shots is required")
    if args.music is not None and args.voice is None:
        parser.error("music needs --voice: the bed is ducked under the voice")

    w, h = spec["size"]
    total = total_seconds(spec)
    if args.out is None:
        out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{spec['name']}.mp4"
    else:
        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
    thumb = out.with_name(f"{out.stem}-thumbnail-{w}x{h}.png")

    # 1. Tools.
    for tool in ("ffmpeg", "ffprobe", "rsvg-convert"):
        if shutil.which(tool) is None:
            fail(f"needs {tool} on PATH")

    # 2. Existing outputs.
    if not args.overwrite:
        for path in (out, thumb):
            if path.exists():
                fail(f"exists: {path} (pass --overwrite to replace it)")

    # 3. Required files.
    line_font = resolve_line_font(spec, args.line_font)
    wordmark = args.brand / spec["brand"]["wordmark"]
    tagline_font = args.brand / spec["brand"]["tagline_font"]
    required = []
    if any(b["kind"] == "clip" for b in beats):
        required.append(args.take)
    required += [args.shots / b["image"] for b in beats if b["kind"] == "still"]
    required += [wordmark, tagline_font, line_font]
    voice_files = []
    if args.voice is not None:
        voice_files = [(n, args.voice / b["voice"]["file"]) for n, b in enumerate(beats, start=1) if "voice" in b]
        required += [path for _n, path in voice_files]
    if args.music is not None:
        required.append(args.music)
    missing = [path for path in required if not path.is_file()]
    if missing:
        for path in missing:
            print(f"missing: {path}", file=sys.stderr)
        sys.exit(1)

    # 4. Voice lengths.
    if voice_files:
        errors = check_voice_lengths(spec, {n: probe_duration(path) for n, path in voice_files})
        if errors:
            fail("\n".join(errors))

    # 5. Music length.
    offset = 0.0
    if args.music is not None:
        offset = args.music_offset if args.music_offset is not None else spec["music"].get("offset", 0)
        duration = probe_duration(args.music)
        minimum = spec["music"]["min_seconds"]
        if duration - offset < minimum:
            fail(f"music is {fmt(duration)} s (offset {fmt(offset)}); the cut needs at least {fmt(minimum)} s")

    # 6. Thumbnail time.
    if not spec["thumbnail_at"] < total:
        fail(f"thumbnail_at {fmt(spec['thumbnail_at'])} must be before the end of the cut ({fmt(total)} s)")

    with tempfile.TemporaryDirectory() as work_name:
        work = Path(work_name)
        print("Rasterising the wordmark")
        wordmark_png = work / "wordmark.png"
        subprocess.run(
            ["rsvg-convert", "-w", str(spec["brand"]["wordmark_width"]), str(wordmark), "-o", str(wordmark_png)],
            check=True,
        )
        print("Rendering cards")
        texts, ends = render_cards(spec, line_font, tagline_font, wordmark_png, work)
        print(f"Cutting {out}")
        cmd = build_ffmpeg_args(spec, args.take, args.shots, texts, ends, voice_files, args.music, offset, out)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            fail(f"ffmpeg failed (exit {e.returncode})")

    print(f"Thumbnail {thumb}")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", fmt(spec["thumbnail_at"]), "-i", str(out)]
        + ["-frames:v", "1", str(thumb)],
        check=True,
    )
    subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,r_frame_rate"]
        + ["-of", "compact=p=0", str(out)],
        check=True,
    )


if __name__ == "__main__":
    main()
