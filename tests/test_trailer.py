"""Tests for trailer_cut.py and trailer_cards.py (no ffmpeg needed)."""

import copy
import json
from pathlib import Path

from PIL import Image

from trailer_cards import render_cards
from trailer_cut import build_ffmpeg_args, check_voice_lengths, fmt

ROOT = Path(__file__).resolve().parent.parent
SPEC = json.loads((ROOT / "trailer-specs" / "coop-comms-v1.json").read_text())
TAKE = Path("/fake/take.mp4")
SHOTS = Path("/fake/shots")
TEXTS = {n: Path(f"/fake/work/text-{n}.png") for n in (1, 2, 3, 4)}
ENDS = {5: Path("/fake/work/end-5.png")}
VOICE = [(n, Path(f"/fake/vo/vo-{n}.wav")) for n in range(1, 6)]
MUSIC = Path("/fake/music.wav")
OUT = Path("/fake/out.mp4")


def build(voice=None, music=None, offset=0.0, spec=SPEC):
    return build_ffmpeg_args(spec, TAKE, SHOTS, TEXTS, ENDS, voice or [], music, offset, OUT)


def graph_of(args):
    return args[args.index("-filter_complex") + 1]


def input_paths(args):
    return [args[i + 1] for i, a in enumerate(args) if a == "-i"]


def test_fmt():
    assert fmt(3.0) == "3"
    assert fmt(3.5) == "3.5"
    assert fmt(-24) == "-24"
    assert fmt(0.03) == "0.03"
    assert fmt(round(15 - 0.8, 6)) == "14.2"


def test_no_voice():
    args = build()
    graph = graph_of(args)
    expected = [
        "zoompan=z='1+0.15*on/(90-1)':x='(iw-iw/zoom)*0.26':y='(ih-ih/zoom)*0.25':d=1:s=1920x1080:fps=30",
        "zoompan=z='1+0.15*on/(105-1)':x='(iw-iw/zoom)*0.26':y='(ih-ih/zoom)*0.37':d=1:s=1920x1080:fps=30",
        "[0:v]trim=0:3,setpts=PTS-STARTPTS,fps=30,scale=1920:1080:flags=lanczos,setsar=1,fade=t=in:st=0:d=0.4[b1]",
        "[4:v]fps=30,setsar=1,fade=t=in:st=0:d=0.3[b5]",
        "[5:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[t1]",
        "[8:v]format=rgba,fade=t=in:st=0.2:d=0.3:alpha=1[t4]",
        "[v1][v2][v3][v4][b5]concat=n=5:v=1:a=0,format=yuv420p[v]",
    ]
    for text in expected:
        assert text in graph
    assert args[args.index("-map", args.index("-map") + 1) + 1] == "9:a"
    inputs = input_paths(args)
    assert inputs[0] == str(TAKE)
    assert inputs[1:4] == [str(SHOTS / f"co-op-comms-screenshot-{s}-3840x2160.png") for s in ("01", "03", "02")]
    assert inputs[4] == str(ENDS[5])
    assert inputs[5:9] == [str(TEXTS[n]) for n in (1, 2, 3, 4)]
    assert inputs[9] == "anullsrc=r=48000:cl=stereo"
    null_at = args.index("anullsrc=r=48000:cl=stereo")
    assert args[null_at - 3 : null_at - 1] == ["-t", "15"]


def test_push_supersample():
    spec = copy.deepcopy(SPEC)
    spec["beats"][1]["push"]["supersample"] = 4
    graph = graph_of(build(spec=spec))
    assert (
        "[1:v]scale=iw*4:ih*4:flags=lanczos,zoompan=z='1+0.15*on/(90-1)':x='(iw-iw/zoom)*0.26'"
        ":y='(ih-ih/zoom)*0.25':d=1:s=1920x1080:fps=30,setsar=1[b2]"
    ) in graph
    assert "[2:v]zoompan=" in graph


def test_push_supersample_default_unchanged():
    assert "scale=iw*" not in graph_of(build())


def test_voice():
    args = build(voice=VOICE)
    graph = graph_of(args)
    for ms in (200, 3200, 6200, 9200, 12700):
        assert f"adelay={ms}|{ms}" in graph
    assert "amix=inputs=6:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]" in graph
    inputs = input_paths(args)
    assert inputs[10:15] == [str(p) for _n, p in VOICE]
    for k in range(5):
        assert f"[{10 + k}:a]aresample=48000" in graph


def test_voice_and_music():
    args = build(voice=VOICE, music=MUSIC)
    graph = graph_of(args)
    assert (
        "[15:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:15,asetpts=PTS-STARTPTS,"
        "loudnorm=I=-24:TP=-3:LRA=11,aresample=48000,afade=t=in:st=0:d=0.3,afade=t=out:st=14.2:d=0.8[mus]"
    ) in graph
    assert "[mus][sc]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=400[duck]" in graph
    assert "[vo][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.84:level=0,aresample=48000[a]" in graph
    assert input_paths(args)[15] == str(MUSIC)


def test_music_offset():
    assert "atrim=start=1.5,asetpts=PTS-STARTPTS,apad" in graph_of(build(voice=VOICE, music=MUSIC, offset=1.5))
    assert "adelay=500|500,apad" in graph_of(build(voice=VOICE, music=MUSIC, offset=-0.5))


def test_custom_spec():
    spec = copy.deepcopy(SPEC)
    spec["beats"] = [
        {"kind": "still", "seconds": 2.0, "image": "a.png"},
        {"kind": "end_card", "seconds": 1.0},
    ]
    args = build_ffmpeg_args(spec, None, SHOTS, {}, {2: ENDS[5]}, [], None, 0.0, OUT)
    graph = graph_of(args)
    assert "concat=n=2" in graph
    assert "fps=30,scale=1920:1080:flags=lanczos,setsar=1[b1]" in graph


def test_render_cards(tmp_path):
    font = ROOT / "fonts" / "AtkinsonHyperlegibleNext.ttf"
    mark = tmp_path / "mark.png"
    Image.new("RGBA", (200, 60), (255, 255, 255, 255)).save(mark)
    texts, ends = render_cards(SPEC, font, font, mark, tmp_path / "cards")
    assert sorted(texts) == [1, 2, 3, 4]
    assert sorted(ends) == [5]
    for path in texts.values():
        img = Image.open(path)
        assert img.mode == "RGBA"
        assert img.size == (1920, 1080)
        assert img.getpixel((0, 0))[3] == 0
    end = Image.open(ends[5])
    assert end.mode == "RGB"
    assert end.size == (1920, 1080)
    assert end.getpixel((0, 0)) == (0x12, 0x15, 0x19)


def test_check_voice_lengths():
    errors = check_voice_lengths(SPEC, {1: 2.7})
    assert len(errors) == 1
    assert "beat 1 allows 2.6 s" in errors[0]
    assert check_voice_lengths(SPEC, {1: 2.41}) == []
