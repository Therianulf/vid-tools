# Frame Drift

`frame_drift.py` scores an AI video take, or one still, for drift: SSIM and the loop finder's similarity score
against an approved reference still and against the previous sample, over the whole frame and named crops such
as faces and props. We built it to judge Runway Gen-4.5 takes whose faces drift. The runwayTools session (an agent
driving the CLI) runs it on every new still batch and take, and a cheap sweep model and a judge model read the
result. The contract below is for any agent.

- stdout is exactly one JSON document, the same text that is written to `<out>/drift.json`.
- stderr carries all progress and error text.
- The exit code is the verdict: 0 clean, 2 a sample crossed a threshold, 1 usage or runtime error (stdout is empty on 1).

This tool puts its result on stdout and its progress on stderr, unlike the other tools, by Ben's call.

## Origin

It grew out of the loop finder, `video_looper.py --scan`, which Ben made when he built the background video on his
website. It reuses `iter_frames` (the frame reader, factored out of the two scans with their output unchanged) and
`frame_similarity` (reported as `loop_score_*`). It adds SSIM, the reference still, crops, thresholds, the contact
sheet and the JSON.

## Usage

```
python frame_drift.py [-h] [--reference PATH] [--every N | --fps F] [--start S] [--end S]
                      [--size WxH] [--crop NAME=X,Y,W,H] [--threshold SSIM]
                      [--prev-threshold SSIM] [--match-light] [-o DIR] INPUT
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `INPUT` | required | The take (.mp4, .mov) or one still (.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff). A still is detected by suffix, is one sample, and needs `--reference`. |
| `--reference PATH` | none | Approved still; adds SSIM vs reference checks. It is centre-cropped to the analysis aspect, then resized; the crop is recorded in `reference.crop_to_aspect`. |
| `--every N` | 6 | Sample every Nth frame. Exclusive with `--fps`. |
| `--fps F` | none | Sample F frames a second instead of `--every`. |
| `--start S` | 0 | First second to sample. |
| `--end S` | the end | Stop before this second. |
| `--size WxH` | native | Analysis size. Frames are centre-cropped to its aspect only if it differs; sample PNGs and crops use this size. |
| `--crop NAME=X,Y,W,H` | none | Named region in analysis pixels; repeatable. Names are letters, digits, `_` and `-`; `whole` is reserved. The crop must lie inside the frame. |
| `--threshold SSIM` | 0.6 | Flag SSIM vs reference below this. |
| `--prev-threshold SSIM` | 0.5 | Flag SSIM vs previous sample below this. |
| `--match-light` | off | Also score light-matched SSIM (`ssim_ref_lm`, `ssim_prev_lm`) and flag on it instead of the plain SSIM. The plain scores stay reported. |
| `-o, --out DIR` | `output/<input stem>-drift` | Output folder. Re-running replaces the earlier sample PNGs. |

A still ignores `--every`, `--fps`, `--start` and `--end`.

## Metrics

- `ssim_*`: Wang et al. 2004 on luma (OpenCV BGR2GRAY), Gaussian 11x11 sigma 1.5, K1 0.01, K2 0.03, edges reflected;
  the mean of the SSIM map over the whole frame or inside each crop. These are thresholded.
- `ssim_ref_lm`, `ssim_prev_lm`: with `--match-light`, the same SSIM after the sample's grey levels in the region are
  histogram-matched (CDF matching) to the reference's (or the previous sample's) same region. A light change that keeps
  the shapes scores near 1; shape drift still scores low. With `--match-light` these are thresholded instead of
  `ssim_ref` and `ssim_prev`, and the check names and the verdict say so; null otherwise.
- `loop_score_*`: the loop finder's `frame_similarity`, 40% grey histogram correlation plus 60% inverse MSE at 64x64.
  Reported, not thresholded.

SSIM measures structure, not identity. A talking mouth or a head turn lowers it like a face that becomes someone
else, so the flags say where to look, and the contact sheet and sample PNGs are what to look at.

## Why these defaults

On take-04 (against `beat1-redo-h-01`, at 1920x1080, every 6th frame):

- The whole frame never fell below 0.83 against the reference; the mic crop never below 0.78.
- The face crops start at 0.94 at frame 0 and are below 0.6 by 0.75 s (face_left) and 1.0 s (face_right), lowest about 0.30.
- The face crops score 0.47 to 0.94 between samples 0.25 s apart.
- `--prev-threshold` assumes `--every 6` at 24 fps; with smaller steps neighbouring frames score higher, so raise it.

Most talking takes will exit 2 on their face crops; the judge then looks. Variable-frame-rate input gives
approximate times (`frame / average fps`). A long take at `--every 1` grows the contact sheet without a cap; raise
`--every`.

## Output files

- `frames/sNNNN-fFFFFFF.png`: each sample at the analysis size (N is the sample index, F the frame number).
- `contact-sheet.png`: six thumbnails a row, the reference first. Under each, three label lines: index, frame and
  time; whole-frame SSIM vs reference and previous; the worst crop (or the loop score). A flagged sample has a red
  border and a red first line ending `FLAG`.
- `drift.json`: the document that is also on stdout.

## The JSON

| Key | Contents |
|---|---|
| `tool`, `version` | `frame_drift` and the document version. |
| `verdict` | One line: `CLEAN: ...` or `FLAGGED: ...` naming the first crossing and what else fired. |
| `exit_code` | 0 or 2. |
| `output_dir`, `drift_json`, `contact_sheet` | Absolute paths. |
| `legend` | What each metric and field means. |
| `input` | `path`, `kind` (video or still), `native_size`, `fps`, `total_frames`, `duration`, `crop_to_aspect`. |
| `reference` | null, or `path`, `native_size`, `crop_to_aspect`. |
| `settings` | `analysis_size`, `every`, `sample_fps`, `start`, `end`, `threshold`, `prev_threshold`, `match_light`, `ssim`. |
| `crops` | `[{name, box}]` in CLI order. |
| `summary` | `samples`, `flagged_samples`, `checks`, `metrics`, `worst`. |
| `flagged` | Flagged samples only: `index`, `frame`, `time`, `file`, `flags`. |
| `samples` | Every sample: `index`, `frame`, `time`, `file`, `flagged`, `flags`, `whole` and `crops` (each with `ssim_ref`, `ssim_prev`, `ssim_ref_lm`, `ssim_prev_lm`, `loop_score_ref`, `loop_score_prev`). |
| `samples[]` | A metric is null when it does not apply (no reference; the first sample has no previous one). |
| `summary.checks` | Per `region:metric`: `threshold`, `failed`, `first_below` (`index`, `frame`, `time`, or null). |
| `summary.metrics` | Per region and metric: `min`, `mean`, `min_index`. |
| `summary.worst` | Up to 5 samples by lowest SSIM vs reference (previous without a reference): `index`, `frame`, `time`, `file`, `check`, `value`. |

## Examples for the runwayTools session

Each "Read:" starts from the exit code, then `verdict`, then the named fields.

(a) Still batch. The tool takes one still as INPUT, so loop:

```
for s in "$T"/beat1-redo-i-0?-gen4image.png; do
  python frame_drift.py "$s" --reference "$T/beat1-redo-h-01-gen4image.png" --size 1920x1080 \
    --crop face_left=400,240,240,260 --crop face_right=1320,250,250,260 --crop mic=900,620,130,160 \
    -o "output/stills-drift/$(basename "$s" .png)" > "output/stills-drift-$(basename "$s" .png).json"
  echo "$s exit $?"
done
```

Read: the exit code per still; `verdict`; `samples[0].whole.ssim_ref` and `samples[0].crops.*.ssim_ref` (the
`ssim_prev` values are null for a still); `contact_sheet` shows the reference beside the still. Crops only mean
something when the new still keeps the reference's layout. Always pass the face crops for a still check: without
crops, two different stills of the same layout can pass on the whole frame alone (our tester measured
`beat1-redo-h-01` against `beat1-redo-i-03` at whole-frame SSIM 0.67, above the 0.6 threshold, so CLEAN).

(b) A take against its source still, with face and mic crops:

```
python frame_drift.py "$T/beat1-take-04-gen45-5s.mp4" --reference "$T/beat1-redo-h-01-gen4image.png" --size 1920x1080 \
  --crop face_left=400,240,240,260 --crop face_right=1320,250,250,260 --crop mic=900,620,130,160 \
  -o output/beat1-take-04-drift
```

Read: exit 2 or 0; `verdict` names the first crop and time; `flagged[]` lists each flagged sample's time, file and
failed checks; `summary.checks["face_left:ssim_ref"].first_below`; `summary.metrics.face_left.ssim_ref.min`;
`summary.worst`; then open `contact_sheet`. For takes whose light changes across the faces (game light, flicker), add
`--match-light`, and read both `ssim_*_lm` (flagged) and the plain `ssim_*` (reported): a face whose plain SSIM falls
while its light-matched SSIM holds is light, not drift.

(c) A take alone, frame to frame:

```
python frame_drift.py "$T/beat1-take-04-gen45-5s.mp4" --crop face_left=267,160,160,173 --crop face_right=880,167,167,173 --crop mic=600,413,87,107 -o output/beat1-take-04-prev
```

The same crops in native 1280x720 pixels, since there is no `--size`. Read: only `*:ssim_prev` checks exist;
`verdict`; `flagged[]`; `summary.metrics.*.ssim_prev.min`. A dip in one sample and recovery in the next is a jump.

## Requirements

Python 3 with opencv-python and numpy, plus torch because it imports `video_looper.py`. No ffmpeg to run; the tests
use ffmpeg to make clips.

Run `frame_drift.py` from a vid-tools checkout that has it (the frame-drift branch until it merges, then main), so
it imports the `video_looper.py` beside it. The vid-tools venv carries an editable install that points at
`~/Github/vid-tools`; a copy of the script run from anywhere else imports that checkout's `video_looper.py`, which
may lack `iter_frames` (runwayTools hit this with a git-archive copy on 2026-10-21).
