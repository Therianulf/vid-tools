#!/bin/bash
# Cut segments from "raw 1to24.mkv" into individual MP4 files (stream copy, no re-encode)

INPUT="srcs/raw 1to24.mkv"
OUTDIR="output/segments"
mkdir -p "$OUTDIR"

segments=(
  "00:00:17 00:07:26"
  "00:07:33 00:14:46"
  "00:21:44 00:28:52"
  "00:32:56 00:40:08"
  "00:40:21 00:47:29"
  "00:55:43 01:02:55"
  "01:03:04 01:10:15"
  "01:10:25 01:17:36"
  "01:17:45 01:24:56"
  "01:25:04 01:32:17"
  "01:32:25 01:39:37"
  "01:39:46 01:46:57"
  "01:47:07 01:54:18"
  "01:54:27 02:01:40"
  "02:01:47 02:09:01"
  "02:09:08 02:16:20"
  "02:16:29 02:23:27"
  "02:23:35 02:30:47"
  "02:30:56 02:38:08"
  "02:38:17 02:45:28"
)

for i in "${!segments[@]}"; do
  num=$((i + 1))
  read -r start end <<< "${segments[$i]}"
  outfile=$(printf "%s/%02d.mp4" "$OUTDIR" "$num")
  echo "=== Segment $num: $start -> $end ==="
  ffmpeg -y -ss "$start" -to "$end" -i "$INPUT" \
    -c copy \
    "$outfile"
  echo ""
done

echo "Done! All segments saved to $OUTDIR/"
