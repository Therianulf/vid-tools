#!/bin/bash
# Cut segments from "24 to 52.mkv" into individual MP4 files (stream copy, no re-encode)

INPUT="srcs/ 24 to 52.mkv"
OUTDIR="output/segments_24to52"
mkdir -p "$OUTDIR"

segments=(
  "00:00:43 00:07:45"
  "00:07:58 00:15:06"
  "00:15:19 00:22:26"
  "00:30:09 00:37:05"
  "00:37:20 00:44:32"
  "00:44:40 00:51:45"
  "00:52:01 00:59:08"
  "00:59:21 01:06:26"
  "01:06:43 01:13:53"
  "01:14:02 01:21:10"
  "01:21:25 01:28:31"
  "01:28:50 01:35:47"
  "01:36:09 01:43:09"
  "01:43:29 01:50:35"
  "01:50:48 01:57:55"
  "01:58:06 02:05:18"
  "02:05:31 02:12:37"
  "02:12:48 02:19:50"
  "02:20:11 02:27:17"
  "02:27:29 02:34:32"
  "02:34:53 02:41:59"
  "02:42:10 02:49:18"
  "02:49:29 02:56:39"
  "02:56:52 03:03:52"
  "03:11:24 03:18:26"
  "03:18:49 03:25:40"
)

for i in "${!segments[@]}"; do
  num=$((i + 1))
  read -r start end <<< "${segments[$i]}"
  outfile=$(printf "%s/%02d.mp4" "$OUTDIR" "$num")
  echo "=== Segment $num: $start -> $end ==="
  ffmpeg -y -ss "$start" -to "$end" -i "$INPUT" \
    -c copy -tag:v hvc1 \
    "$outfile"
  echo ""
done

echo "Done! All segments saved to $OUTDIR/"
