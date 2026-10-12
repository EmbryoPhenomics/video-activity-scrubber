import argparse, json, os
import csv
import cv2
import numpy as np

def compute_activity_scores(video_path, window_s=1.0, diff_thresh=25, blur=5, sample_fps=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = (total_frames / fps) if total_frames else None

    # We sample at ~sample_fps to make it faster + smoother timeline
    step = max(int(round(fps / sample_fps)), 1)
    window_frames = max(int(round(window_s * fps / step)), 1)

    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Rolling window accumulator for a smooth-ish "activity level"
    buf = []
    times = []
    scores = []

    frame_idx = 0
    while True:
        # Skip frames to hit sample_fps
        for _ in range(step - 1):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += step
        t = frame_idx / fps

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if blur and blur > 0:
            k = (blur | 1)
            prev_bl = cv2.GaussianBlur(prev_g, (k, k), 0)
            g_bl = cv2.GaussianBlur(g, (k, k), 0)
        else:
            prev_bl, g_bl = prev_g, g

        diff = cv2.absdiff(g_bl, prev_bl)
        # binary motion mask
        _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # A simple scalar “motion amount”: proportion of pixels moving
        motion_frac = float((mask > 0).mean())

        buf.append(motion_frac)
        if len(buf) > window_frames:
            buf.pop(0)

        smooth_score = float(np.mean(buf))
        times.append(t)
        scores.append(smooth_score)

        prev_g = g

    cap.release()

    return {
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "duration_s": duration_s if duration_s is not None else (times[-1] if times else 0.0),
        "sample_dt": (times[1] - times[0]) if len(times) > 1 else (1.0 / sample_fps),
        "times": times,
        "scores": scores,
        "meta": {
            "window_s": window_s,
            "diff_thresh": diff_thresh,
            "blur": blur,
            "sample_fps": sample_fps,
        }
    }

def write_activity_csv(data, html_path):
    """Export beside the HTML output as <video_stem>_activity.csv.

    Each row corresponds to an existing timeline sample. frame_index is the
    zero-based source frame at time_s (frame_index / fps). activity_level is
    the rolling mean of the fraction of pixels whose grayscale difference
    exceeds diff_thresh, after optional blur; its range is 0 to 1. The first
    frame is only a comparison baseline, so no artificial time-zero row is
    added. Existing CSV output for this video is replaced on each run.
    """
    video_path = os.path.abspath(data["video_path"])
    video_name = os.path.basename(video_path)
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(html_path)),
        os.path.splitext(video_name)[0] + "_activity.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_path", "video_name", "frame_index", "time_s", "activity_level"
        ])
        for time_s, score in zip(data["times"], data["scores"]):
            writer.writerow([
                video_path, video_name, int(round(time_s * data["fps"])),
                time_s, score
            ])
    return csv_path


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Activity Scrubber</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
    .wrap { max-width: 1100px; margin: 0 auto; }
    video { width: 100%; border-radius: 12px; background: #111; }
    .row { margin-top: 14px; }
    canvas { width: 100%; height: 90px; border-radius: 12px; background: #0f0f0f; display:block; }
    .meta { color:#555; font-size: 13px; margin-top:8px; }
    .controls { display:flex; gap:10px; align-items:center; margin-top:8px; }
    input[type="range"]{ width: 260px; }
    button { padding:8px 10px; border-radius:10px; border:1px solid #ddd; background:#fff; cursor:pointer; }
  </style>
</head>
<body>
<div class="wrap">
  <h2>Video activity scrubber</h2>

  <video id="vid" controls preload="metadata"></video>

  <div class="row">
    <canvas id="timeline" width="1100" height="90"></canvas>
    <div class="controls">
      <button id="autoScale">Auto-scale</button>
      <span>Gain</span>
      <input id="gain" type="range" min="0.5" max="10" step="0.1" value="1.0">
      <span id="readout" style="margin-left:auto;color:#555;font-size:13px;"></span>
    </div>
    <div class="meta" id="meta"></div>
  </div>
</div>

<script>
const data = __DATA__;
const vid = document.getElementById("vid");
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const gainEl = document.getElementById("gain");
const readout = document.getElementById("readout");
const meta = document.getElementById("meta");

vid.src = "file://" + data.video_path.replaceAll("\\\\","/");
meta.textContent = `window=${data.meta.window_s}s, diff_thresh=${data.meta.diff_thresh}, blur=${data.meta.blur}, sample_fps=${data.meta.sample_fps}`;

let gain = parseFloat(gainEl.value);
let autoscale = true;
let scoreMax = Math.max(...data.scores) || 1e-6;

function fmtTime(s){
  const m = Math.floor(s/60);
  const ss = Math.floor(s%60).toString().padStart(2,'0');
  return `${m}:${ss}`;
}

function drawTimeline(playT=null){
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  // background
  ctx.fillStyle = "#0f0f0f";
  ctx.fillRect(0,0,W,H);

  const n = data.scores.length;
  if(n < 2) return;

  const maxV = autoscale ? scoreMax : 1.0;
  const scale = (maxV > 0 ? maxV : 1e-6);

  // draw bars
  for(let i=0;i<n;i++){
    const x0 = Math.floor(i * W / n);
    const x1 = Math.floor((i+1) * W / n);
    const v = Math.min((data.scores[i] * gain) / scale, 1.0);
    const barH = Math.floor(v * (H-18));
    // grayscale intensity
    const g = Math.floor(30 + 220*v);
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(x0, H-10-barH, Math.max(1,x1-x0), barH);
  }

  // baseline axis
  ctx.strokeStyle = "#333";
  ctx.beginPath();
  ctx.moveTo(0, H-10);
  ctx.lineTo(W, H-10);
  ctx.stroke();

  // playhead
  if(playT !== null){
    const x = Math.max(0, Math.min(W, (playT / data.duration_s) * W));
    ctx.strokeStyle = "#00c2ff";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // labels
  ctx.fillStyle = "#bbb";
  ctx.font = "12px system-ui";
  ctx.fillText("0", 6, 14);
  ctx.fillText(fmtTime(data.duration_s), W-60, 14);
}

function seekFromX(x){
  const t = (x / canvas.width) * data.duration_s;
  vid.currentTime = t;
}

canvas.addEventListener("click", (e)=>{
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  seekFromX(x);
});

gainEl.addEventListener("input", ()=>{
  gain = parseFloat(gainEl.value);
  drawTimeline(vid.currentTime);
});

document.getElementById("autoScale").addEventListener("click", ()=>{
  autoscale = !autoscale;
  drawTimeline(vid.currentTime);
});

vid.addEventListener("timeupdate", ()=>{
  const t = vid.currentTime || 0;
  // approximate score at time
  const idx = Math.min(data.scores.length-1, Math.max(0, Math.floor(t / data.sample_dt)));
  readout.textContent = `t=${fmtTime(t)} | activity=${data.scores[idx].toFixed(4)}`;
  drawTimeline(t);
});

vid.addEventListener("loadedmetadata", ()=>{
  // in case duration differs slightly
  if (isFinite(vid.duration) && vid.duration > 0) data.duration_s = vid.duration;
  drawTimeline(0);
});

drawTimeline(0);
</script>
</body>
</html>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--window_s", type=float, default=1.0, help="Smoothing window (seconds)")
    ap.add_argument("--diff_thresh", type=int, default=25)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--sample_fps", type=int, default=10)
    ap.add_argument("--out_prefix", default="activity")
    args = ap.parse_args()

    data = compute_activity_scores(
        args.video,
        window_s=args.window_s,
        diff_thresh=args.diff_thresh,
        blur=args.blur,
        sample_fps=args.sample_fps
    )

    json_path = f"{args.out_prefix}_scores.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    html_path = f"{args.out_prefix}_viewer.html"
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    csv_path = write_activity_csv(data, html_path)

    print("Wrote:")
    print(" ", os.path.abspath(json_path))
    print(" ", os.path.abspath(html_path))
    print(" ", csv_path)
    print("\nOpen the HTML file in a browser (Chrome/Edge usually best for local file video).")

if __name__ == "__main__":
    main()
