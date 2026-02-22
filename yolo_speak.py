import os
import sys
import time
import glob
import argparse
import subprocess
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help='Path to model (.pt or .engine)')
parser.add_argument("--source", required=True, help='image file, folder, video file, usb0, picamera0')
parser.add_argument("--thresh", default=0.5, help="confidence threshold")
parser.add_argument("--resolution", default=None, help='WxH like 640x480')
parser.add_argument("--speak_every", default=2.0, type=float, help="min seconds between spoken updates")
parser.add_argument("--cooldown", default=5.0, type=float, help="min seconds between repeating same class")
parser.add_argument("--voice", default="en-us", help="espeak-ng voice")
parser.add_argument("--rate", default=175, type=int, help="espeak-ng words per minute")
parser.add_argument("--show", action="store_true", help="show window with boxes")
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
speak_every = float(args.speak_every)
cooldown = float(args.cooldown)

if not os.path.exists(model_path):
    print("ERROR: model not found:", model_path)
    sys.exit(1)

model = YOLO(model_path, task="detect")
labels = model.names

img_ext_list = [".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"]
vid_ext_list = [".avi",".mov",".mp4",".mkv",".wmv",".MOV",".MP4",".MKV",".WMV"]

if os.path.isdir(img_source):
    source_type = "folder"
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = "image"
    elif ext in vid_ext_list:
        source_type = "video"
    else:
        print("Unsupported file:", ext)
        sys.exit(1)
elif img_source.startswith("usb"):
    source_type = "usb"
    usb_idx = int(img_source[3:])
elif img_source.startswith("picamera"):
    source_type = "picamera"
    picam_idx = int(img_source[8:])
else:
    print("Invalid source:", img_source)
    sys.exit(1)

resize = False
resW = resH = None
if user_res:
    resize = True
    resW, resH = map(int, user_res.split("x"))

if source_type == "image":
    imgs_list = [img_source]
elif source_type == "folder":
    imgs_list = []
    for f in glob.glob(img_source + "/*"):
        _, e = os.path.splitext(f)
        if e in img_ext_list:
            imgs_list.append(f)
    imgs_list.sort()
elif source_type in ("video","usb"):
    cap_arg = img_source if source_type == "video" else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == "picamera":
    from picamera2 import Picamera2
    cap = Picamera2()
    if not user_res:
        resW, resH = 640, 480
        resize = True
    cap.configure(cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

def speak(text: str):
    # non-blocking-ish: start a short process; keep it simple
    subprocess.Popen(
        ["espeak-ng", "-v", args.voice, "-s", str(args.rate), text],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

last_spoken_time = 0.0
last_class_spoken = defaultdict(lambda: 0.0)

img_count = 0
frame_rate_buffer = []
fps_avg_len = 120
avg_frame_rate = 0.0

while True:
    t_start = time.perf_counter()

    if source_type in ("image","folder"):
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
        if frame is None:
            continue
    elif source_type == "video":
        ret, frame = cap.read()
        if not ret:
            break
    elif source_type == "usb":
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            break
    elif source_type == "picamera":
        frame = cap.capture_array()
        if frame is None:
            break

    if resize and resW and resH:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Count classes above threshold
    seen = []
    for i in range(len(detections)):
        conf = float(detections[i].conf.item())
        if conf < min_thresh:
            continue
        classidx = int(detections[i].cls.item())
        seen.append(labels[classidx])

        if args.show:
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{labels[classidx]} {conf:.2f}"
            cv2.putText(frame, label, (xmin, max(20, ymin-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    now = time.time()
    counts = Counter(seen)

    # Speak summary every speak_every seconds, but avoid repeating same class too often
    if counts and (now - last_spoken_time) >= speak_every:
        # build a short phrase like: "I see 2 persons and 1 dog"
        parts = []
        for name, cnt in counts.most_common(3):
            if (now - last_class_spoken[name]) < cooldown:
                continue
            parts.append(f"{cnt} {name}" + ("" if cnt == 1 else "s"))
            last_class_spoken[name] = now

        if parts:
            phrase = "I see " + " and ".join(parts)
            speak(phrase)
            last_spoken_time = now

    if args.show:
        t_stop = time.perf_counter()
        fps = 1.0 / max(1e-6, (t_stop - t_start))
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = float(np.mean(frame_rate_buffer))

        cv2.putText(frame, f"FPS: {avg_frame_rate:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow("YOLO Speak", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

print("Done.")

if source_type in ("video","usb"):
    cap.release()
elif source_type == "picamera":
    cap.stop()
cv2.destroyAllWindows()