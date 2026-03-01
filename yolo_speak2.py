import os
import sys
import time
import glob
import argparse
import subprocess
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model (.pt or .engine)")
    p.add_argument(
        "--source",
        required=True,
        help='image file, folder, video file, usb0, picamera0, or an rtsp/http url',
    )
    p.add_argument("--thresh", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--resolution", default=None, help="WxH like 640x480 (capture and/or resize)")
    p.add_argument("--speak_every", type=float, default=2.0, help="min seconds between spoken updates")
    p.add_argument("--cooldown", type=float, default=5.0, help="min seconds between repeating same class")
    p.add_argument("--max_speak", type=int, default=3, help="max classes to speak per update")
    p.add_argument("--voice", default="en-us", help="espeak-ng voice")
    p.add_argument("--rate", type=int, default=175, help="espeak-ng words per minute")
    p.add_argument("--no_speak", action="store_true", help="disable speaking")
    p.add_argument("--show", action="store_true", help="show window with boxes")
    p.add_argument("--window", default="960x720", help="initial window size WxH when --show (default 960x720)")
    return p.parse_args()


def parse_wxh(s):
    if not s:
        return None
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise ValueError(f"Bad WxH format: {s}")


def choose_tts():
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    return exe


def pluralize(name, n):
    if n == 1:
        return name
    if name.endswith("s"):
        return name
    return name + "s"


def open_capture(source_type, img_source, usb_idx, resW, resH):
    if source_type in ("video", "usb", "url"):
        cap_arg = img_source if source_type in ("video", "url") else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if resW and resH:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open capture: {img_source}")
        return cap
    return None


def main():
    args = parse_args()

    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    speak_every = float(args.speak_every)
    cooldown = float(args.cooldown)

    if not os.path.exists(model_path):
        print("ERROR: model not found:", model_path)
        sys.exit(1)

    resW = resH = None
    if args.resolution:
        resW, resH = parse_wxh(args.resolution)

    winW, winH = parse_wxh(args.window) if args.window else (960, 720)

    img_ext_list = {".jpg", ".jpeg", ".png", ".bmp"}
    vid_ext_list = {".avi", ".mov", ".mp4", ".mkv", ".wmv"}

    usb_idx = None
    source_type = None

    if os.path.isdir(img_source):
        source_type = "folder"
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        ext = ext.lower()
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
    elif img_source.startswith("rtsp://") or img_source.startswith("http://") or img_source.startswith("https://"):
        source_type = "url"
    else:
        print("Invalid source:", img_source)
        sys.exit(1)

    tts_exe = None
    speak_enabled = (not args.no_speak)
    if speak_enabled:
        tts_exe = choose_tts()
        if not tts_exe:
            print("NOTE: espeak-ng/espeak not found. Speech disabled.")
            speak_enabled = False

    def speak(text):
        if not speak_enabled:
            return
        subprocess.Popen(
            [tts_exe, "-v", args.voice, "-s", str(args.rate), text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print("Loading model:", model_path)
    model = YOLO(model_path, task="detect")
    labels = model.names

    imgs_list = []
    cap = None
    picam = None
    resize_frame = bool(resW and resH)

    try:
        if source_type == "image":
            imgs_list = [img_source]
        elif source_type == "folder":
            for f in glob.glob(os.path.join(img_source, "*")):
                _, e = os.path.splitext(f)
                if e.lower() in img_ext_list:
                    imgs_list.append(f)
            imgs_list.sort()
        elif source_type in ("video", "usb", "url"):
            cap = open_capture(source_type, img_source, usb_idx, resW, resH)
        elif source_type == "picamera":
            from picamera2 import Picamera2
            picam = Picamera2()
            if not (resW and resH):
                resW, resH = 640, 480
                resize_frame = True
            picam.configure(
                picam.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)})
            )
            picam.start()

        bbox_colors = [
            (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
            (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
        ]

        if args.show:
            cv2.namedWindow("YOLO Speak", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLO Speak", int(winW), int(winH))

        last_spoken_t = 0.0
        last_class_spoken = defaultdict(lambda: 0.0)

        img_i = 0
        fps_buf = []
        fps_avg_len = 120

        while True:
            loop_t0 = time.perf_counter()

            if source_type in ("image", "folder"):
                if img_i >= len(imgs_list):
                    break
                frame = cv2.imread(imgs_list[img_i])
                img_i += 1
                if frame is None:
                    continue
            elif source_type in ("video", "usb", "url"):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            elif source_type == "picamera":
                frame = picam.capture_array()
                if frame is None:
                    break
            else:
                break

            if resize_frame and resW and resH:
                frame = cv2.resize(frame, (resW, resH), interpolation=cv2.INTER_LINEAR)

            results = model(frame, verbose=False)
            boxes = results[0].boxes

            seen = []
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    conf = float(boxes[i].conf.item())
                    if conf < min_thresh:
                        continue
                    classidx = int(boxes[i].cls.item())
                    name = labels.get(classidx, str(classidx))
                    seen.append(name)

                    if args.show:
                        xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                        xmin, ymin, xmax, ymax = xyxy
                        color = bbox_colors[classidx % len(bbox_colors)]
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        text = f"{name} {conf:.2f}"
                        cv2.putText(
                            frame,
                            text,
                            (xmin, max(20, ymin - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            now = time.monotonic()
            counts = Counter(seen)

            if counts and (now - last_spoken_t) >= speak_every:
                parts = []
                for name, cnt in counts.most_common(args.max_speak):
                    if (now - last_class_spoken[name]) < cooldown:
                        continue
                    parts.append(f"{cnt} {pluralize(name, cnt)}")
                    last_class_spoken[name] = now

                if parts:
                    speak("I see " + " and ".join(parts))
                    last_spoken_t = now

            if args.show:
                loop_t1 = time.perf_counter()
                fps = 1.0 / max(1e-6, (loop_t1 - loop_t0))
                fps_buf.append(fps)
                if len(fps_buf) > fps_avg_len:
                    fps_buf.pop(0)
                fps_avg = float(np.mean(fps_buf))

                cv2.putText(
                    frame,
                    f"FPS: {fps_avg:.2f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("YOLO Speak", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break

        print("Done.")

    finally:
        if cap is not None:
            cap.release()
        if picam is not None:
            try:
                picam.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()