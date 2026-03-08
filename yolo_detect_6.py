#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO


def menu_select(title, options, default_index=0, allow_custom=False, custom_label="Enter custom value"):
    while True:
        print("")
        print(title)
        for i, opt in enumerate(options, start=1):
            marker = " (default)" if (i - 1) == default_index else ""
            print(f"  {i}. {opt}{marker}")

        if allow_custom:
            custom_num = len(options) + 1
            print(f"  {custom_num}. {custom_label}")

        raw = input("Select option number: ").strip()

        if raw == "":
            return options[default_index]

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            if allow_custom and idx == len(options) + 1:
                return None

        print("Invalid selection. Type a valid option number.")


def menu_yes_no(title, default=True):
    options = ["Yes", "No"]
    default_index = 0 if default else 1
    result = menu_select(title, options, default_index=default_index, allow_custom=False)
    return result == "Yes"


def prompt_text_required(label, default=None):
    while True:
        if default is None:
            raw = input(f"{label}: ").strip()
        else:
            raw = input(f"{label} [{default}]: ").strip()

        if raw == "":
            if default is not None:
                return str(default)
            print("Value required.")
            continue

        return raw


def prompt_float_required(label, default):
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Enter a valid number (example: 0.5 or 1.0).")


def choose_model_path(default_model=None):
    model_candidates = sorted(set(glob.glob("*.engine") + glob.glob("*.pt")))
    print("")
    print("Model selection")

    if model_candidates:
        print("Model files found in current folder:")
        for i, f in enumerate(model_candidates, start=1):
            print(f"  {i}. {f}")

        default_idx = 0
        if default_model and default_model in model_candidates:
            default_idx = model_candidates.index(default_model)

        choice = menu_select(
            "Choose model file",
            model_candidates,
            default_index=default_idx,
            allow_custom=True,
            custom_label="Enter custom model path",
        )
        if choice is not None:
            return choice

    return prompt_text_required("Custom model path", default_model or "yolo26s.pt")


def choose_source(default_source="usb0"):
    source_options = [
        "usb0",
        "usb1",
        "picamera0",
        "Video file path",
        "Image file path",
        "Image folder path",
        "Custom source string",
    ]

    if default_source in source_options:
        default_idx = source_options.index(default_source)
    else:
        default_idx = 0

    choice = menu_select("Source selection", source_options, default_index=default_idx)

    if choice == "Video file path":
        return prompt_text_required("Enter video file path")
    if choice == "Image file path":
        return prompt_text_required("Enter image file path")
    if choice == "Image folder path":
        return prompt_text_required("Enter image folder path")
    if choice == "Custom source string":
        return prompt_text_required("Enter custom source (example: usb0, picamera0, rtsp://..., file path)", default_source)

    return choice


def choose_resolution(default_resolution="640x480"):
    options = [
        "640x480",
        "1280x720",
        "1920x1080",
        "none (use source resolution)",
        "Custom WxH",
    ]

    default_idx = 0
    if default_resolution is None:
        default_idx = 3
    elif str(default_resolution) in ["640x480", "1280x720", "1920x1080"]:
        default_idx = ["640x480", "1280x720", "1920x1080"].index(str(default_resolution))

    choice = menu_select("Resolution selection", options, default_index=default_idx)

    if choice == "none (use source resolution)":
        return None
    if choice == "Custom WxH":
        return prompt_text_required("Enter resolution WxH", default_resolution or "640x480")

    return choice


def choose_threshold(default_thresh=0.5):
    options = ["0.25", "0.40", "0.50", "0.60", "0.75", "Custom threshold"]
    default_map = {"0.25": 0, "0.40": 1, "0.50": 2, "0.60": 3, "0.75": 4}
    default_key = f"{float(default_thresh):0.2f}" if default_thresh is not None else "0.50"
    default_idx = default_map.get(default_key, 2)

    choice = menu_select("Confidence threshold", options, default_index=default_idx)
    if choice == "Custom threshold":
        return prompt_float_required("Enter threshold", default_thresh if default_thresh is not None else 0.5)
    return float(choice)


def choose_presence_timeout(default_timeout=1.0):
    options = ["0.5", "1.0", "2.0", "5.0", "10.0", "Custom timeout"]
    default_map = {"0.5": 0, "1.0": 1, "2.0": 2, "5.0": 3, "10.0": 4}
    default_key = f"{float(default_timeout):0.1f}" if default_timeout is not None else "1.0"
    default_idx = default_map.get(default_key, 1)

    choice = menu_select("Presence OFF timeout (seconds)", options, default_index=default_idx)
    if choice == "Custom timeout":
        return prompt_float_required("Enter timeout seconds", default_timeout if default_timeout is not None else 1.0)
    return float(choice)


def choose_text_setting(title, current_value):
    choice = menu_select(
        f"{title} (current: {current_value})",
        ["Use current/default value", "Enter custom value"],
        default_index=0,
    )
    if choice == "Use current/default value":
        return current_value
    return prompt_text_required(f"Enter {title.lower()}", current_value)


def interactive_wizard(args):
    print("")
    print("YOLO Detect Setup Wizard")
    print("Choose options by number to avoid typos.")
    print("Press Enter to accept the default selection.")
    print("")

    args.model = choose_model_path(default_model=args.model or "yolo26s.engine")
    args.source = choose_source(default_source=args.source or "usb0")
    args.resolution = choose_resolution(default_resolution=args.resolution if args.resolution is not None else "640x480")
    args.thresh = choose_threshold(default_thresh=args.thresh if args.thresh is not None else 0.5)
    args.record = menu_yes_no("Record output video", default=bool(args.record))

    print("")
    print("MQTT / Home Assistant settings")

    args.ha_ip = choose_text_setting("MQTT broker IP", args.ha_ip or "192.168.1.52")
    args.mqtt_user = choose_text_setting("MQTT username", args.mqtt_user or "jtrdev")
    args.mqtt_pass = choose_text_setting("MQTT password", args.mqtt_pass or "1010Maxisthebest9911#")
    args.presence_topic = choose_text_setting("Presence topic", args.presence_topic or "home/orin/yolo/person_present")
    args.presence_class = choose_text_setting("Presence class", args.presence_class or "person")
    args.presence_off_timeout = choose_presence_timeout(default_timeout=args.presence_off_timeout if args.presence_off_timeout is not None else 1.0)
    args.mqtt_retain = menu_yes_no("Publish retained MQTT messages", default=bool(args.mqtt_retain))
    args.presence_debug = menu_yes_no("Enable presence debug prints", default=True if args.presence_debug else False)

    print("")
    print("Starting with these settings:")
    print(f"  model={args.model}")
    print(f"  source={args.source}")
    print(f"  resolution={args.resolution}")
    print(f"  thresh={args.thresh}")
    print(f"  record={args.record}")
    print(f"  ha_ip={args.ha_ip}")
    print(f"  mqtt_user={args.mqtt_user}")
    print(f"  presence_topic={args.presence_topic}")
    print(f"  presence_class={args.presence_class}")
    print(f"  presence_off_timeout={args.presence_off_timeout}")
    print(f"  mqtt_retain={args.mqtt_retain}")
    print(f"  presence_debug={args.presence_debug}")
    print("")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=None,
        help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
    )
    parser.add_argument(
        "--source",
        default=None,
        help='Image source: image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), "usb0", or "picamera0"',
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.5,
        help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help='Resolution WxH for display (example: "640x480"), otherwise match source resolution',
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help='Record results (video/camera only) to "demo1.avi" (requires --resolution)',
    )

    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Launch interactive setup wizard (also runs automatically when no arguments are provided)",
    )

    # MQTT presence publishing (uses mosquitto_pub)
    parser.add_argument("--ha-ip", default="192.168.1.52", help="MQTT broker IP (Home Assistant)")
    parser.add_argument("--mqtt-user", default="jtrdev", help="MQTT username")
    parser.add_argument("--mqtt-pass", default="1010Maxisthebest9911#", help="MQTT password")
    parser.add_argument(
        "--presence-topic",
        default="home/orin/yolo/person_present",
        help="Topic to publish ON/OFF",
    )
    parser.add_argument(
        "--presence-class",
        default="person",
        help='Class name for presence (default "person")',
    )
    parser.add_argument(
        "--presence-off-timeout",
        type=float,
        default=1.0,
        help="Seconds with no presence-class before publishing OFF",
    )
    parser.add_argument(
        "--mqtt-retain",
        action="store_true",
        help="Publish retained MQTT messages (optional, useful for HA state topics)",
    )
    parser.add_argument(
        "--presence-debug",
        action="store_true",
        help="Print presence timing/debug messages",
    )

    return parser.parse_args(), parser


def mqtt_pub(host, user, pw, topic, msg, retain=False):
    cmd = [
        "mosquitto_pub",
        "-h", str(host),
        "-p", "1883",
        "-u", str(user),
        "-P", str(pw),
        "-t", str(topic),
        "-m", str(msg),
        "-V", "5",
    ]
    if retain:
        cmd.append("-r")

    r = subprocess.run(cmd, text=True, capture_output=True)
    print(f"[MQTT] publish topic={topic} msg={msg} rc={r.returncode}", flush=True)

    if r.returncode != 0:
        if r.stderr:
            print("[MQTT] stderr:", r.stderr.strip(), flush=True)
        if r.stdout:
            print("[MQTT] stdout:", r.stdout.strip(), flush=True)

    return r.returncode == 0


def parse_source_type(img_source):
    img_ext_list = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]
    vid_ext_list = [".avi", ".mov", ".mp4", ".mkv", ".wmv"]

    if os.path.isdir(img_source):
        return "folder", None
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            return "image", None
        elif ext in vid_ext_list:
            return "video", None
        else:
            print(f"File extension {ext} is not supported.")
            sys.exit(0)
    elif img_source.startswith("usb"):
        try:
            usb_idx = int(img_source[3:])
        except ValueError:
            print(f'Invalid USB camera source "{img_source}". Use format like usb0')
            sys.exit(0)
        return "usb", usb_idx
    elif img_source == "picamera0" or "picamera" in img_source:
        return "picamera", None
    else:
        print(f"Input {img_source} is invalid. Please try again.")
        sys.exit(0)


def main():
    args, parser = parse_args()

    no_cli_args = (len(sys.argv) == 1)

    if no_cli_args or args.wizard:
        interactive_wizard(args)

    if not args.model:
        parser.error("--model is required (or run with no args / --wizard)")
    if not args.source:
        parser.error("--source is required (or run with no args / --wizard)")

    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    user_res = args.resolution
    record = args.record

    if not os.path.exists(model_path):
        print("ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.")
        sys.exit(0)

    model = YOLO(model_path, task="detect")
    labels = model.names

    source_type, usb_idx = parse_source_type(img_source)

    resize = False
    resW = None
    resH = None
    if user_res:
        try:
            resW, resH = int(user_res.split("x")[0]), int(user_res.split("x")[1])
            resize = True
        except Exception:
            print('Invalid --resolution format. Use WxH, example: --resolution 640x480')
            sys.exit(0)

    recorder = None
    if record:
        if source_type not in ["video", "usb", "picamera"]:
            print("Recording only works for video and camera sources. Please try again.")
            sys.exit(0)
        if not user_res:
            print("Please specify resolution to record video at.")
            sys.exit(0)

        record_name = "demo1.avi"
        record_fps = 30
        recorder = cv2.VideoWriter(
            record_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            record_fps,
            (resW, resH),
        )

    cap = None
    imgs_list = None

    if source_type == "image":
        imgs_list = [img_source]

    elif source_type == "folder":
        imgs_list = []
        filelist = glob.glob(img_source + "/*")
        img_ext_list = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)
        imgs_list.sort()

        if len(imgs_list) == 0:
            print("No supported image files found in folder.")
            sys.exit(0)

    elif source_type in ["video", "usb"]:
        cap_arg = img_source if source_type == "video" else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            print("Unable to open video/camera source.")
            sys.exit(0)

        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    elif source_type == "picamera":
        if not user_res:
            print('Please specify --resolution WxH when using picamera0.')
            sys.exit(0)

        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)}))
        cap.start()

    bbox_colors = [
        (164, 120, 87),
        (68, 148, 228),
        (93, 97, 209),
        (178, 182, 133),
        (88, 159, 106),
        (96, 202, 231),
        (159, 124, 168),
        (169, 162, 241),
        (98, 118, 150),
        (172, 176, 184),
    ]

    avg_frame_rate = 0.0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    # Presence tracking
    presence_on = False
    last_present_time = 0.0
    presence_class = args.presence_class.strip().lower()
    presence_topic = args.presence_topic

    # Startup state
    mqtt_pub(
        args.ha_ip,
        args.mqtt_user,
        args.mqtt_pass,
        presence_topic,
        "OFF",
        retain=args.mqtt_retain,
    )

    try:
        while True:
            t_start = time.perf_counter()

            # Load frame
            if source_type in ["image", "folder"]:
                if img_count >= len(imgs_list):
                    print("All images have been processed. Exiting program.")
                    break
                img_filename = imgs_list[img_count]
                frame = cv2.imread(img_filename)
                img_count += 1
                if frame is None:
                    print(f"Failed to read image: {img_filename}")
                    continue

            elif source_type == "video":
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Reached end of the video file. Exiting program.")
                    break

            elif source_type == "usb":
                ret, frame = cap.read()
                if (frame is None) or (not ret):
                    print("Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.")
                    break

            elif source_type == "picamera":
                frame = cap.capture_array()
                if frame is None:
                    print("Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.")
                    break

            # Resize if requested
            if resize:
                frame = cv2.resize(frame, (resW, resH))

            # Inference
            results = model(frame, verbose=False)
            detections = results[0].boxes

            object_count = 0
            now = time.time()
            present_detected_this_frame = False

            # Draw detections
            for i in range(len(detections)):
                xyxy_tensor = detections[i].xyxy.cpu()
                xyxy = xyxy_tensor.numpy().squeeze()
                xmin, ymin, xmax, ymax = xyxy.astype(int)

                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                conf = float(detections[i].conf.item())

                if conf > min_thresh:
                    detected_name = str(classname).strip().lower()

                    if detected_name == presence_class:
                        present_detected_this_frame = True
                        last_present_time = now
                        if args.presence_debug:
                            print(f"[PRESENCE] seen {detected_name} conf={conf:.2f} at {now:.2f}", flush=True)

                    color = bbox_colors[classidx % len(bbox_colors)]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                    label = f"{classname}: {int(conf * 100)}%"
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(
                        frame,
                        (xmin, label_ymin - labelSize[1] - 10),
                        (xmin + labelSize[0], label_ymin + baseLine - 10),
                        color,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

                    object_count += 1

            # Presence transitions
            if present_detected_this_frame:
                if not presence_on:
                    print("[STATE] OFF->ON", flush=True)
                    mqtt_pub(
                        args.ha_ip,
                        args.mqtt_user,
                        args.mqtt_pass,
                        presence_topic,
                        "ON",
                        retain=args.mqtt_retain,
                    )
                    presence_on = True
            else:
                if presence_on:
                    elapsed = now - last_present_time
                    if args.presence_debug:
                        print(
                            f"[PRESENCE] no person this frame, elapsed={elapsed:.2f}s / timeout={args.presence_off_timeout:.2f}s",
                            flush=True,
                        )

                    if elapsed >= args.presence_off_timeout:
                        print("[STATE] ON->OFF", flush=True)
                        mqtt_pub(
                            args.ha_ip,
                            args.mqtt_user,
                            args.mqtt_pass,
                            presence_topic,
                            "OFF",
                            retain=args.mqtt_retain,
                        )
                        presence_on = False

            # FPS overlay
            if source_type in ["video", "usb", "picamera"]:
                cv2.putText(
                    frame,
                    f"FPS: {avg_frame_rate:0.2f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Object count overlay
            cv2.putText(
                frame,
                f"Number of objects: {object_count}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if presence_on:
                state_text = "Presence: ON"
            else:
                state_text = "Presence: OFF"

            cv2.putText(
                frame,
                state_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("YOLO detection results", frame)

            if record and recorder is not None:
                recorder.write(frame)

            # Key handling
            if source_type in ["image", "folder"]:
                key = cv2.waitKey()
            else:
                key = cv2.waitKey(5)

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("s") or key == ord("S"):
                cv2.waitKey()
            elif key == ord("p") or key == ord("P"):
                cv2.imwrite("capture.png", frame)

            # FPS calc
            t_stop = time.perf_counter()
            dt = (t_stop - t_start)
            if dt > 0:
                frame_rate_calc = float(1.0 / dt)

                if len(frame_rate_buffer) >= fps_avg_len:
                    frame_rate_buffer.pop(0)
                frame_rate_buffer.append(frame_rate_calc)

                avg_frame_rate = float(np.mean(frame_rate_buffer))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Force OFF on exit so HA light does not stick ON if script stops while active
        if presence_on:
            print("[STATE] EXIT -> OFF", flush=True)
            mqtt_pub(
                args.ha_ip,
                args.mqtt_user,
                args.mqtt_pass,
                presence_topic,
                "OFF",
                retain=args.mqtt_retain,
            )

        print(f"Average pipeline FPS: {avg_frame_rate:.2f}")

        if source_type in ["video", "usb"] and cap is not None:
            cap.release()
        elif source_type == "picamera" and cap is not None:
            try:
                cap.stop()
            except Exception:
                pass

        if record and recorder is not None:
            recorder.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()