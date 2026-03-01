#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time
import json
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True, help='Path to YOLO model file (.pt or .engine)')
    p.add_argument(
        "--source",
        required=True,
        help='Image source: image file, folder, video file, "usb0", "picamera0", or rtsp/http URL'
    )
    p.add_argument("--thresh", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--resolution", default=None, help="WxH like 640x480 (capture and/or resize)")
    p.add_argument("--record", action="store_true", help='Record video to "demo1.avi" (requires --resolution)')

    p.add_argument("--presence-class", default="person", help='Class name to treat as "present" (default: person)')
    p.add_argument("--enter-frames", type=int, default=3, help="frames required to switch to PRESENT")
    p.add_argument("--leave-frames", type=int, default=30, help="frames required to switch to NOT present")

    p.add_argument("--mqtt-host", default=None, help="MQTT broker host/ip (enable MQTT publishing)")
    p.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    p.add_argument("--mqtt-user", default=None, help="MQTT username")
    p.add_argument("--mqtt-pass", default=None, help="MQTT password")
    p.add_argument("--mqtt-topic", default="vision/living_room/person_present", help="MQTT topic for presence ON/OFF")
    p.add_argument("--mqtt-client-id", default="orin-yolo-presence", help="MQTT client id")
    p.add_argument("--mqtt-qos", type=int, default=0, choices=[0, 1, 2], help="MQTT QoS 0/1/2")
    p.add_argument("--mqtt-retain", action="store_true", help="Retain last presence message on broker")
    p.add_argument("--mqtt-json", action="store_true", help="Publish JSON instead of plain ON/OFF")

    p.add_argument("--window", default="YOLO detection results", help="OpenCV window title")
    p.add_argument("--no-window", action="store_true", help="Disable GUI window (headless mode)")
    p.add_argument("--save-frame", default="capture.png", help='Filename when pressing "p"')
    p.add_argument("--publish-every", type=float, default=0.0, help="Optional periodic publish in seconds (0 = only on state change)")

    args = p.parse_args()

    if args.enter_frames < 1:
        p.error("--enter-frames must be >= 1")
    if args.leave_frames < 1:
        p.error("--leave-frames must be >= 1")

    if args.resolution:
        try:
            w, h = args.resolution.lower().split("x")
            w, h = int(w), int(h)
            if w < 1 or h < 1:
                raise ValueError
        except Exception:
            p.error('--resolution must be in the form WxH, e.g. 640x480')

    return args


def is_url_source(src: str) -> bool:
    try:
        u = urlparse(src)
        return u.scheme.lower() in ("rtsp", "rtsps", "http", "https")
    except Exception:
        return False


def mqtt_connect(args):
    if args.mqtt_host is None:
        return None

    if mqtt is None:
        print("ERROR: paho-mqtt not installed. Install with:")
        print("python3 -m pip install paho-mqtt")
        sys.exit(1)

    try:
        # Works across paho versions without forcing callback API version syntax
        client = mqtt.Client(client_id=args.mqtt_client_id, protocol=mqtt.MQTTv311)
    except TypeError:
        client = mqtt.Client(client_id=args.mqtt_client_id)

    if args.mqtt_user is not None:
        client.username_pw_set(args.mqtt_user, args.mqtt_pass if args.mqtt_pass is not None else "")

    try:
        client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    except Exception as e:
        print(f"ERROR: MQTT connect failed to {args.mqtt_host}:{args.mqtt_port}: {e}")
        sys.exit(1)

    client.loop_start()
    return client


def mqtt_publish_presence(client, args, present, count=0, conf_max=0.0):
    if client is None:
        return

    if args.mqtt_json:
        payload = json.dumps(
            {
                "present": bool(present),
                "count": int(count),
                "conf_max": float(conf_max),
                "ts": time.time(),
            },
            separators=(",", ":"),
        )
    else:
        payload = "ON" if present else "OFF"

    try:
        info = client.publish(
            args.mqtt_topic,
            payload=payload,
            qos=args.mqtt_qos,
            retain=args.mqtt_retain,
        )
        # Non-blocking, but check immediate queueing result when available
        if hasattr(info, "rc") and info.rc != 0:
            print(f"WARNING: MQTT publish queue failed rc={info.rc}")
    except Exception as e:
        print(f"WARNING: MQTT publish failed: {e}")


def build_label_maps(labels):
    if isinstance(labels, dict):
        # ultralytics often returns {id: name}
        name_to_id = {str(v): int(k) for k, v in labels.items()}
        id_to_name = {int(k): str(v) for k, v in labels.items()}
    else:
        name_to_id = {str(v): int(i) for i, v in enumerate(labels)}
        id_to_name = {int(i): str(v) for i, v in enumerate(labels)}
    return name_to_id, id_to_name


def parse_source(img_source):
    img_ext_list = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    vid_ext_list = {".avi", ".mov", ".mp4", ".mkv", ".wmv", ".m4v", ".mpeg", ".mpg"}

    if os.path.isdir(img_source):
        return {"type": "folder"}

    if os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        ext = ext.lower()
        if ext in img_ext_list:
            return {"type": "image"}
        if ext in vid_ext_list:
            return {"type": "video"}
        print(f"ERROR: File extension {ext} is not supported.")
        sys.exit(1)

    if is_url_source(img_source):
        return {"type": "stream"}

    s = img_source.lower().strip()
    if s.startswith("usb"):
        try:
            return {"type": "usb", "index": int(s[3:])}
        except Exception:
            print('ERROR: USB source must look like "usb0", "usb1", etc.')
            sys.exit(1)

    if s.startswith("picamera"):
        try:
            return {"type": "picamera", "index": int(s[8:])}
        except Exception:
            print('ERROR: Picamera source must look like "picamera0".')
            sys.exit(1)

    print(f"ERROR: Input {img_source} is invalid. Please try again.")
    sys.exit(1)


def get_detections(results0):
    # Defensive wrapper around ultralytics result object
    if results0 is None:
        return []
    boxes = getattr(results0, "boxes", None)
    if boxes is None:
        return []
    return boxes


def main():
    args = parse_args()

    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    user_res = args.resolution
    record = args.record

    if not os.path.exists(model_path):
        print("ERROR: Model path is invalid or model was not found.")
        sys.exit(1)

    resize = False
    resW = None
    resH = None
    if user_res:
        resize = True
        resW, resH = [int(x) for x in user_res.lower().split("x")]

    if record and not user_res:
        print("ERROR: Please specify --resolution to record video at.")
        sys.exit(1)

    print("Loading model...")
    model = YOLO(model_path, task="detect")
    labels = model.names
    name_to_id, id_to_name = build_label_maps(labels)

    if args.presence_class not in name_to_id:
        print(f'ERROR: presence class "{args.presence_class}" not in model labels.')
        print("Available labels include (sample):", list(name_to_id.keys())[:50])
        sys.exit(1)

    presence_id = name_to_id[args.presence_class]
    source_info = parse_source(img_source)
    source_type = source_info["type"]

    cap = None
    recorder = None
    imgs_list = []

    if source_type == "image":
        imgs_list = [img_source]

    elif source_type == "folder":
        exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP", "*.webp", "*.WEBP")
        for pat in exts:
            imgs_list.extend(glob.glob(os.path.join(img_source, pat)))
        imgs_list = sorted(imgs_list)
        if not imgs_list:
            print("ERROR: No supported images found in folder.")
            sys.exit(1)

    elif source_type in ("video", "stream"):
        cap = cv2.VideoCapture(img_source)
        if not cap.isOpened():
            print(f"ERROR: Unable to open source: {img_source}")
            sys.exit(1)
        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    elif source_type == "usb":
        cap = cv2.VideoCapture(source_info["index"])
        if not cap.isOpened():
            print(f"ERROR: Unable to open USB camera index {source_info['index']}")
            sys.exit(1)
        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    elif source_type == "picamera":
        try:
            from picamera2 import Picamera2
        except Exception:
            print("ERROR: picamera2 is not installed. Install it or use another source.")
            sys.exit(1)

        if not user_res:
            print("ERROR: picamera source requires --resolution WxH")
            sys.exit(1)

        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)}))
        cap.start()

    if record:
        if source_type not in ("video", "usb", "picamera", "stream"):
            print("ERROR: Recording only works for video/camera/stream sources.")
            sys.exit(1)

        record_name = "demo1.avi"
        record_fps = 30.0
        recorder = cv2.VideoWriter(
            record_name,
            cv2.VideoWriter_fourcc(*"MJPG"),
            record_fps,
            (resW, resH),
        )
        if not recorder.isOpened():
            print(f"ERROR: Failed to open recorder output: {record_name}")
            sys.exit(1)

    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]

    avg_frame_rate = 0.0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    present = False
    enter_count = 0
    leave_count = 0
    last_publish_time = 0.0

    mqtt_client = mqtt_connect(args)

    # Initial publish (useful if retained)
    mqtt_publish_presence(mqtt_client, args, present, count=0, conf_max=0.0)
    last_publish_time = time.time()

    if not args.no_window:
        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    try:
        while True:
            t_start = time.perf_counter()

            if source_type in ("image", "folder"):
                if img_count >= len(imgs_list):
                    print("All images have been processed. Exiting program.")
                    break
                img_filename = imgs_list[img_count]
                frame = cv2.imread(img_filename)
                img_count += 1
                if frame is None:
                    print(f"WARNING: Failed to read image: {img_filename}")
                    continue

            elif source_type in ("video", "usb", "stream"):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Reached end of stream/video or failed to read frame. Exiting program.")
                    break

            elif source_type == "picamera":
                frame = cap.capture_array()
                if frame is None:
                    print("Unable to read frames from the Picamera. Exiting program.")
                    break

            else:
                print("ERROR: Unknown source type.")
                break

            if resize:
                frame = cv2.resize(frame, (resW, resH), interpolation=cv2.INTER_LINEAR)

            try:
                results = model(frame, verbose=False)
            except Exception as e:
                print(f"ERROR: Model inference failed: {e}")
                break

            if not results:
                detections = []
            else:
                detections = get_detections(results[0])

            object_count = 0
            target_count = 0
            target_conf_max = 0.0

            for i in range(len(detections)):
                try:
                    conf = float(detections[i].conf.item())
                    if conf < min_thresh:
                        continue

                    classidx = int(detections[i].cls.item())
                    classname = id_to_name.get(classidx, str(classidx))

                    xyxy = detections[i].xyxy[0].detach().cpu().numpy().astype(int)
                    xmin, ymin, xmax, ymax = [int(v) for v in xyxy]

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
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    object_count += 1

                    if classidx == presence_id:
                        target_count += 1
                        if conf > target_conf_max:
                            target_conf_max = conf

                except Exception:
                    # Skip malformed detection without crashing the whole loop
                    continue

            seen_target = target_count > 0

            prev_present = present
            if seen_target:
                enter_count += 1
                leave_count = 0
                if (not present) and (enter_count >= args.enter_frames):
                    present = True
            else:
                leave_count += 1
                enter_count = 0
                if present and (leave_count >= args.leave_frames):
                    present = False

            now = time.time()
            state_changed = (present != prev_present)
            periodic_due = (args.publish_every > 0 and (now - last_publish_time) >= args.publish_every)

            if state_changed or periodic_due:
                mqtt_publish_presence(
                    mqtt_client,
                    args,
                    present,
                    count=target_count,
                    conf_max=target_conf_max,
                )
                last_publish_time = now

            if source_type in ("video", "usb", "picamera", "stream"):
                cv2.putText(frame, f"FPS: {avg_frame_rate:0.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Objects: {object_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"{args.presence_class}: {target_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Present: {'ON' if present else 'OFF'}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if not args.no_window:
                cv2.imshow(args.window, frame)

            if recorder is not None:
                if user_res:
                    recorder.write(frame)
                else:
                    # Should not happen because --record requires --resolution, but keep safe
                    pass

            if source_type in ("image", "folder"):
                key = cv2.waitKey(0) if not args.no_window else ord("q")
            else:
                key = cv2.waitKey(1) if not args.no_window else -1

            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                if not args.no_window:
                    cv2.waitKey(0)
            elif key in (ord("p"), ord("P")):
                cv2.imwrite(args.save_frame, frame)
                print(f"Saved frame to {args.save_frame}")

            t_stop = time.perf_counter()
            dt = max(1e-9, (t_stop - t_start))
            frame_rate_calc = 1.0 / dt

            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = float(np.mean(frame_rate_buffer))

        print(f"Average pipeline FPS: {avg_frame_rate:.2f}")

    finally:
        try:
            if source_type in ("video", "usb", "stream") and cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            if source_type == "picamera" and cap is not None:
                cap.stop()
        except Exception:
            pass

        try:
            if recorder is not None:
                recorder.release()
        except Exception:
            pass

        try:
            if mqtt_client is not None:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
        except Exception:
            pass

        try:
            if not args.no_window:
                cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()