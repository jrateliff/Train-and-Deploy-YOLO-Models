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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
    )
    parser.add_argument(
        "--source",
        required=True,
        help='Image source: image file ("test.jpg"), image folder ("test_dir"), video file ("testvid.mp4"), "usb0", or "picamera0"',
    )
    parser.add_argument(
        "--thresh",
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
        default=10.0,
        help="Seconds with no presence-class before publishing OFF",
    )

    return parser.parse_args()


def mqtt_pub(host, user, pw, topic, msg):
    cmd = ["mosquitto_pub", "-h", host, "-p", "1883", "-u", user, "-P", pw, "-t", topic, "-m", msg, "-V", "5"]
    r = subprocess.run(cmd, text=True, capture_output=True)
    print(f"[MQTT] publish topic={topic} msg={msg} rc={r.returncode}", flush=True)
    if r.returncode != 0:
        if r.stderr:
            print("[MQTT] stderr:", r.stderr.strip(), flush=True)
        if r.stdout:
            print("[MQTT] stdout:", r.stdout.strip(), flush=True)


def main():
    args = parse_args()

    # Parse user inputs
    model_path = args.model
    img_source = args.source
    min_thresh = float(args.thresh)
    user_res = args.resolution
    record = args.record

    # Check model file exists
    if not os.path.exists(model_path):
        print("ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.")
        sys.exit(0)

    # Load YOLO model and labels
    model = YOLO(model_path, task="detect")
    labels = model.names

    # Determine source type
    img_ext_list = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]
    vid_ext_list = [".avi", ".mov", ".mp4", ".mkv", ".wmv"]

    if os.path.isdir(img_source):
        source_type = "folder"
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = "image"
        elif ext in vid_ext_list:
            source_type = "video"
        else:
            print(f"File extension {ext} is not supported.")
            sys.exit(0)
    elif "usb" in img_source:
        source_type = "usb"
        usb_idx = int(img_source[3:])
    elif "picamera" in img_source:
        source_type = "picamera"
    else:
        print(f"Input {img_source} is invalid. Please try again.")
        sys.exit(0)

    # Parse display resolution
    resize = False
    resW = None
    resH = None
    if user_res:
        resize = True
        resW, resH = int(user_res.split("x")[0]), int(user_res.split("x")[1])

    # Recording setup
    recorder = None
    if record:
        if source_type not in ["video", "usb"]:
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

    # Load/init capture source
    cap = None
    imgs_list = None

    if source_type == "image":
        imgs_list = [img_source]
    elif source_type == "folder":
        imgs_list = []
        filelist = glob.glob(img_source + "/*")
        for file in filelist:
            _, file_ext = os.path.splitext(file)
            if file_ext in img_ext_list:
                imgs_list.append(file)
    elif source_type in ["video", "usb"]:
        cap_arg = img_source if source_type == "video" else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if user_res:
            cap.set(3, resW)
            cap.set(4, resH)
    elif source_type == "picamera":
        if not user_res:
            print('Please specify --resolution WxH when using picamera0.')
            sys.exit(0)
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": "RGB888", "size": (resW, resH)}))
        cap.start()

    # Bounding box colors (Tableau 10)
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

    # Status variables
    avg_frame_rate = 0.0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0

    # Presence tracking
    presence_on = False
    last_present_time = 0.0
    presence_class = args.presence_class
    presence_topic = args.presence_topic

    # Set known startup state
    mqtt_pub(args.ha_ip, args.mqtt_user, args.mqtt_pass, presence_topic, "OFF")

    # Main loop
    while True:
        t_start = time.perf_counter()

        # Load frame
        if source_type in ["image", "folder"]:
            if img_count >= len(imgs_list):
                print("All images have been processed. Exiting program.")
                sys.exit(0)
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
        elif source_type == "video":
            ret, frame = cap.read()
            if not ret:
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
            conf = detections[i].conf.item()

            if conf > min_thresh:
                if classname == presence_class:
                    present_detected_this_frame = True
                    last_present_time = now

                color = bbox_colors[classidx % 10]
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

        # Publish presence transitions (no spam)
        if present_detected_this_frame and (not presence_on):
            print("[STATE] OFF->ON", flush=True)
            mqtt_pub(args.ha_ip, args.mqtt_user, args.mqtt_pass, presence_topic, "ON")
            presence_on = True

        if presence_on and (now - last_present_time) > args.presence_off_timeout:
            print("[STATE] ON->OFF", flush=True)
            mqtt_pub(args.ha_ip, args.mqtt_user, args.mqtt_pass, presence_topic, "OFF")
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

        # Display
        cv2.putText(
            frame,
            f"Number of objects: {object_count}",
            (10, 40),
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
        frame_rate_calc = float(1.0 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)

        avg_frame_rate = float(np.mean(frame_rate_buffer))

    # Clean up
    print(f"Average pipeline FPS: {avg_frame_rate:.2f}")

    if source_type in ["video", "usb"] and cap is not None:
        cap.release()
    elif source_type == "picamera" and cap is not None:
        cap.stop()

    if record and recorder is not None:
        recorder.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()