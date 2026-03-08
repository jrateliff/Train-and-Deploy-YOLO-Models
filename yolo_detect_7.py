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


def find_candidate_models():
    return sorted(set(glob.glob("*.engine") + glob.glob("*.pt")))


def choose_model_path(default_model=None):
    model_candidates = find_candidate_models()
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

    return prompt_text_required("Custom model path", default_model or "yolo26s.engine")


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

    default_idx = source_options.index(default_source) if default_source in source_options else 0
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
    elif str(default_resolution) == "640x480":
        default_idx = 0
    elif str(default_resolution) == "1280x720":
        default_idx = 1
    elif str(default_resolution) == "1920x1080":
        default_idx = 2

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


def interactive_terminal_wizard(args):
    print("")
    print("YOLO Detect Setup Wizard (Terminal)")
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


def interactive_gui_wizard(args):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        print(f"GUI unavailable (tkinter import failed): {e}")
        return False

    model_candidates = find_candidate_models()
    default_model = args.model or (model_candidates[0] if model_candidates else "yolo26s.engine")

    # Defaults
    source_default = args.source or "usb0"
    resolution_default = args.resolution if args.resolution is not None else "640x480"
    thresh_default = float(args.thresh if args.thresh is not None else 0.5)
    timeout_default = float(args.presence_off_timeout if args.presence_off_timeout is not None else 1.0)

    result = {"start": False}

    root = tk.Tk()
    root.title("YOLO Detect 5 Setup GUI")
    root.geometry("980x900")
    root.minsize(900, 700)

    # Scrollable container
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    outer = tk.Frame(canvas)

    outer.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=outer, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    title = tk.Label(
        outer,
        text="YOLO Detect Setup GUI\nDefaults are preselected. Review and click Start.",
        justify="left",
        font=("TkDefaultFont", 11, "bold")
    )
    title.pack(anchor="w", padx=10, pady=(10, 6))

    # Model section
    model_frame = tk.LabelFrame(outer, text="Model (select one)")
    model_frame.pack(fill="x", padx=10, pady=6)

    model_choice_var = tk.StringVar(value="preset")
    model_value_var = tk.StringVar(value=default_model)
    model_custom_var = tk.StringVar(value=default_model if default_model not in model_candidates else "")

    if model_candidates:
        for m in model_candidates:
            rb = tk.Radiobutton(
                model_frame,
                text=m,
                variable=model_value_var,
                value=m,
                command=lambda: model_choice_var.set("preset"),
                anchor="w"
            )
            rb.pack(anchor="w", padx=8, pady=2)
    else:
        tk.Label(model_frame, text="No .pt or .engine files found in current folder. Use custom path below.").pack(anchor="w", padx=8, pady=2)

    custom_model_row = tk.Frame(model_frame)
    custom_model_row.pack(fill="x", padx=8, pady=4)

    def select_model_custom():
        model_choice_var.set("custom")

    model_custom_rb = tk.Radiobutton(custom_model_row, text="Custom model path", variable=model_choice_var, value="custom")
    model_custom_rb.pack(side="left")

    model_custom_entry = tk.Entry(custom_model_row, textvariable=model_custom_var, width=70)
    model_custom_entry.pack(side="left", padx=6, fill="x", expand=True)
    model_custom_entry.bind("<FocusIn>", lambda e: select_model_custom())

    def browse_model():
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("YOLO models", "*.pt *.engine"), ("All files", "*.*")]
        )
        if path:
            model_custom_var.set(path)
            model_choice_var.set("custom")

    tk.Button(custom_model_row, text="Browse", command=browse_model).pack(side="left", padx=4)

    # If default model is one of detected files, preselect preset
    if default_model in model_candidates:
        model_choice_var.set("preset")
        model_value_var.set(default_model)
    else:
        model_choice_var.set("custom")
        model_custom_var.set(default_model)

    # Source section
    source_frame = tk.LabelFrame(outer, text="Source (select one)")
    source_frame.pack(fill="x", padx=10, pady=6)

    source_mode_var = tk.StringVar()
    source_text_var = tk.StringVar()

    source_modes = [
        ("usb0", "USB Camera 0 (usb0)"),
        ("usb1", "USB Camera 1 (usb1)"),
        ("picamera0", "Pi Camera (picamera0)"),
        ("video_file", "Video file"),
        ("image_file", "Image file"),
        ("image_folder", "Image folder"),
        ("custom", "Custom source string"),
    ]

    source_mode_defaults = {"usb0", "usb1", "picamera0"}
    if source_default in source_mode_defaults:
        source_mode_var.set(source_default)
        source_text_var.set(source_default)
    else:
        # Infer likely mode from path
        if os.path.isdir(source_default):
            source_mode_var.set("image_folder")
            source_text_var.set(source_default)
        elif os.path.isfile(source_default):
            _, ext = os.path.splitext(source_default)
            ext = ext.lower()
            if ext in [".avi", ".mov", ".mp4", ".mkv", ".wmv"]:
                source_mode_var.set("video_file")
            else:
                source_mode_var.set("image_file")
            source_text_var.set(source_default)
        else:
            source_mode_var.set("custom")
            source_text_var.set(source_default)

    for mode_key, mode_label in source_modes:
        tk.Radiobutton(
            source_frame,
            text=mode_label,
            variable=source_mode_var,
            value=mode_key,
            anchor="w"
        ).pack(anchor="w", padx=8, pady=2)

    source_path_row = tk.Frame(source_frame)
    source_path_row.pack(fill="x", padx=8, pady=4)

    tk.Label(source_path_row, text="Path / custom source").pack(side="left")
    source_path_entry = tk.Entry(source_path_row, textvariable=source_text_var, width=70)
    source_path_entry.pack(side="left", padx=6, fill="x", expand=True)

    def browse_source_file():
        path = filedialog.askopenfilename(title="Select video or image file")
        if path:
            source_text_var.set(path)
            _, ext = os.path.splitext(path)
            ext = ext.lower()
            if ext in [".avi", ".mov", ".mp4", ".mkv", ".wmv"]:
                source_mode_var.set("video_file")
            else:
                source_mode_var.set("image_file")

    def browse_source_folder():
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            source_text_var.set(path)
            source_mode_var.set("image_folder")

    tk.Button(source_path_row, text="Browse File", command=browse_source_file).pack(side="left", padx=4)
    tk.Button(source_path_row, text="Browse Folder", command=browse_source_folder).pack(side="left", padx=4)

    # Resolution section
    res_frame = tk.LabelFrame(outer, text="Resolution (select one)")
    res_frame.pack(fill="x", padx=10, pady=6)

    resolution_mode_var = tk.StringVar()
    resolution_custom_var = tk.StringVar(value=str(resolution_default if resolution_default is not None else "640x480"))

    if resolution_default is None:
        resolution_mode_var.set("none")
    elif str(resolution_default) in ["640x480", "1280x720", "1920x1080"]:
        resolution_mode_var.set(str(resolution_default))
    else:
        resolution_mode_var.set("custom")

    for val, label in [
        ("640x480", "640x480"),
        ("1280x720", "1280x720"),
        ("1920x1080", "1920x1080"),
        ("none", "Use source resolution (none)"),
        ("custom", "Custom WxH"),
    ]:
        tk.Radiobutton(res_frame, text=label, variable=resolution_mode_var, value=val, anchor="w").pack(anchor="w", padx=8, pady=2)

    res_custom_row = tk.Frame(res_frame)
    res_custom_row.pack(fill="x", padx=8, pady=4)
    tk.Label(res_custom_row, text="Custom WxH").pack(side="left")
    res_custom_entry = tk.Entry(res_custom_row, textvariable=resolution_custom_var, width=20)
    res_custom_entry.pack(side="left", padx=6)
    res_custom_entry.bind("<FocusIn>", lambda e: resolution_mode_var.set("custom"))

    # Threshold section
    thresh_frame = tk.LabelFrame(outer, text="Confidence threshold (select one)")
    thresh_frame.pack(fill="x", padx=10, pady=6)

    thresh_mode_var = tk.StringVar()
    thresh_custom_var = tk.StringVar(value=f"{thresh_default:.2f}")

    preset_thresh = [0.25, 0.40, 0.50, 0.60, 0.75]
    if thresh_default in preset_thresh:
        thresh_mode_var.set(f"{thresh_default:.2f}")
    else:
        thresh_mode_var.set("custom")

    for v in preset_thresh:
        s = f"{v:.2f}"
        tk.Radiobutton(thresh_frame, text=s, variable=thresh_mode_var, value=s, anchor="w").pack(anchor="w", padx=8, pady=2)

    thresh_custom_row = tk.Frame(thresh_frame)
    thresh_custom_row.pack(fill="x", padx=8, pady=4)
    tk.Radiobutton(thresh_custom_row, text="Custom threshold", variable=thresh_mode_var, value="custom").pack(side="left")
    thresh_custom_entry = tk.Entry(thresh_custom_row, textvariable=thresh_custom_var, width=12)
    thresh_custom_entry.pack(side="left", padx=6)
    thresh_custom_entry.bind("<FocusIn>", lambda e: thresh_mode_var.set("custom"))

    # Presence and MQTT section
    mqtt_frame = tk.LabelFrame(outer, text="MQTT / Home Assistant")
    mqtt_frame.pack(fill="x", padx=10, pady=6)

    ha_ip_var = tk.StringVar(value=args.ha_ip or "192.168.1.52")
    mqtt_user_var = tk.StringVar(value=args.mqtt_user or "jtrdev")
    mqtt_pass_var = tk.StringVar(value=args.mqtt_pass or "1010Maxisthebest9911#")
    presence_topic_var = tk.StringVar(value=args.presence_topic or "home/orin/yolo/person_present")
    presence_class_var = tk.StringVar(value=args.presence_class or "person")

    def add_labeled_entry(parent, label_text, tk_var, show=None):
        row = tk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        tk.Label(row, text=label_text, width=20, anchor="w").pack(side="left")
        entry = tk.Entry(row, textvariable=tk_var, show=show)
        entry.pack(side="left", fill="x", expand=True)
        return entry

    add_labeled_entry(mqtt_frame, "MQTT broker IP", ha_ip_var)
    add_labeled_entry(mqtt_frame, "MQTT username", mqtt_user_var)
    add_labeled_entry(mqtt_frame, "MQTT password", mqtt_pass_var, show="*")
    add_labeled_entry(mqtt_frame, "Presence topic", presence_topic_var)
    add_labeled_entry(mqtt_frame, "Presence class", presence_class_var)

    timeout_frame = tk.LabelFrame(outer, text="Presence OFF timeout (seconds) (select one)")
    timeout_frame.pack(fill="x", padx=10, pady=6)

    timeout_mode_var = tk.StringVar()
    timeout_custom_var = tk.StringVar(value=f"{timeout_default:.1f}")
    preset_timeout = [0.5, 1.0, 2.0, 5.0, 10.0]

    if timeout_default in preset_timeout:
        timeout_mode_var.set(f"{timeout_default:.1f}")
    else:
        timeout_mode_var.set("custom")

    for v in preset_timeout:
        s = f"{v:.1f}"
        tk.Radiobutton(timeout_frame, text=s, variable=timeout_mode_var, value=s, anchor="w").pack(anchor="w", padx=8, pady=2)

    timeout_custom_row = tk.Frame(timeout_frame)
    timeout_custom_row.pack(fill="x", padx=8, pady=4)
    tk.Radiobutton(timeout_custom_row, text="Custom timeout", variable=timeout_mode_var, value="custom").pack(side="left")
    timeout_custom_entry = tk.Entry(timeout_custom_row, textvariable=timeout_custom_var, width=12)
    timeout_custom_entry.pack(side="left", padx=6)
    timeout_custom_entry.bind("<FocusIn>", lambda e: timeout_mode_var.set("custom"))

    # Flags section (multiple checkboxes can be selected)
    flags_frame = tk.LabelFrame(outer, text="Flags (multiple selections allowed)")
    flags_frame.pack(fill="x", padx=10, pady=6)

    record_var = tk.BooleanVar(value=bool(args.record))
    retain_var = tk.BooleanVar(value=bool(args.mqtt_retain))
    debug_var = tk.BooleanVar(value=True if args.presence_debug else False)

    tk.Checkbutton(flags_frame, text="Record output video", variable=record_var).pack(anchor="w", padx=8, pady=2)
    tk.Checkbutton(flags_frame, text="Publish retained MQTT messages", variable=retain_var).pack(anchor="w", padx=8, pady=2)
    tk.Checkbutton(flags_frame, text="Enable presence debug prints", variable=debug_var).pack(anchor="w", padx=8, pady=2)

    summary_var = tk.StringVar(value="Ready. Review selections and click Start.")

    summary_frame = tk.LabelFrame(outer, text="Status")
    summary_frame.pack(fill="x", padx=10, pady=6)
    tk.Label(summary_frame, textvariable=summary_var, justify="left", anchor="w").pack(fill="x", padx=8, pady=6)

    button_row = tk.Frame(outer)
    button_row.pack(fill="x", padx=10, pady=(8, 14))

    def collect_gui_values():
        # Model
        if model_choice_var.get() == "custom":
            model_path = model_custom_var.get().strip()
        else:
            model_path = model_value_var.get().strip()

        if not model_path:
            raise ValueError("Model path is required.")

        # Source
        src_mode = source_mode_var.get().strip()
        if src_mode in ("usb0", "usb1", "picamera0"):
            source_value = src_mode
        elif src_mode == "video_file":
            source_value = source_text_var.get().strip()
            if not source_value:
                raise ValueError("Video file path is required.")
        elif src_mode == "image_file":
            source_value = source_text_var.get().strip()
            if not source_value:
                raise ValueError("Image file path is required.")
        elif src_mode == "image_folder":
            source_value = source_text_var.get().strip()
            if not source_value:
                raise ValueError("Image folder path is required.")
        elif src_mode == "custom":
            source_value = source_text_var.get().strip()
            if not source_value:
                raise ValueError("Custom source string is required.")
        else:
            raise ValueError("Select a source.")

        # Resolution
        rmode = resolution_mode_var.get().strip()
        if rmode == "none":
            resolution_value = None
        elif rmode in ("640x480", "1280x720", "1920x1080"):
            resolution_value = rmode
        elif rmode == "custom":
            resolution_value = resolution_custom_var.get().strip()
            if not resolution_value:
                raise ValueError("Custom resolution is required.")
        else:
            raise ValueError("Select a resolution.")

        # Threshold
        tmode = thresh_mode_var.get().strip()
        if tmode == "custom":
            thresh_value = float(thresh_custom_var.get().strip())
        else:
            thresh_value = float(tmode)

        # Timeout
        tomode = timeout_mode_var.get().strip()
        if tomode == "custom":
            timeout_value = float(timeout_custom_var.get().strip())
        else:
            timeout_value = float(tomode)

        # MQTT text fields
        ha_ip = ha_ip_var.get().strip()
        mqtt_user = mqtt_user_var.get().strip()
        mqtt_pass = mqtt_pass_var.get().strip()
        presence_topic = presence_topic_var.get().strip()
        presence_class = presence_class_var.get().strip()

        if not ha_ip:
            raise ValueError("MQTT broker IP is required.")
        if not mqtt_user:
            raise ValueError("MQTT username is required.")
        if not mqtt_pass:
            raise ValueError("MQTT password is required.")
        if not presence_topic:
            raise ValueError("Presence topic is required.")
        if not presence_class:
            raise ValueError("Presence class is required.")

        return {
            "model": model_path,
            "source": source_value,
            "resolution": resolution_value,
            "thresh": thresh_value,
            "record": bool(record_var.get()),
            "ha_ip": ha_ip,
            "mqtt_user": mqtt_user,
            "mqtt_pass": mqtt_pass,
            "presence_topic": presence_topic,
            "presence_class": presence_class,
            "presence_off_timeout": timeout_value,
            "mqtt_retain": bool(retain_var.get()),
            "presence_debug": bool(debug_var.get()),
        }

    def on_start():
        try:
            values = collect_gui_values()
        except ValueError as e:
            summary_var.set(f"Error: {e}")
            messagebox.showerror("Invalid selection", str(e))
            return
        except Exception as e:
            summary_var.set(f"Error: {e}")
            messagebox.showerror("Invalid input", f"{e}")
            return

        args.model = values["model"]
        args.source = values["source"]
        args.resolution = values["resolution"]
        args.thresh = values["thresh"]
        args.record = values["record"]
        args.ha_ip = values["ha_ip"]
        args.mqtt_user = values["mqtt_user"]
        args.mqtt_pass = values["mqtt_pass"]
        args.presence_topic = values["presence_topic"]
        args.presence_class = values["presence_class"]
        args.presence_off_timeout = values["presence_off_timeout"]
        args.mqtt_retain = values["mqtt_retain"]
        args.presence_debug = values["presence_debug"]

        result["start"] = True
        root.destroy()

    def on_cancel():
        result["start"] = False
        root.destroy()

    tk.Button(button_row, text="Start", command=on_start, width=16, height=2).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Cancel", command=on_cancel, width=12, height=2).pack(side="left")

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        try:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    try:
        root.mainloop()
    finally:
        try:
            canvas.bind_all("<MouseWheel>", lambda e: None)
        except Exception:
            pass

    return result["start"]


def choose_setup_mode_and_run_wizard(args, prefer_gui=True):
    print("")
    print("Setup mode")
    print("Choose how you want to configure the program before it starts.")
    print("")

    options = [
        "GUI setup (recommended)",
        "Terminal setup wizard",
    ]
    default_idx = 0 if prefer_gui else 1
    mode = menu_select("Select setup mode", options, default_index=default_idx)

    if mode.startswith("GUI"):
        ok = interactive_gui_wizard(args)
        if not ok:
            print("GUI canceled or unavailable.")
            use_terminal = menu_yes_no("Use terminal setup wizard instead", default=True)
            if use_terminal:
                interactive_terminal_wizard(args)
                return True
            return False
        return True

    interactive_terminal_wizard(args)
    return True


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
        help="Launch setup mode selector (GUI or terminal). Also runs automatically when no arguments are provided.",
    )
    parser.add_argument(
        "--wizard-gui",
        action="store_true",
        help="Launch GUI setup directly.",
    )
    parser.add_argument(
        "--wizard-terminal",
        action="store_true",
        help="Launch terminal setup wizard directly.",
    )

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
        ok = choose_setup_mode_and_run_wizard(args, prefer_gui=True)
        if not ok:
            print("Canceled.")
            return
    elif args.wizard_gui:
        ok = interactive_gui_wizard(args)
        if not ok:
            print("Canceled.")
            return
    elif args.wizard_terminal:
        interactive_terminal_wizard(args)

    if not args.model:
        parser.error("--model is required (or run with no args / --wizard / --wizard-gui / --wizard-terminal)")
    if not args.source:
        parser.error("--source is required (or run with no args / --wizard / --wizard-gui / --wizard-terminal)")

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

    presence_on = False
    last_present_time = 0.0
    presence_class = args.presence_class.strip().lower()
    presence_topic = args.presence_topic

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

            if resize:
                frame = cv2.resize(frame, (resW, resH))

            results = model(frame, verbose=False)
            detections = results[0].boxes

            object_count = 0
            now = time.time()
            present_detected_this_frame = False

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

            cv2.putText(
                frame,
                f"Number of objects: {object_count}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            state_text = "Presence: ON" if presence_on else "Presence: OFF"
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