"""EyeMouse - calibration plein écran avec contrôle de stabilité par cible."""
import json
import time
import tkinter as tk

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "models/face_landmarker.task"
OUTPUT_PATH = "calibration_screen_v2.json"
CAMERA_INDEX = 0

# 6 x 4 correspond à l'aspect d'un écran 16:9 : les cibles sont espacées
# presque pareil horizontalement et verticalement.
GRID_COLS = 6
GRID_ROWS = 4
SAMPLES_PER_POINT = 15
WAIT_BEFORE_CAPTURE = 1.0
MAX_ATTEMPTS_PER_POINT = 3

# Écart-type maximal admis après retrait des outliers, dans l'ordre des
# features retournées par feature_vector(). Les valeurs viennent du bruit
# observé dans le jeu 5x18 fourni, avec une marge raisonnable.
STABILITY_LIMITS = np.array([0.010, 0.018, 0.012, 0.020, 0.50, 0.55, 0.25])

IRIS_LEFT = [468, 469, 470, 471, 472]
IRIS_RIGHT = [473, 474, 475, 476, 477]
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}


def p3(face, index):
    landmark = face[index]
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def normalize(vector):
    length = np.linalg.norm(vector)
    return np.zeros(3) if length < 1e-9 else vector / length


def eye_features(face, eye, iris_indices):
    outer, inner = p3(face, eye["outer"]), p3(face, eye["inner"])
    top, bottom = p3(face, eye["top"]), p3(face, eye["bottom"])
    iris = np.mean([p3(face, i) for i in iris_indices], axis=0)
    axis_x = normalize(inner - outer)
    vertical = bottom - top
    axis_y = normalize(vertical - np.dot(vertical, axis_x) * axis_x)
    axis_z = normalize(np.cross(axis_x, axis_y))
    center = (outer + inner + top + bottom) / 4.0
    delta = iris - center
    width, height = max(np.linalg.norm(inner - outer), 1e-9), max(np.linalg.norm(bottom - top), 1e-9)
    return {
        "iris_x": float(np.dot(delta, axis_x) / width),
        "iris_y": float(np.dot(delta, axis_y) / height),
        "iris_z": float(np.dot(delta, axis_z) / width),
        "center": center, "iris": iris, "width": float(width), "height": float(height),
    }


def head_pose(result):
    matrices = result.facial_transformation_matrixes
    if not matrices:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    rotation = np.asarray(matrices[0], dtype=np.float64)[:3, :3]
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy < 1e-6:
        pitch, yaw, roll = np.arctan2(-rotation[1, 2], rotation[1, 1]), np.arctan2(-rotation[2, 0], sy), 0.0
    else:
        pitch, yaw, roll = np.arctan2(rotation[2, 1], rotation[2, 2]), np.arctan2(-rotation[2, 0], sy), np.arctan2(rotation[1, 0], rotation[0, 0])
    return {"yaw": float(np.degrees(yaw)), "pitch": float(np.degrees(pitch)), "roll": float(np.degrees(roll))}


def extract_sample(face, result):
    left, right, pose = eye_features(face, LEFT_EYE, IRIS_LEFT), eye_features(face, RIGHT_EYE, IRIS_RIGHT), head_pose(result)
    face_width = np.linalg.norm(p3(face, 454) - p3(face, 234))
    face_height = np.linalg.norm(p3(face, 152) - p3(face, 10))
    return {
        "left_eye_center_3d": left["center"].tolist(), "right_eye_center_3d": right["center"].tolist(),
        "eye_center_3d": ((left["center"] + right["center"]) / 2).tolist(),
        "left_iris_x": left["iris_x"], "left_iris_y": left["iris_y"], "left_iris_z": left["iris_z"],
        "right_iris_x": right["iris_x"], "right_iris_y": right["iris_y"], "right_iris_z": right["iris_z"],
        "left_eye_width": left["width"], "left_eye_height": left["height"],
        "right_eye_width": right["width"], "right_eye_height": right["height"],
        "iris_distance_3d": float(np.linalg.norm(right["iris"] - left["iris"])),
        "face_width_3d": float(face_width), "face_height_3d": float(face_height), "head_pose": pose,
    }


def feature_vector(sample):
    pose = sample["head_pose"]
    return np.array([sample["left_iris_x"], sample["left_iris_y"], sample["right_iris_x"], sample["right_iris_y"], pose["yaw"], pose["pitch"], pose["roll"]])


def assess_samples(samples):
    """Retire les outliers locaux et mesure si le regard est resté stable."""
    values = np.asarray([feature_vector(sample) for sample in samples])
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust_z = np.max(np.abs(values - median) / (1.4826 * mad + 1e-6), axis=1)
    kept = [sample for sample, score in zip(samples, robust_z) if score <= 4.0]
    kept_values = np.asarray([feature_vector(sample) for sample in kept])
    deviations = np.std(kept_values, axis=0) if len(kept_values) else np.full(7, np.inf)
    stable = len(kept) >= 12 and bool(np.all(deviations <= STABILITY_LIMITS))
    return kept, {
        "status": "stable" if stable else "volatile",
        "raw_samples": len(samples), "accepted_samples": len(kept),
        "removed_outliers": len(samples) - len(kept),
        "feature_std": deviations.tolist(), "stability_limits": STABILITY_LIMITS.tolist(),
    }


def draw_target(canvas, screen_w, screen_h, x, y, label, color):
    canvas.delete("all")
    canvas.create_text(screen_w // 2, 35, text=label, fill="white", font=("Arial", 20))
    canvas.create_oval(x - 14, y - 14, x + 14, y + 14, fill=color, outline="white", width=2)
    canvas.create_line(x - 25, y, x + 25, y, fill="white")
    canvas.create_line(x, y - 25, x, y + 25, fill="white")


def main():
    root = tk.Tk()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.attributes("-fullscreen", True); root.attributes("-topmost", True); root.configure(bg="black")
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    cancelled = {"value": False}
    root.bind("<Escape>", lambda event: cancelled.__setitem__("value", True))
    root.focus_force()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        root.destroy(); raise SystemExit("Impossible d'ouvrir la caméra.")
    camera_w, camera_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH), running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_facial_transformation_matrixes=True)
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    points = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = int((0.10 + col * 0.80 / (GRID_COLS - 1)) * screen_w)
            y = int((0.10 + row * 0.80 / (GRID_ROWS - 1)) * screen_h)
            points.append((x, y))

    calibration_points, timestamp_ms = [], 0
    try:
        for point_index, (x, y) in enumerate(points):
            accepted, quality = [], None
            for attempt in range(1, MAX_ATTEMPTS_PER_POINT + 1):
                if cancelled["value"]: break
                draw_target(canvas, screen_w, screen_h, x, y,
                            f"Calibration {point_index + 1}/{len(points)} — stabilise le regard ({attempt}/{MAX_ATTEMPTS_PER_POINT})", "red")
                root.update(); start, raw = time.time(), []
                while not cancelled["value"] and len(raw) < SAMPLES_PER_POINT:
                    root.update(); ok, frame = cap.read()
                    if not ok: continue
                    frame = cv2.flip(frame, 1)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    timestamp_ms += 33; result = landmarker.detect_for_video(image, timestamp_ms)
                    if time.time() - start >= WAIT_BEFORE_CAPTURE and result.face_landmarks:
                        raw.append(extract_sample(result.face_landmarks[0], result))
                if cancelled["value"]: break
                accepted, quality = assess_samples(raw)
                if quality["status"] == "stable": break
                draw_target(canvas, screen_w, screen_h, x, y, "Capture instable : recommence sans bouger la tête", "#ffb000")
                root.update(); time.sleep(1.0)
            if cancelled["value"]: break
            calibration_points.append({"screen": {"x": x, "y": y}, "samples": accepted, "quality": quality})
            print(f"Point {point_index + 1}: {quality['status']}, {quality['accepted_samples']} samples retenus")
    finally:
        cap.release(); landmarker.close(); root.destroy()

    if cancelled["value"]:
        print("Calibration annulée."); return
    stable_points = sum(point["quality"]["status"] == "stable" for point in calibration_points)
    calibration = {
        "version": 4, "method": "iris_head_pose_mapping_with_stability_gate",
        "screen": {"width": screen_w, "height": screen_h}, "camera": {"width": camera_w, "height": camera_h},
        "grid": {"columns": GRID_COLS, "rows": GRID_ROWS, "points": len(points), "samples_per_point": SAMPLES_PER_POINT},
        "quality_summary": {"stable_points": stable_points, "volatile_points": len(points) - stable_points},
        "points": calibration_points,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file: json.dump(calibration, file, indent=2)
    print(f"Calibration écrite : {OUTPUT_PATH} ({stable_points}/{len(points)} points stables)")


if __name__ == "__main__":
    main()
