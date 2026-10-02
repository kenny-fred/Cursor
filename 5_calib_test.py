"""EyeMouse V2 - contrôle du regard compatible avec calibration_ray.json.

Le calibrateur V3 stocke des features d'iris et de pose, pas un rayon 3D.
Ce programme apprend donc directement : features -> position écran.
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk


MODEL_PATH = "models/face_landmarker.task"
# Le contrôleur privilégie le jeu nettoyé, mais reste utilisable juste après
# une nouvelle calibration qui produit le fichier standard.
CALIBRATION_PATHS = (
    "calibration_screen_v2.json",
    "calibration_screen_v2_cleaned.json",
)
CAMERA_INDEX = 0
# Filtre adaptatif : le point ne bouge pas pour le micro-bruit, mais rattrape
# vite une vraie intention de regard.
DEAD_ZONE_PX = 12
MIN_SMOOTHING = 0.22
MAX_SMOOTHING = 0.72
FAST_MOVE_DISTANCE_PX = 220

IRIS_LEFT = [468, 469, 470, 471, 472]
IRIS_RIGHT = [473, 474, 475, 476, 477]

LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}


def p3(face, index):
    landmark = face[index]
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def mean3(face, indices):
    return np.mean([p3(face, index) for index in indices], axis=0)


def normalize(vector):
    length = np.linalg.norm(vector)
    return np.zeros(3) if length < 1e-9 else vector / length


def extract_eye_features(face, eye, iris_indices):
    """Copie volontairement la géométrie de calib_ray.py."""
    outer, inner = p3(face, eye["outer"]), p3(face, eye["inner"])
    top, bottom = p3(face, eye["top"]), p3(face, eye["bottom"])
    iris = mean3(face, iris_indices)

    axis_x = normalize(inner - outer)
    axis_y_raw = bottom - top
    axis_y = normalize(axis_y_raw - np.dot(axis_y_raw, axis_x) * axis_x)
    axis_z = normalize(np.cross(axis_x, axis_y))
    center = (outer + inner + top + bottom) / 4.0

    width = max(np.linalg.norm(inner - outer), 1e-9)
    height = max(np.linalg.norm(bottom - top), 1e-9)
    delta = iris - center

    return {
        "iris_x": float(np.dot(delta, axis_x) / width),
        "iris_y": float(np.dot(delta, axis_y) / height),
        # Conservé uniquement pour vérifier la compatibilité avec calib_ray.
        "iris_z": float(np.dot(delta, axis_z) / width),
    }


def extract_head_pose(result):
    matrices = result.facial_transformation_matrixes
    if not matrices:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rotation = np.asarray(matrices[0], dtype=np.float64)[:3, :3]
    sy = np.hypot(rotation[0, 0], rotation[1, 0])
    if sy < 1e-6:
        pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(-rotation[2, 0], sy)
        roll = np.arctan2(rotation[1, 0], rotation[0, 0])
    return {
        "yaw": float(np.degrees(yaw)),
        "pitch": float(np.degrees(pitch)),
        "roll": float(np.degrees(roll)),
    }


def feature_vector(left, right, pose):
    """Les 7 variables réellement apprises, dans le même ordre partout."""
    return np.array([
        left["iris_x"], left["iris_y"],
        right["iris_x"], right["iris_y"],
        pose["yaw"], pose["pitch"], pose["roll"],
    ], dtype=np.float64)


def feature_vector_from_sample(sample):
    pose = sample["head_pose"]
    return np.array([
        sample["left_iris_x"], sample["left_iris_y"],
        sample["right_iris_x"], sample["right_iris_y"],
        pose["yaw"], pose["pitch"], pose["roll"],
    ], dtype=np.float64)


def polynomial_features(features):
    """Termes linéaires, carrés et interactions, sans constante."""
    rows, columns = features.shape
    terms = [features]
    interactions = [features[:, i] * features[:, j]
                    for i in range(columns) for j in range(i, columns)]
    return np.column_stack(terms + interactions).reshape(rows, -1)



class RidgeMapper:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, raw_features, targets):
        self.mean = raw_features.mean(axis=0)
        self.scale = raw_features.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        x = polynomial_features((raw_features - self.mean) / self.scale)
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        return self

    def predict(self, raw_features):
        x = polynomial_features((raw_features - self.mean) / self.scale)
        design = np.column_stack([np.ones(len(x)), x])
        return design @ self.coefficients


def adaptive_smooth(current, target):
    """Stabilise le point sans ajouter de retard sur les grands mouvements."""
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= DEAD_ZONE_PX:
        return current

    progress = min(
        (distance - DEAD_ZONE_PX) / FAST_MOVE_DISTANCE_PX,
        1.0,
    )
    alpha = MIN_SMOOTHING + progress * (MAX_SMOOTHING - MIN_SMOOTHING)
    return current + alpha * delta


def load_calibration(path):
    with Path(path).open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    if calibration.get("version") not in (3, 4):
        raise ValueError("Ce contrôleur attend un JSON de calibration V3 ou V4.")

    features, targets, groups = [], [], []
    for point_index, point in enumerate(calibration.get("points", [])):
        # V4 marque les captures qui n'ont jamais été stables, même après les
        # reprises. Elles ne doivent pas entraîner le curseur.
        if point.get("quality", {}).get("status") == "volatile":
            continue
        target = [point["screen"]["x"], point["screen"]["y"]]
        samples = point.get("samples", [])
        if not samples:
            continue
        for sample in samples:
            features.append(feature_vector_from_sample(sample))
            targets.append(target)
            groups.append(point_index)

    if len(set(groups)) < 6:
        raise ValueError("Calibration insuffisante : il faut au moins 6 points complets.")
    return calibration, np.asarray(features), np.asarray(targets), np.asarray(groups)


def find_calibration_path():
    for path in CALIBRATION_PATHS:
        if Path(path).is_file():
            return path
    expected = " ou ".join(CALIBRATION_PATHS)
    raise FileNotFoundError(f"Calibration introuvable : place {expected} à côté de ce script.")


def choose_alpha(features, targets, groups):
    """Validation leave-one-point-out : aucune mesure du point testé n'est apprise."""
    candidates = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha, best_error = None, np.inf
    for alpha in candidates:
        errors = []
        for group in np.unique(groups):
            train = groups != group
            test = ~train
            model = RidgeMapper(alpha).fit(features[train], targets[train])
            prediction = model.predict(features[test])
            errors.extend(np.linalg.norm(prediction - targets[test], axis=1))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_alpha, best_error = alpha, mean_error
    return best_alpha, best_error


def main():
    calibration_path = find_calibration_path()
    calibration, features, targets, groups = load_calibration(calibration_path)
    alpha, validation_error = choose_alpha(features, targets, groups)
    mapper = RidgeMapper(alpha).fit(features, targets)

    screen_w = calibration["screen"]["width"]
    screen_h = calibration["screen"]["height"]
    print(f"Calibration : {calibration_path}")
    print(f"EyeMouse — {len(features)} samples, {len(set(groups))} points")
    print(f"Validation (point jamais vu) : {validation_error:.1f} px")
    print(f"Régularisation choisie : {alpha:g}")
    print("ESC = quitter")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la caméra.")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    timestamp_ms, smooth = 0, None

    # Le calibrateur présentait les cibles dans une fenêtre Tk plein écran.
    # Le contrôle doit afficher la prédiction dans ce même repère (pixels de
    # l'écran), pas la réduire aux pixels de la prévisualisation caméra.
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    stop = {"value": False}

    def quit_control(event=None):
        stop["value"] = True

    root.bind("<Escape>", quit_control)
    root.focus_force()

    def draw_marker(position, text):
        canvas.delete("all")
        if position is not None:
            x, y = position
            radius = 14
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                               fill="#1687ff", outline="white", width=2)
            canvas.create_line(x - 25, y, x + 25, y, fill="white", width=1)
            canvas.create_line(x, y - 25, x, y + 25, fill="white", width=1)
        canvas.create_text(16, 16, anchor="nw", fill="white",
                           font=("Arial", 14), text=text)
        canvas.create_text(16, screen_h - 16, anchor="sw", fill="#b0b0b0",
                           font=("Arial", 12), text="ESC = quitter")

    try:
        while not stop["value"]:
            ok, frame = cap.read()
            if not ok:
                draw_marker(None, "CAMERA ERROR")
                root.update()
                continue
            frame = cv2.flip(frame, 1)  # identique à calib_ray
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.face_landmarks:
                face = result.face_landmarks[0]
                left = extract_eye_features(face, LEFT_EYE, IRIS_LEFT)
                right = extract_eye_features(face, RIGHT_EYE, IRIS_RIGHT)
                pose = extract_head_pose(result)
                raw = feature_vector(left, right, pose)[None, :]
                predicted = mapper.predict(raw)[0]
                predicted[0] = np.clip(predicted[0], 0, screen_w - 1)
                predicted[1] = np.clip(predicted[1], 0, screen_h - 1)

                smooth = predicted if smooth is None else adaptive_smooth(smooth, predicted)
                draw_marker((int(smooth[0]), int(smooth[1])),
                            f"EyeMouse  {smooth[0]:.0f}, {smooth[1]:.0f}")
            else:
                draw_marker(None, "EyeMouse  NO FACE")

            root.update()
    finally:
        cap.release()
        landmarker.close()
        root.destroy()


if __name__ == "__main__":
    main()
