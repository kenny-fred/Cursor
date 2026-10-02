import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk


def get_resource_path(relative_path):
    """Calcule le chemin absolu vers la ressource, compatible PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


MODEL_PATH = get_resource_path(os.path.join("models", "face_landmarker.task"))
APP_ICON_PATH = get_resource_path("icon.png")
CAMERA_INDEX = 0

# Configuration de la Grille (5% x 5%)
GRID_PERCENT_X = 0.05
GRID_PERCENT_Y = 0.05

# Filtre de stabilisation
BASE_SMOOTHING_ALPHA = 0.22
STABILIZATION_TIME_MS = 200


def get_system_font():
    """Détection de la meilleure police sans-serif native."""
    system = os.name
    if system == "nt":  # Windows
        return "Segoe UI"
    if system == "posix":  # macOS / Linux
        if platform.system() == "Darwin":  # macOS
            return "Helvetica Neue"
        return "Ubuntu"  # Linux
    return "Arial"  # Fallback universel


FONT_FAMILY = get_system_font()

# Palette Dark Mode moderne
COLOR_BG = "#0F172A"  # Slate très sombre / Fond principal (Bleu foncé exact)
COLOR_TEXT_MAIN = "#F8FAFC"  # Texte blanc cassé principal
COLOR_TEXT_MUTED = "#94A3B8"  # Texte secondaire gris clair

# Style du bouton translucide / contrasté sur fond sombre
COLOR_BTN_BG = "#1E293B"  # Fond bouton sombre
COLOR_BTN_HOVER = "#334155"  # Légèrement plus clair au survol
COLOR_BTN_BORDER = "#475569"  # Bordure subtile
COLOR_BTN_TEXT = "#38BDF8"  # Texte bleu néon/cyan

# Couleurs Overlay Suivi (Case active = bleu, Case candidate = orange)
COLOR_GRID_LINE = "#334155"
COLOR_CELL_TARGET = (
    "#EA580C"  # Orange (case où le regard se déplace / temporisation)
)
COLOR_CELL_FILL = "#2563EB"  # Bleu (case active stabilisée)
COLOR_POINT = "#FFFFFF"

IRIS_LEFT = [468, 469, 470, 471, 472]
IRIS_RIGHT = [473, 474, 475, 476, 477]

LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}


def apply_dark_title_bar(window):
    """Force Windows à afficher la barre de titre en mode sombre."""
    if os.name != "nt":
        return
    try:
        window.update()
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        get_parent = ctypes.windll.user32.GetParent
        hwnd = get_parent(window.winfo_id())
        rendering_policy = ctypes.c_int(2)  # 2 active le mode sombre
        set_window_attribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(rendering_policy),
            ctypes.sizeof(rendering_policy),
        )

        # Astuce complémentaire pour forcer Windows à repeindre le fond de la barre de titre avec le code couleur exact (#0F172A sous format BGR: 0x2A170F)
        DWMWA_CAPTION_COLOR = 35
        # #0F172A en BGR hexadécimal -> 0x2A170F
        caption_color = ctypes.c_int(0x2A170F)
        set_window_attribute(
            hwnd,
            DWMWA_CAPTION_COLOR,
            ctypes.byref(caption_color),
            ctypes.sizeof(caption_color),
        )
    except Exception as e:
        print(f"Impossible d'appliquer le thème sombre à la barre de titre : {e}")


def p3(face, index):
    landmark = face[index]
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def mean3(face, indices):
    return np.mean([p3(face, index) for index in indices], axis=0)


def normalize(vector):
    length = np.linalg.norm(vector)
    return np.zeros(3) if length < 1e-9 else vector / length


def extract_face_distance(face):
    left_corner = p3(face, LEFT_EYE["outer"])
    right_corner = p3(face, RIGHT_EYE["outer"])
    return float(np.linalg.norm(left_corner - right_corner))


def extract_eye_features(face, eye, iris_indices):
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


def feature_vector(left, right, pose, face_dist):
    return np.array(
        [
            left["iris_x"],
            left["iris_y"],
            right["iris_x"],
            right["iris_y"],
            pose["yaw"],
            pose["pitch"],
            pose["roll"],
            face_dist,
        ],
        dtype=np.float64,
    )


def feature_vector_from_sample(sample, face_dist_fallback=0.3):
    pose = sample["head_pose"]
    face_dist = sample.get(
        "face_distance", sample.get("face_size", face_dist_fallback)
    )
    return np.array(
        [
            sample["left_iris_x"],
            sample["left_iris_y"],
            sample["right_iris_x"],
            sample["right_iris_y"],
            pose["yaw"],
            pose["pitch"],
            pose["roll"],
            face_dist,
        ],
        dtype=np.float64,
    )


def polynomial_features(features):
    rows, columns = features.shape
    terms = [features]
    interactions = [
        features[:, i] * features[:, j]
        for i in range(columns)
        for j in range(i, columns)
    ]
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


def load_calibration_data(filepath):
    with Path(filepath).open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    if calibration.get("version") not in (3, 4):
        raise ValueError(
            "Format JSON de calibration non supporté (v3/v4 requis)."
        )

    features, targets, groups = [], [], []
    for point_index, point in enumerate(calibration.get("points", [])):
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
        raise ValueError(
            "Calibration insuffisante : minimum 6 points valides requis."
        )
    return (
        calibration,
        np.asarray(features),
        np.asarray(targets),
        np.asarray(groups),
    )


def choose_alpha(features, targets, groups):
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


def set_app_icon(window):
    """Charge l'icône dans la barre de titre (.ico ou .png)."""
    icon_path = Path(APP_ICON_PATH)
    if not icon_path.is_file():
        return

    try:
        if icon_path.suffix.lower() == ".ico":
            window.iconbitmap(default=str(icon_path))
        else:
            icon_img = Image.open(icon_path)
            photo = ImageTk.PhotoImage(icon_img)
            window.iconphoto(True, photo)
            window._icon_photo = photo
    except Exception as e:
        print(f"Impossible de charger l'icône : {e}")


def create_rounded_button(
    parent, text, command, width=260, height=48, radius=18
):
    """Crée un bouton sur-mesure (sans gras sur le texte)."""
    canvas = tk.Canvas(
        parent,
        width=width,
        height=height,
        bg=COLOR_BG,
        highlightthickness=0,
        bd=0,
        cursor="hand2",
    )

    def draw(bg_color):
        canvas.delete("all")
        x1, y1, x2, y2 = 1, 1, width - 1, height - 1
        r = radius
        points = [
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
        canvas.create_polygon(
            points,
            smooth=True,
            fill=bg_color,
            outline=COLOR_BTN_BORDER,
            width=1,
        )
        canvas.create_text(
            width / 2,
            height / 2,
            text=text,
            fill=COLOR_BTN_TEXT,
            font=(FONT_FAMILY, 11),
        )  # Police normale (sans "bold")

    draw(COLOR_BTN_BG)

    canvas.bind("<Enter>", lambda e: draw(COLOR_BTN_HOVER))
    canvas.bind("<Leave>", lambda e: draw(COLOR_BTN_BG))
    canvas.bind("<Button-1>", lambda e: command())

    return canvas


class EyeMouseApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Cursor")
        self.root.geometry("500x300")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        set_app_icon(self.root)
        apply_dark_title_bar(self.root)  # Applique la couleur exacte du body
        self.calibration_path = None
        self.build_ui()

    def build_ui(self):
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.place(relx=0.5, rely=0.5, anchor="center")

        lbl_title = tk.Label(
            main_frame,
            text="Cursor",
            font=(FONT_FAMILY, 26, "bold"),
            fg=COLOR_TEXT_MAIN,
            bg=COLOR_BG,
        )
        lbl_title.pack(pady=(0, 4))

        lbl_sub = tk.Label(
            main_frame,
            text="Controle de la souris par le regard !",
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_BG,
        )
        lbl_sub.pack(pady=(0, 24))

        btn_canvas = create_rounded_button(
            main_frame,
            text="Charger la calibration",
            command=self.select_calibration_file,
            width=270,
            height=46,
            radius=6,
        )
        btn_canvas.pack(pady=(0, 12))

        self.lbl_status = tk.Label(
            main_frame, text="", font=(FONT_FAMILY, 9), bg=COLOR_BG
        )
        self.lbl_status.pack()

    def select_calibration_file(self):
        filepath = filedialog.askopenfilename(
            title="Sélectionner le fichier de calibration",
            filetypes=[
                ("Calibration JSON", "*.json"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if filepath:
            self.calibration_path = filepath
            filename = Path(filepath).name
            self.lbl_status.config(text=f"✓ Chargé : {filename}", fg="#34D399")
            self.root.after(400, self.start_tracking)

    def start_tracking(self):
        try:
            calibration, features, targets, groups = load_calibration_data(
                self.calibration_path
            )
            alpha, _ = choose_alpha(features, targets, groups)
            mapper = RidgeMapper(alpha).fit(features, targets)
        except Exception as err:
            messagebox.showerror("Erreur de calibration", str(err))
            return

        try:
            self.root.destroy()
            run_tracking_overlay(calibration, mapper)
        except Exception as err:
            messagebox.showerror("Erreur lors du suivi", f"Détails : {err}")


def run_tracking_overlay(calibration, mapper):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fichier modèle introuvable : {MODEL_PATH}")

    screen_w = calibration["screen"]["width"]
    screen_h = calibration["screen"]["height"]

    cell_w = screen_w * GRID_PERCENT_X
    cell_h = screen_h * GRID_PERCENT_Y
    num_cols = int(1 / GRID_PERCENT_X)
    num_rows = int(1 / GRID_PERCENT_Y)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Erreur : Impossible d'accéder à la caméra.")

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
    timestamp_ms = 0

    smooth_x, smooth_y = None, None
    active_cell = None
    candidate_cell = None
    candidate_start_time = 0

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="#090D16")
    root.resizable(False, False)
    set_app_icon(root)

    canvas = tk.Canvas(
        root,
        width=screen_w,
        height=screen_h,
        bg="#090D16",
        highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)

    stop = {"value": False}

    def quit_app(event=None):
        stop["value"] = True

    root.bind("<Escape>", quit_app)
    root.focus_force()

    def process_stabilization(pos_x, pos_y, current_time_ms):
        nonlocal active_cell, candidate_cell, candidate_start_time

        if pos_x is None or pos_y is None:
            active_cell = None
            candidate_cell = None
            return None, None

        col = int(np.clip(pos_x // cell_w, 0, num_cols - 1))
        row = int(np.clip(pos_y // cell_h, 0, num_rows - 1))
        instant_cell = (col, row)

        if active_cell is None:
            active_cell = instant_cell
            return active_cell, None

        if instant_cell == active_cell:
            candidate_cell = None
            return active_cell, None

        if instant_cell != candidate_cell:
            candidate_cell = instant_cell
            candidate_start_time = current_time_ms
        else:
            elapsed = current_time_ms - candidate_start_time
            if elapsed >= STABILIZATION_TIME_MS:
                active_cell = candidate_cell
                candidate_cell = None

        return active_cell, candidate_cell

    def draw_grid(active, candidate):
        canvas.delete("all")

        # Case candidate en transition (Orange)
        if candidate is not None and candidate != active:
            c_col, c_row = candidate
            x1, y1 = c_col * cell_w, c_row * cell_h
            canvas.create_rectangle(
                x1 + 1,
                y1 + 1,
                x1 + cell_w - 1,
                y1 + cell_h - 1,
                fill=COLOR_CELL_TARGET,
                outline="#F97316",
                width=1,
            )

        # Case active fixée (Bleue)
        if active is not None:
            a_col, a_row = active
            x1, y1 = a_col * cell_w, a_row * cell_h
            canvas.create_rectangle(
                x1 + 1,
                y1 + 1,
                x1 + cell_w - 1,
                y1 + cell_h - 1,
                fill=COLOR_CELL_FILL,
                outline="#60A5FA",
                width=1,
            )

        # Lignes de la grille
        for c in range(num_cols + 1):
            x = c * cell_w
            canvas.create_line(x, 0, x, screen_h, fill=COLOR_GRID_LINE, width=1)

        for r in range(num_rows + 1):
            y = r * cell_h
            canvas.create_line(0, y, screen_w, y, fill=COLOR_GRID_LINE, width=1)

        # Cible au centre de la case active
        if active is not None:
            a_col, a_row = active
            center_x = (a_col + 0.5) * cell_w
            center_y = (a_row + 0.5) * cell_h

            r = min(cell_w, cell_h) * 0.18
            canvas.create_oval(
                center_x - r,
                center_y - r,
                center_x + r,
                center_y + r,
                fill="",
                outline=COLOR_POINT,
                width=2,
            )
            canvas.create_oval(
                center_x - 3,
                center_y - 3,
                center_x + 3,
                center_y + 3,
                fill="#FFFFFF",
                outline="",
            )

    try:
        while not stop["value"]:
            ok, frame = cap.read()
            if not ok:
                draw_grid(None, None)
                root.update()
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            result = landmarker.detect_for_video(image, timestamp_ms)

            if result.face_landmarks:
                face = result.face_landmarks[0]
                left = extract_eye_features(face, LEFT_EYE, IRIS_LEFT)
                right = extract_eye_features(face, RIGHT_EYE, IRIS_RIGHT)
                pose = extract_head_pose(result)
                face_dist = extract_face_distance(face)

                raw = feature_vector(left, right, pose, face_dist)[None, :]
                predicted = mapper.predict(raw)[0]

                rx = np.clip(predicted[0], 0, screen_w - 1)
                ry = np.clip(predicted[1], 0, screen_h - 1)

                dynamic_alpha = BASE_SMOOTHING_ALPHA * np.clip(
                    face_dist / 0.35, 0.5, 1.2
                )

                if smooth_x is None:
                    smooth_x, smooth_y = rx, ry
                else:
                    smooth_x = smooth_x + dynamic_alpha * (rx - smooth_x)
                    smooth_y = smooth_y + dynamic_alpha * (ry - smooth_y)

                active, candidate = process_stabilization(
                    smooth_x, smooth_y, timestamp_ms
                )
                draw_grid(active, candidate)
            else:
                smooth_x, smooth_y = None, None
                process_stabilization(None, None, timestamp_ms)
                draw_grid(None, None)

            root.update()
    finally:
        cap.release()
        landmarker.close()
        root.destroy()


if __name__ == "__main__":
    main_root = tk.Tk()
    app = EyeMouseApp(main_root)
    main_root.mainloop()