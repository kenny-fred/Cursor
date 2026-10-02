import cv2
import mediapipe as mp

# MODÈLE
MODEL_PATH = "models/face_landmarker.task"

# MEDIAPIPE
mp_vision = mp.tasks.vision


# Configuration du Face Landmarker.
options = mp_vision.FaceLandmarkerOptions(

    # Chemin vers notre modèle.
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    # Mode vidéo = plusieurs images successives.
    running_mode=mp_vision.RunningMode.VIDEO,

    # Un seul visage nous suffit.
    num_faces=1,

    # Confiance minimale de détection.
    min_face_detection_confidence=0.5,

    # Confiance minimale de présence.
    min_face_presence_confidence=0.5,

    # Confiance minimale du tracking.
    min_tracking_confidence=0.5,

    # On veut les landmarks, mais pas les blendshapes.
    output_face_blendshapes=False,

    # Pas besoin de transformation 3D pour l'instant.
    output_facial_transformation_matrixes=False
)

# Création du Face Landmarker.
landmarker = mp_vision.FaceLandmarker.create_from_options(
    options
)

# WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera : {width}x{height}")


WINDOW_NAME = "Cursor"

cv2.namedWindow(WINDOW_NAME)

# Timestamp obligatoire pour le mode VIDEO.
timestamp_ms = 0

# LANDMARKS IMPORTANTS
# Ces indices correspondent aux landmarks de l'iris.
# 5 points pour un iris :
#
#        ●
#     ●  ●  ●
#        ●
IRIS_1 = [468, 469, 470, 471, 472]
IRIS_2 = [473, 474, 475, 476, 477]

# BOUCLE
while True:
    success, frame = cap.read()
    if not success:
        print("Impossible de lire une image")
        break

    # Conversion BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Création de l'image MediaPipe.
    mp_frame = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Détection
    result = landmarker.detect_for_video(
        mp_frame,
        timestamp_ms
    )

    timestamp_ms += 33

    # Si un visage est trouvé
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # DESSIN DES IRIS
        for index in IRIS_1:

            point = landmarks[index]

            px = int(point.x * width)
            py = int(point.y * height)

            cv2.circle(
                frame,
                (px, py),
                3,
                (0, 0, 255),
                -1
            )

        for index in IRIS_2:

            point = landmarks[index]

            px = int(point.x * width)
            py = int(point.y * height)

            cv2.circle(
                frame,
                (px, py),
                3,
                (0, 0, 255),
                -1
            )

        # CALCUL DU CENTRE DE CHAQUE IRIS
        def iris_center(indices):

            x_values = []
            y_values = []

            for index in indices:

                point = landmarks[index]

                x_values.append(point.x)
                y_values.append(point.y)

            center_x = sum(x_values) / len(x_values)
            center_y = sum(y_values) / len(y_values)

            return (
                int(center_x * width),
                int(center_y * height)
            )

        # Centre iris 1.
        center_1 = iris_center(IRIS_1)

        # Centre iris 2.
        center_2 = iris_center(IRIS_2)

        # DESSIN DES CENTRES
        cv2.circle(
            frame,
            center_1,
            7,
            (255, 0, 0),
            2
        )

        cv2.circle(
            frame,
            center_2,
            7,
            (255, 0, 0),
            2
        )

        # 9. AFFICHAGE DES COORDONNÉES
        cv2.putText(
            frame,
            f"Iris 1: {center_1}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Iris 2: {center_2}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # AFFICHAGE
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    # Fermeture avec X.
    try:
        visible = cv2.getWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_VISIBLE
        )

        if visible < 1:
            break

    except cv2.error:
        break

# NETTOYAGE
cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("Cursor arrêté proprement.")