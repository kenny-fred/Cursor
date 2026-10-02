import json
import numpy as np
from pathlib import Path

JSON_FILE = "calibration_screen_v2.json"

FEATURES = [
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
]

POSE_FEATURES = [
    "yaw",
    "pitch",
    "roll",
]


def extract_vector(sample):
    vec = []

    for f in FEATURES:
        vec.append(sample[f])

    for f in POSE_FEATURES:
        vec.append(sample["head_pose"][f])

    return np.array(vec, dtype=np.float64)


def score_dataset(data):

    points = data["points"]

    point_centers = []
    point_noise = []

    total_samples = 0

    print("\n========== ANALYSE DES POINTS ==========\n")

    for idx, point in enumerate(points):

        vectors = np.array(
            [extract_vector(s) for s in point["samples"]],
            dtype=np.float64
        )

        total_samples += len(vectors)

        center = np.mean(vectors, axis=0)
        std = np.std(vectors, axis=0)

        mean_noise = np.mean(std)

        point_centers.append(center)
        point_noise.append(mean_noise)

        screen_x = point["screen"]["x"]
        screen_y = point["screen"]["y"]

        print(
            f"Point {idx:02d} "
            f"({screen_x:4d},{screen_y:4d}) "
            f"samples={len(vectors):2d} "
            f"noise={mean_noise:.5f}"
        )

    point_centers = np.array(point_centers)

    print("\n========== SEPARATION ==========\n")

    separations = []

    for i in range(len(point_centers)):
        for j in range(i + 1, len(point_centers)):

            d = np.linalg.norm(
                point_centers[i] -
                point_centers[j]
            )

            separations.append(d)

    mean_separation = np.mean(separations)
    min_separation = np.min(separations)

    print(f"Séparation moyenne : {mean_separation:.5f}")
    print(f"Séparation minimale : {min_separation:.5f}")

    print("\n========== STABILITE ==========\n")

    mean_noise = np.mean(point_noise)
    max_noise = np.max(point_noise)

    print(f"Bruit moyen : {mean_noise:.5f}")
    print(f"Bruit maximum : {max_noise:.5f}")

    signal_noise_ratio = mean_separation / (
        mean_noise + 1e-9
    )

    print(
        f"Signal/Bruit : {signal_noise_ratio:.2f}"
    )

    quality = min(
        100,
        max(
            0,
            signal_noise_ratio * 8
        )
    )

    print("\n========== RAPPORT ==========\n")

    print(
        f"Version dataset : {data['version']}"
    )

    print(
        f"Points calibration : {len(points)}"
    )

    print(
        f"Samples total : {total_samples}"
    )

    print(
        f"Score dataset : {quality:.1f}/100"
    )

    if quality >= 85:
        level = "EXCELLENT"

    elif quality >= 70:
        level = "BON"

    elif quality >= 50:
        level = "MOYEN"

    else:
        level = "FAIBLE"

    print(f"Qualité : {level}")

    return quality


def main():

    path = Path(JSON_FILE)

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n================================")
    print(" MOUSE-EYE DATASET ANALYZER")
    print("================================")

    score_dataset(data)


if __name__ == "__main__":
    main()