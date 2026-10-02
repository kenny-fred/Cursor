# 🧠 Core idea

Most eye-tracking systems can be represented approximately as:

```text
Webcam
   ↓
Face / Eye detection
   ↓
Gaze estimation
   ↓
Screen coordinates
   ↓
Mouse cursor
```

Cursor takes a slightly different approach:

```text
                  ┌── Iris position
Webcam ───────────┼── Eye geometry
                  └── Head pose
                         ↓
                 Personal calibration
                         ↓
                  Feature → Screen
                     mapping
                         ↓
                  Filtering / Smoothing
                         ↓
                  Windows cursor
```

The system does not attempt to answer:

> "Where does a human look in 3D?"

Instead, it asks:

> **"When this user produces these visual features, which point on this screen are they looking at?"**

This makes the calibration personal to the user and their physical setup.


# 🔬 Calibration

Calibration is the central component of Cursor.

A target appears at different positions on the screen. The user looks at each target while the webcam captures the corresponding eye and head measurements.

For each calibration point, Cursor stores a lightweight numerical representation.

For example:

```json
{
  "screen": {
    "x": 640,
    "y": 360
  },
  "samples": [
    {
      "left_iris_x": 0.51,
      "left_iris_y": 0.48,
      "right_iris_x": 0.50,
      "right_iris_y": 0.49,
      "head_yaw": 0.02,
      "head_pitch": -0.01
    }
  ]
}
```

The actual dataset contains **measurements rather than photographs or video frames**.

This makes the calibration data:

* lightweight;
* easy to inspect;
* easy to regenerate;
* specific to the user;
* suitable for local processing.

> **Privacy note:** Cursor is designed so that calibration data can consist only of extracted numerical features. This is different from claiming that no camera data is ever processed: the webcam frames still have to be processed in memory to extract those features.

# 🏗️ Architecture

The first version is divided into several independent components.

### `camera_test.py`

Hardware validation.

Responsibilities:

* open the webcam;
* verify that the camera is accessible;
* read frames;
* display the camera stream;
* handle clean shutdown.

It is deliberately simple and is not an eye-tracking algorithm.

---

### `eye_test.py`

First eye-tracking prototype.

Uses:

* OpenCV;
* NumPy;
* MediaPipe.

It extracts information about:

* eye geometry;
* iris position;
* facial landmarks;
* head pose.

---

### `calibration.py`

The main calibration pipeline.

It:

1. displays calibration targets;
2. collects measurements;
3. associates measurements with screen coordinates;
4. stores the resulting personal dataset as JSON.

---

### `accuracy.py`

Evaluation and analysis.

It compares predicted positions with reference positions and can be used to investigate:

* positional error;
* signal/noise characteristics;
* calibration quality;
* outliers;
* consistency between calibration points.

---

### `calib_test.py`

Real-time cursor control.

It converts the measurements obtained from the webcam into Windows cursor movements using the calibration model.

# 🟦🟨 Grid Map

Cursor also experiments with a grid-based mechanism for improving cursor stability.

The grid currently contains two conceptual states.

### 🟨 Prediction / movement region

The yellow region represents the part of the system concerned with **movement anticipation**.

When the extracted eye/head features begin moving consistently toward another region, the system can react before the gaze has completely stabilized.

The objective is to reduce perceived cursor latency.

---

### 🟦 Stable region

The blue region represents a stable gaze area.

When the user's measurements remain sufficiently stable, the system can maintain the corresponding cursor position instead of continuously reacting to small variations.

The objective is to reduce:

* webcam noise;
* cursor jitter;
* unnecessary recalculation;
* small unwanted cursor movements.

This mechanism is an experimental part of the project and is being evaluated rather than presented as a universally optimal solution.

---

# 📊 Current architecture

```text
             ┌─────────────────┐
             │     Webcam      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    MediaPipe    │
             │ Face landmarks  │
             │     + iris      │
             └────────┬────────┘
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
       Iris features       Head pose
             │                 │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │   Calibration   │
             │ Personal model  │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │    Filtering    │
             │  + Grid logic   │
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Windows Cursor  │
             └─────────────────┘
```

# 🧪 Technologies

| Technology                  | Purpose                     |
| --------------------------- | --------------------------- |
| Python                      | Main programming language   |
| OpenCV                      | Webcam and image processing |
| MediaPipe                   | Face and iris landmarks     |
| NumPy                       | Numerical processing        |
| JSON                        | Calibration dataset         |
| Windows API / mouse control | Cursor control              |


# 💻 Requirements

* Windows
* Python 3.x
* Standard webcam
* Working microphone is **not required**
* No dedicated eye-tracking hardware

Install the Python dependencies:

```bash
pip install opencv-python numpy mediapipe
```

---

# 🚀 Getting started

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Cursor.git
cd Cursor
```

Test the webcam:

```bash
python camera_test.py
```

Test eye detection:

```bash
python eye_test.py
```

Create a personal calibration dataset:

```bash
python calibration.py
```

Test the calibrated cursor:

```bash
python calib_test.py
```

---

# 📈 Evaluation

Cursor is still an experimental project.

Rather than claiming a fixed accuracy, the project uses calibration data to evaluate the actual performance of the system.

Possible metrics include:

```text
Mean Error
Maximum Error
Median Error
Signal / Noise Ratio
Calibration consistency
Cursor jitter
Latency
CPU usage
```

Future experiments will compare different:

* calibration grid sizes;
* regression models;
* filtering methods;
* lighting conditions;
* camera positions;
* user distances from the screen.

# 🔒 Privacy

Cursor is designed around **local processing**.
```
The webcam is used to extract numerical features such as iris position and head pose.

The calibration dataset can contain numerical measurements rather than raw photographs or video.

No cloud service is required for the core eye-tracking pipeline.

MIT License
```
