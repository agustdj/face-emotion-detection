# Face Emotion Detection Web App

A **Streamlit** web application that predicts facial emotions (Angry, Happy, Neutral) from **webcam snapshots** or **uploaded images** using **MTCNN for face detection** and **Mini-Xception for emotion classification**.

---

## Features

* Detects faces in images using **MTCNN**, robust to different angles, lighting, and small faces.
* Predicts **three emotion classes**:

  * Angry 😡
  * Happy 😄
  * Neutral 😐
* Displays prediction **confidence**.
* Works with **webcam** or **uploaded images**.
* Shows warnings if **no face is detected**.
* Shows sample prediction results with images.

---

## Requirements

* Python 3.8
* Streamlit
* TensorFlow
* OpenCV
* NumPy
* Pillow
* MTCNN

Install dependencies:

```bash
pip install streamlit tensorflow opencv-python-headless numpy pillow mtcnn
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/agustdj/face-emotion-detection.git
cd face-emotion-detection
```

2. Make sure the **Mini-Xception model** is in `models/`.

3. Run the app:

```bash
streamlit run app.py
```

4. Open the browser link provided by Streamlit.

---

## Usage

1. Select **Webcam** to capture a snapshot or **Upload Image** to choose a photo.
2. The app detects the first face in the image and predicts the emotion.
3. If no face is detected, a warning appears.

---

## Sample Results

| Angry 😡                                                  | Happy 😄                                                  | Neutral 😐                                                    |
| --------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| ![Angry](assets/angry_example.jpg){width="200px"} | ![Happy](assets/happy_example.jpg){width="200px"} | ![Neutral](assets/neutral_example.jpg){width="200px"} |

---

## Notes

* Currently supports **single-face detection** for simplicity.
* Emotion prediction is limited to **Angry, Happy, Neutral**.
* Use clear frontal images for best accuracy.

