# Speech Emotion Recognition Model

A Python-based Speech Emotion Recognition project built in a Kaggle Notebook environment.

The project focuses on extracting acoustic features from speech audio files and using those features to train an **Artificial Neural Network (ANN)** model for emotion classification.

---

## Overview

Speech Emotion Recognition (SER) is the task of detecting a speaker's emotional state from their voice.

Instead of analyzing the meaning of the spoken words, SER systems mainly analyze acoustic patterns such as:

* Pitch
* Energy
* Tone
* Rhythm
* Spectral information
* Voice intensity
* Frequency distribution

This project demonstrates a classical machine learning / deep learning pipeline for SER:

```text
Audio Files
    ↓
Audio Loading
    ↓
Waveform Visualization
    ↓
Feature Extraction
    ↓
Feature Vector Creation
    ↓
ANN Model Training
    ↓
Emotion Prediction
```

---

## What This Project Does

This project shows how to:

* Load speech audio files in Python
* Play and visualize audio signals
* Compare waveforms of different emotions
* Extract useful audio features with `librosa`
* Prepare feature vectors for machine learning
* Train an ANN-based emotion classification model
* Build a baseline speech emotion recognition system

---

## Dataset

The notebook uses a Kaggle dataset located under:

```text
/kaggle/input/speech-emotion-recognition-en/
```

The visible examples in the notebook use CREMA-style audio files such as:

```text
/kaggle/input/speech-emotion-recognition-en/Crema/1001_IEO_HAP_HI.wav
/kaggle/input/speech-emotion-recognition-en/Crema/1001_IEO_SAD_HI.wav
```

These filenames include emotion codes such as:

| Code  | Emotion |
| ----- | ------- |
| `HAP` | Happy   |
| `SAD` | Sad     |
| `ANG` | Angry   |
| `FEA` | Fear    |
| `DIS` | Disgust |
| `NEU` | Neutral |

Depending on the full dataset version used in Kaggle, the dataset may contain multiple speakers and multiple acted emotional speech recordings.

---

## Example Audio Files

The notebook starts by comparing two opposite emotions:

```python
happy_audio_path = "/kaggle/input/speech-emotion-recognition-en/Crema/1001_IEO_HAP_HI.wav"
sad_audio_path = "/kaggle/input/speech-emotion-recognition-en/Crema/1001_IEO_SAD_HI.wav"
```

The goal is to visually and acoustically inspect how different emotions may appear in the waveform.

---

## Audio Visualization

The project uses `librosa` and `IPython.display` to load, play, and visualize audio files.

Example function:

```python
def audio_display_and_waveshow(audio_path):
    au_obj = ipd.Audio(audio_path)
    audio, sr = librosa.load(audio_path)

    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(audio, alpha=0.7)
    plt.title(audio_path.split("/")[-1])
    plt.show()

    return au_obj
```

This function:

1. Loads an audio file
2. Plays the audio inside the notebook
3. Displays the waveform
4. Helps compare emotional speech signals visually

---

## Feature Extraction

Speech emotion recognition models usually do not use raw audio directly in classical ANN pipelines.

Instead, the audio signal is converted into numerical features.

Common audio features for SER include:

### 1. MFCC

**MFCC** stands for Mel-Frequency Cepstral Coefficients.

MFCCs are widely used in speech and audio processing because they summarize the spectral shape of the sound in a way that is closer to human hearing.

MFCCs can help capture:

```text
voice tone
speaker timbre
frequency structure
speech characteristics
```

---

### 2. Chroma Features

Chroma features describe the distribution of energy across pitch classes.

They are often used in music analysis, but can also provide useful spectral information for speech-related tasks.

---

### 3. Mel Spectrogram

A Mel spectrogram represents the energy of the audio signal over time and frequency using the Mel scale.

It is useful because it captures how the frequency content of speech changes over time.

---

### 4. Spectral Features

Other spectral features may include:

```text
spectral centroid
spectral bandwidth
spectral contrast
zero crossing rate
root mean square energy
```

These can help represent loudness, sharpness, noisiness, and other acoustic properties of the voice.

---

## Model

The project builds an **ANN model** for emotion classification.

A typical ANN-based SER architecture looks like this:

```text
Extracted Audio Features
        ↓
Dense Layer
        ↓
Activation Function
        ↓
Dropout / Regularization
        ↓
Dense Layer
        ↓
Softmax Output
        ↓
Predicted Emotion
```

The output layer predicts one of the emotion classes.

Example output classes:

```text
happy
sad
angry
fear
disgust
neutral
```

---

## Input and Output

### Input

The input is a speech audio file, usually in `.wav` format.

Example:

```text
1001_IEO_HAP_HI.wav
```

### Output

The output is the predicted emotion label.

Example:

```text
happy
```

A model can also return probabilities for each emotion:

```text
happy:   0.72
sad:     0.04
angry:   0.08
fear:    0.06
disgust: 0.03
neutral: 0.07
```

---

## General Pipeline

The full pipeline can be described as:

```text
1. Load audio file
2. Visualize waveform
3. Extract acoustic features
4. Convert features into a numerical vector
5. Encode emotion labels
6. Split data into train/test sets
7. Train ANN classifier
8. Evaluate model performance
9. Predict emotion for new audio
```

---

## Repository Structure

Current repository structure:

```text
Speech-Emotion-Recognition-Model/
│
├── README.md
└── speech-emotion-recognition-model.ipynb
```

Suggested future structure:

```text
Speech-Emotion-Recognition-Model/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── speech-emotion-recognition-model.ipynb
├── src/
│   ├── audio_loader.py
│   ├── feature_extraction.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── assets/
│   ├── waveform_happy.png
│   ├── waveform_sad.png
│   ├── confusion_matrix.png
│   └── training_history.png
└── examples/
    └── sample_predictions.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/VafaKnm/Speech-Emotion-Recognition-Model.git
cd Speech-Emotion-Recognition-Model
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
pip install numpy pandas librosa matplotlib seaborn scikit-learn tensorflow keras jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
speech-emotion-recognition-model.ipynb
```

---

## Suggested `requirements.txt`

A useful `requirements.txt` file for this project could be:

```txt
numpy
pandas
librosa
matplotlib
seaborn
scikit-learn
tensorflow
keras
ipython
jupyter
```

---

## Example Feature Extraction Function

A future script-based version of the project could include a reusable feature extraction function like this:

```python
import librosa
import numpy as np

def extract_features(audio_path):
    """
    Extract acoustic features from a speech audio file.
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean
```

This converts an audio file into a numerical vector that can be passed to a machine learning model.

---

## Example Inference Function

A future inference function could look like this:

```python
def predict_emotion(audio_path, model, label_encoder):
    """
    Predict emotion from an audio file.
    """
    features = extract_features(audio_path)
    features = features.reshape(1, -1)

    prediction = model.predict(features)
    predicted_index = prediction.argmax(axis=1)[0]

    emotion = label_encoder.inverse_transform([predicted_index])[0]

    return emotion
```

Example:

```python
predict_emotion("sample.wav", model, label_encoder)
```

Expected output:

```text
happy
```

---

## Evaluation

The current repository README does not report a final numeric accuracy, precision, recall, or F1-score.

A stronger evaluation section should include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Per-class performance
* Training and validation loss curves
* Training and validation accuracy curves

Example evaluation code:

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

print(classification_report(y_test, y_pred_classes))
```

---

## Recommended Metrics

Speech emotion recognition should be evaluated carefully because class imbalance and speaker differences can strongly affect results.

Recommended metrics:

| Metric                       | Why It Matters                                   |
| ---------------------------- | ------------------------------------------------ |
| Accuracy                     | Overall classification performance               |
| Macro F1                     | Treats all emotions equally                      |
| Weighted F1                  | Accounts for class frequency                     |
| Confusion Matrix             | Shows which emotions are confused                |
| Per-class Recall             | Shows whether the model misses specific emotions |
| Speaker-independent Accuracy | Tests generalization to unseen speakers          |

---

## Suggested Improvements

This repository can be improved in several practical ways.

### 1. Add a Full Evaluation Report

The README should eventually include a table like this:

| Metric      | Score |
| ----------- | ----: |
| Accuracy    |   TBD |
| Macro F1    |   TBD |
| Weighted F1 |   TBD |

And a per-class table like this:

| Emotion | Precision | Recall | F1-score |
| ------- | --------: | -----: | -------: |
| Happy   |       TBD |    TBD |      TBD |
| Sad     |       TBD |    TBD |      TBD |
| Angry   |       TBD |    TBD |      TBD |
| Fear    |       TBD |    TBD |      TBD |
| Disgust |       TBD |    TBD |      TBD |
| Neutral |       TBD |    TBD |      TBD |

---

### 2. Add Confusion Matrix

A confusion matrix is very important for SER because some emotions are naturally similar.

For example, models may confuse:

```text
sad ↔ neutral
fear ↔ angry
happy ↔ neutral
disgust ↔ angry
```

Suggested code:

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Speech Emotion Recognition Confusion Matrix")
plt.show()
```

---

### 3. Add Data Augmentation

Speech models often benefit from audio augmentation.

Useful augmentation methods:

```text
noise injection
time stretching
pitch shifting
random gain
time shifting
speed perturbation
```

Example:

```python
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio
```

---

### 4. Compare Multiple Models

The current ANN model is a good baseline. Future versions can compare it with stronger models:

| Model       | Input Type                | Notes                                   |
| ----------- | ------------------------- | --------------------------------------- |
| ANN         | Handcrafted features      | Current baseline                        |
| CNN         | Mel spectrogram           | Learns local time-frequency patterns    |
| LSTM        | MFCC sequence             | Models temporal structure               |
| CNN-LSTM    | Spectrogram/MFCC sequence | Combines spatial and temporal modeling  |
| Wav2Vec2    | Raw audio / embeddings    | Strong pretrained speech representation |
| HuBERT      | Raw audio / embeddings    | Self-supervised speech model            |
| emotion2vec | Speech emotion embeddings | Specialized SER representation model    |

---

### 5. Add Speaker-Independent Evaluation

A common problem in speech emotion recognition is speaker leakage.

If the same speaker appears in both train and test sets, the model may learn speaker characteristics instead of emotion.

A better evaluation setup is:

```text
Train speakers: speaker IDs 1–80
Test speakers:  speaker IDs 81–100
```

This gives a more realistic estimate of performance on unseen speakers.

---

### 6. Add Model Saving and Loading

The project should save the trained model:

```python
model.save("models/ser_ann_model.h5")
```

And load it later:

```python
from tensorflow.keras.models import load_model

model = load_model("models/ser_ann_model.h5")
```

This makes the project usable for inference without retraining.

---

### 7. Add a Demo App

A small demo would make the project much easier to understand.

Recommended tools:

* Streamlit
* Gradio
* FastAPI

Example UI:

```text
Upload .wav file
      ↓
Play audio
      ↓
Show waveform
      ↓
Predict emotion
      ↓
Display emotion probabilities
```

---

## Possible Future Work

Recommended future work:

* Add `requirements.txt`
* Add model checkpoint
* Add saved label encoder
* Add inference script
* Add confusion matrix image
* Add classification report
* Add training history plot
* Add speaker-independent train/test split
* Add audio augmentation
* Add comparison with CNN and LSTM models
* Add Wav2Vec2 or HuBERT embeddings
* Add Streamlit or Gradio demo
* Add Dockerfile
* Add API endpoint for audio upload and emotion prediction

---

## Limitations

The current project has some limitations:

* It is notebook-based and not yet structured as a reusable Python package.
* The current README does not report final numerical evaluation metrics.
* The trained model checkpoint is not published in the repository.
* The dataset must be downloaded separately from Kaggle.
* The model is based on extracted acoustic features rather than end-to-end raw audio training.
* ANN models may not capture temporal speech dynamics as well as CNN, LSTM, or Transformer-based models.
* Real-world speech emotion recognition is difficult because emotion depends on speaker, language, context, recording quality, and culture.

---

## Use Cases

Speech emotion recognition can be useful in:

* Call center analytics
* Human-computer interaction
* Customer support monitoring
* Voice assistant personalization
* Mental health research support
* E-learning feedback systems
* Multimedia indexing
* Affective computing research

This project should be considered an educational baseline rather than a production-ready emotion recognition system.

---

## Ethical Considerations

Emotion recognition is sensitive and can be misused.

Important considerations:

* Do not use the model for high-stakes decisions.
* Do not assume the model can truly understand a person's emotional state.
* Predictions may be wrong, biased, or affected by recording conditions.
* Emotion expression varies across people, languages, cultures, and contexts.
* Users should know when their voice is being analyzed.
* Consent and privacy are very important when working with speech data.

---

## Conclusion

This repository demonstrates a practical baseline for speech emotion recognition using Python, `librosa`, acoustic feature extraction, and an ANN classifier.

The project is useful for learning:

```text
audio loading
waveform visualization
feature extraction
emotion label preparation
ANN-based classification
speech emotion recognition workflow
```

With better evaluation, cleaner code structure, model comparison, speaker-independent testing, and a simple demo app, this project can become a much stronger and more practical SER repository.
