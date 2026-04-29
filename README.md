# 🎙️ Multilingual Speech Recognition using Whisper

## 📌 Overview
This project focuses on building an **automatic speech recognition (ASR) system** capable of transcribing audio across multiple languages. The objective is to convert raw audio inputs into **accurate verbatim text transcripts** under limited computational constraints.

_comptetion link: https://www.kaggle.com/competitions/multilingual-speech-recognition_

Developed as part of the **NPPE-2 Multilingual Speech Recognition Challenge**, the project emphasizes **data preprocessing, model selection, and efficient inference pipelines** for real-world ASR tasks.

---

## 🚀 Key Highlights
- Built a **multilingual ASR pipeline** using pretrained models
- Processed raw audio files into text transcripts
- Focused on **efficient inference under compute constraints (Kaggle)**
- Evaluated using **Word Error Rate (WER)**
- Designed an **end-to-end audio → text system**

---

## 🧠 Problem Statement
Develop a system that accurately transcribes speech from audio recordings across multiple languages.

### Challenges:
- Multilingual speech variation  
- Noisy and diverse audio inputs  
- Limited compute (Kaggle-only environment)  
- Need for efficient preprocessing + inference  

---

## ⚙️ Approach

### 🔹 Model
- **Whisper / Faster-Whisper (Transformer-based ASR model)**

### 🔹 Pipeline

1. **Data Loading**
   - Audio files from dataset
   - Metadata from CSV files

2. **Audio Preprocessing**
   - Loading `.wav` files
   - Resampling / normalization (if required)

3. **Model Inference**
   - Transcription using pretrained Whisper model
   - Batch or sequential processing

4. **Post-processing**
   - Cleaning predicted transcripts
   - Formatting outputs

5. **Submission Generation**
   - Creating CSV with predicted transcripts

---

## 📊 Dataset

### Files:
- `train.csv` → labeled transcripts  
- `test.csv` → audio inputs for prediction  
- `competition_data/` → audio files  
- `sample_submission.csv` → format reference  

### Format:
| audio        | text                     |
|-------------|--------------------------|
| audio_00000.wav | this is first prediction |

---

## 📈 Evaluation Metric

### Word Error Rate (WER)

Measures transcription accuracy:
```WER = (Substitutions + Deletions + Insertions) / Total Words```


- **Lower is better**
- 0.0 = perfect transcription

---

## 🧠 Key Learnings

- Importance of **audio preprocessing in ASR systems**
- Working with **large pretrained speech models (Whisper)**
- Handling **multilingual audio variability**
- Trade-offs between **speed vs transcription accuracy**
- Building **end-to-end ML pipelines for real-world tasks**

---

## ⚡ Limitations

- Limited ability to fine-tune due to compute constraints  
- Performance may vary across languages  
- Dependency on pretrained model capabilities  

---

## 🔮 Future Improvements

- Fine-tuning Whisper on domain-specific audio  
- Using **wav2vec2 / hybrid models**  
- Data augmentation (noise injection, speed variation)  
- Language-specific optimization  
- Batch inference for faster processing  

---

## 🧑‍💻 Author
**Sagandeep Kaur**  
