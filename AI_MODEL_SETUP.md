# VoiceShield — AI Voice Cloning Detection Pipeline Specification
**Smart India Hackathon 2026 | PS ID: 26104**
**Title:** AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks

---

## 1. Architectural Overview

VoiceShield delivers real-time, low-latency detection of AI-generated synthetic voice and voice cloning impersonation attacks during live peer-to-peer WebRTC calls.

```
[Android Microphone]
        │
        ▼ (Exclusive Audio Capture)
[WebRTC JavaAudioDeviceModule]
   ├── AudioTrack / RTP Transport ──────────────► Peer Phone (Real-Time Voice Call)
   │
   └── setSamplesReadyCallback (Zero-Latency Audio Tap)
               │
               ▼ (Raw Signed 16-bit PCM Frames)
   [OkHttp WebSocket: /ws/audio/room1?sample_rate=16000&channels=1]
               │
               ▼
   [FastAPI Backend Ingestion]
               │
               ▼
   [Thread-Safe Sliding Ring Buffer (3.0s Window / 1.0s Hop)]
               │
               ▼
   [Audio Preprocessor: Resample -> DC Removal -> Peak Normalization]
               │
               ▼
   ┌────────────────────────────────────────────────────────┐
   │                  AI DETECTION PIPELINE                 │
   │                                                        │
   │  [Silero VAD] (ONNX, 2.3MB)                            │
   │      │                                                 │
   │      ├── Speech Active (P >= 0.50)                     │
   │      │      │                                          │
   │      │      ├──► [RawNet3] (PyTorch, 65MB)             │
   │      │      │    SincConv1d + ResNet + GRU + ASP       │
   │      │      │    (Synthetic Spoof Detection)           │
   │      │      │                                          │
   │      │      ├──► [ECAPA-TDNN] (ONNX, 24.8MB)           │
   │      │      │    192-dim VoxCeleb Speaker Embeddings   │
   │      │      │    (Speaker Identity Verification)       │
   │      │      │                                          │
   │      │      └──► [Bispectral Analyzer] (HOS)           │
   │      │           2D Bicoherence + Harmonic Coupling    │
   │      │           (Neural Vocoder Phase Artifacts)      │
   │      │                                                 │
   │      └── Silence (P < 0.50) ──► Gated (No Fake Scores) │
   └─────────────────────────┬──────────────────────────────┘
                             │
                             ▼
            [Calibrated Score Fusion Engine]
         Dynamic Weight Renormalization across Active Models
                             │
                             ▼
   [Signaling WebSocket Broadcast: /ws/room1 {"type": "ai_result"}]
                             │
                             ▼
               [Android ConnectedScreen UI]
   • Live Risk Score & Meter (Green / Orange / Red)
   • Confidence & Inference Latency
   • 4-Model Transparent Inspection Breakdown
   • Real-Time Cloned Voice Impersonation Warning Banner
```

---

## 2. Models & Weights Inventory

All pre-trained deep learning weights are locally provisioned in `backend/models/`:

| Model | Checkpoint File | Size | Format | Runtime | Task |
|---|---|---|---|---|---|
| **Silero VAD v4** | `backend/models/silero_vad.onnx` | 2.3 MB | ONNX | ONNX Runtime (CPU) | Voice Activity Detection & Speech Gating |
| **RawNet3** | `backend/models/rawnet3.pt` | 65.0 MB | PyTorch (`.pt`) | PyTorch CPU (234 layers) | Deepfake Audio & Voice Cloning Anti-Spoofing |
| **ECAPA-TDNN** | `backend/models/ecapa_tdnn.onnx` | 24.8 MB | ONNX | ONNX Runtime (CPU) | 192-dim Speaker Identity Verification |
| **Bispectral HOS** | Mathematical Engine | — | NumPy / SciPy | Vectorized CPU | Neural Vocoder Quadratic Phase Coupling Detection |

---

## 3. Detailed Component Specifications

### 3.1 WebRTC Audio Tap & Transport
* **Capture Layer:** `org.webrtc.audio.JavaAudioDeviceModule.Builder.setSamplesReadyCallback`.
* **Zero Duplication Guarantee:** Android's `AudioRecord` is instantiated exclusively by WebRTC. The detection pipeline operates purely on tapped PCM frames without competing for hardware microphone ownership.
* **Streaming Protocol:** Raw binary PCM sent over OkHttp WebSocket to `ws://{HOST}:8000/ws/audio/{room_id}?sample_rate={sr}&channels={ch}`.
* **Stereo Handling:** Automatic downmix to mono in `AudioPreprocessor` if the device records in stereo.
* **Resampling:** Polyphase resampling (`scipy.signal.resample_poly`) to exactly 16,000 Hz.

### 3.2 Thread-Safe Ring Buffer (`backend/audio/buffer.py`)
* **Window Duration:** 3.0 seconds (48,000 samples @ 16kHz).
* **Hop Duration:** 1.0 second (16,000 samples @ 16kHz).
* **Queue Strategy:** Atomic `get_and_advance()`, overflow capping at 10.0s to prevent memory accumulation.

### 3.3 Silero Voice Activity Detector (`backend/ai/vad.py`)
* **Purpose:** Gates deep inference to prevent computing fake spoof scores on background room silence.
* **Input:** 512-sample frames @ 16kHz float32.
* **Hidden State:** Persistent LSTM hidden state $(h, c)$ tracked across consecutive frames.
* **Decision Rule:** Segment classified as speech if mean frame speech probability $P_{\text{speech}} \ge 0.50$ or speech ratio $\ge 0.35$.

### 3.4 RawNet3 Spoof Detector (`backend/ai/rawnet3.py`)
* **Architecture:**
  1. Parameterized Sinc-convolutional filterbank (`SincConv1d`) directly operating on raw time-domain waveforms.
  2. Residual blocks with Squeeze-and-Excitation (SE) channel attention.
  3. Bidirectional Gated Recurrent Units (GRU).
  4. Attentive Statistics Pooling (ASP) capturing temporal mean and standard deviation.
  5. Dense fully-connected projection with calibrated spoof probability mapping.
* **Input:** Float32 tensor of shape `(1, 48000)`.
* **Output:** `rawnet_risk` $\in [0.05, 0.95]$.

### 3.5 ECAPA-TDNN Speaker Verification (`backend/ai/ecapa.py`)
* **Architecture:** 1D Time-Delay Neural Network with multi-layer feature aggregation and channel/context attention.
* **Feature Extraction:** 80-dimensional log Mel-frequency filterbanks (25ms window, 10ms frame shift).
* **Embedding Dimension:** 192-dimensional unit-normalized vector.
* **Enrollment Policy (Honest Verification):**
  - If a contact is enrolled: computes cosine similarity $\cos(\theta) = \frac{\mathbf{e}_{\text{live}} \cdot \mathbf{e}_{\text{ref}}}{\|\mathbf{e}_{\text{live}}\| \|\mathbf{e}_{\text{ref}}\|}$. Risk is mapped inversely: $R_{\text{speaker}} = 1.0 - \max(0, \cos(\theta))$.
  - If contact is not enrolled: returns `available: False`, `speaker_risk: None`, and status `"NOT_ENROLLED"`. **NO fake or random scores are generated.**

### 3.6 Bispectral Higher-Order Spectral Analyzer (`backend/ai/bispectral.py`)
* **Theory:** Neural vocoders (HiFi-GAN, WaveGlow, MelGAN, BigVGAN) introduce subtle phase incoherence and non-linear harmonic coupling artifacts in the second-order bifrequency spectrum.
* **Formulation:**
  $$\text{Bispectrum: } B(f_1, f_2) = \mathbb{E}\left[X(f_1) X(f_2) X^*(f_1 + f_2)\right]$$
  $$\text{Bicoherence: } b^2(f_1, f_2) = \frac{\left|B(f_1, f_2)\right|^2}{\mathbb{E}\left[|X(f_1) X(f_2)|^2\right] \mathbb{E}\left[|X(f_1 + f_2)|^2\right]}$$
* **Metrics Extracted:** Harmonic band bicoherence, diagonal slice variance, high-frequency harmonic energy ratio ($>4\text{kHz}$), and normalized phase entropy.
* **Output:** `bispectral_risk` $\in [0.05, 0.95]$.

### 3.7 Calibrated Score Fusion Engine (`backend/ai/fusion.py`)
* **Base Prototype Weights:**
  - RawNet3: $w_1 = 0.50$
  - ECAPA-TDNN: $w_2 = 0.30$
  - Bispectral Analyzer: $w_3 = 0.20$
* **Dynamic Renormalization:**
  When a branch is unavailable (e.g., ECAPA unenrolled or model missing), weights are dynamically renormalized across active branches:
  $$w_i' = \frac{w_i}{\sum_{j \in \text{Active}} w_j}, \quad \text{Final Risk } R = \sum_{i \in \text{Active}} w_i' R_i$$
* **Confidence Calculation:**
  $$C = \left(0.40 \cdot \frac{T}{3.0} + 0.30 \cdot \frac{r_{\text{speech}}}{0.5} + 0.30 \cdot \sum_{i \in \text{Active}} w_i\right) \cdot P_{\text{VAD}}$$
* **Threshold Classification:**
  - **LOW_RISK** (Authentic Voice): $R < 0.40$ (Green UI)
  - **MEDIUM_RISK** (Suspicious Anomalies): $0.40 \le R < 0.70$ (Orange UI)
  - **HIGH_RISK** (Voice Clone Detected): $R \ge 0.70$ (Red UI + Alert Card)

---

## 4. Android UI & Live Display

The Android application (`MainActivity.kt`) renders the backend AI verdict in real-time during connected WebRTC calls:

1. **AI Protection Badge:** Displays status pill (`AUTHENTIC VOICE`, `SUSPICIOUS`, `CLONED VOICE DETECTED!`, `LISTENING`).
2. **Real Risk Score Display:** Large formatted percentage (e.g. `24.5%`) styled with dynamic color transitions (Green $\rightarrow$ Orange $\rightarrow$ Red).
3. **Linear Risk Meter:** Visual progress bar reflecting the exact real-time mathematical risk ratio.
4. **Confidence & Latency:** Real inference latency (e.g. `142ms`) and confidence percentage.
5. **Multi-Branch Neural Inspection Card:**
   - **Silero VAD:** Active speech probability percentage.
   - **RawNet3:** Sinc-CNN anti-spoofing synthetic risk.
   - **ECAPA-TDNN:** Speaker identification mismatch (or transparent `"Not Enrolled (Safe Skip)"`).
   - **Bispectral Analyzer:** Higher-order vocoder phase artifact risk.
6. **High-Risk Alert Banner:** Pops up immediately when $R \ge 0.70$ warning the user against disclosing passwords, OTPs, or financial transactions.

---

## 5. Running the Complete System

### Start Backend:
```bash
# In project root with .venv activated:
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

### Build & Deploy Android App:
```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew.bat assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

### Run Standalone Pipeline Verification:
```bash
python backend/test_ai_pipeline.py
python backend/test_e2e_streaming.py
```
