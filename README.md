# Veritas: Autonomous Real-Time Video Anonymization Engine

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-00599C?style=flat-square&logo=onnx&logoColor=white)

An end-to-end, privacy-first automated video redaction pipeline. **Veritas** leverages **YOLO** for spatial detection, **MobileSAM** for zero-shot promptable instance segmentation, and spatial tracking to anonymize moving human targets, faces, or objects without cloud dependencies.

![Veritas_Interface](assets/veritas_ssinter.png) 

---

## 📌 Demonstration & Visuals

> **Note:** Below are target visual references showcasing how the processing pipeline operates in real time.

| Step | Visual Output | Description |
| :--- | :---: | :--- |
| **1. Target Selection** | ![Bounding Box UI](assets/yolobb.png) | Bounding box detected via YOLO and highlighted in the interface for target tracking. |
| **3. Final Output** | ![Redacted Result](assets/out_veritas.gif) | Feathered Gaussian blur applied dynamically over the target across all moving frames. |

---

## 🚨 Problem Statement

Manual video anonymization for compliance (such as GDPR, HIPAA, or law enforcement records) requires frame-by-frame masking, which is labor-intensive and error-prone. Modern neural network approaches often suffer from two major limitations:

1. **Bounding Box Over-clipping:** Traditional rectangular bounding box blurs either leak facial features during rotation/motion blur or redact vast non-sensitive background areas.

2. **Compute Heavy Transformers:** Running zero-shot segmentation models like Segment Anything (SAM) on every video frame generates massive latency bottlenecks, especially in non-GPU or edge environments.

---

## 🛠️ The Solution

**Veritas** solves this by pairing low-overhead **spatial object detection** with **box-displacement mask caching**:

- **Precise Boundary Segmentation:** Uses MobileSAM to create exact pixel-level masks rather than rigid rectangles.

- **Smart Motion Caching:** Instead of running the heavy MobileSAM image encoder on every frame, the engine evaluates bounding box displacement ($\Delta x, \Delta y$). Full transformer re-segmentation only triggers when significant spatial motion occurs, falling back to fast OpenCV spatial tracking for intermediate frames.

- **Out-of-Bounds Padding:** Automatically applies a **15% expansion margin** to YOLO predictions prior to mask prompt generation, ensuring motion blur, chin contours, and fast-moving limbs remain completely obscured.

---

## 🏗️ System Architecture

```mermaid
graph TD
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef api fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef module fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef core fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    A[Video Input / Upload]

    subgraph API_Layer ["FastAPI Service (api.py)"]
        B["POST /preview"]
        C["POST /process"]
    end

    subgraph Perception_Module ["Perception Engine (perception.py)"]
        D["DetectorTracker (YOLO11 ONNX)<br/>- 640x640 Preprocessing<br/>- Vectorized Confidence Filter<br/>- NMS Deduplication<br/>- 15% Bounding Box Expansion"]
        E["SamRefiner (MobileSAM ONNX)<br/>- 1024x1024 Embedding Encoder<br/>- Displacement-Based Motion Caching<br/>- Box-Prompted Mask Decoder"]
    end

    subgraph Pipeline_Module ["Pipeline & Compositor Engine"]
        F["VideoPipeline (pipeline.py)<br/>- Frame Iteration Loop<br/>- Last-Mask Cache Management"]
        G["Compositor Engine (compositor.py)<br/>- Mask Dilation<br/>- Feathering (GaussianBlur)<br/>- Effect: Blur / Pixelate"]
    end

    H["FFmpeg Subprocess<br/>- Raw BGR Pipe Ingest<br/>- H.264 / AAC Encoding"]
    I[Anonymized Video Output MP4]

    A --> B
    A --> C
    B --> D
    C --> F
    F --> D
    F --> E
    D --> E
    E --> G
    G --> H
    H --> I

    class A input;
    class B,C api;
    class D,E module;
    class F,G,H core;
    class I output;
```

---

## 🔄 Processing Pipeline

```mermaid
flowchart TD
    classDef startEnd fill:#eceff1,stroke:#455a64,stroke-width:2px;
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef decision fill:#fff8e1,stroke:#ffa000,stroke-width:2px;
    classDef sam fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;

    Start([Frame Ingest from FFmpeg]) --> Det[YOLO Detection & Tracking]
    Det --> Pad[Apply 15% Bounding Box Expansion]
    
    Pad --> MotionCheck{Motion Check:<br/>Δpx > Threshold?}

    MotionCheck -- "YES (Significant Motion)" --> SAM_Enc[Run MobileSAM Encoder<br/>1024x1024 Embedding Pass]
    MotionCheck -- "NO (Static / Minor Shift)" --> ReuseCache[Reuse Cached Embedding<br/>from Last Frame]

    SAM_Enc --> SAM_Dec[Run MobileSAM Mask Decoder]
    ReuseCache --> SAM_Dec

    SAM_Dec --> Comp[Compositor Step]
    Comp --> Dilate[Dilate & Feather Mask Bounds]
    Dilate --> ApplyEffect[Apply Gaussian Blur or Pixelation]
    
    ApplyEffect --> PipeOut[Stream Frame to FFmpeg Stdin Pipe]
    PipeOut --> NextFrame{More Frames?}

    NextFrame -- Yes --> Start
    NextFrame -- No --> End([Final Video Saved & Returned])

    class Start,End startEnd;
    class Det,Pad,ReuseCache,Comp,Dilate,ApplyEffect,PipeOut process;
    class MotionCheck,NextFrame decision;
    class SAM_Enc,SAM_Dec sam;
```

1. **Frame Ingestion:** Input frames are scaled and processed by the YOLO ONNX runtime session.

2. **IoU Spatial Tracking:** Bounding boxes are tracked frame-by-frame using spatial Intersection over Union (IoU) matching.

3. **Displacement Check:** Bounding box coordinates are compared against the last SAM encoding state.

4. **Conditional Transformer Passes:** MobileSAM re-encodes full spatial embeddings only when motion exceeds threshold conditions.

5. **Compositing:** Masks are dilated, softly feathered, and composited back into the raw video array using OpenCV Gaussian operations.

---

## 📊 Performance & Metrics

The pipeline was benchmarked under strict CPU-only execution constraints to establish realistic performance baselines on non-GPU hardware:

| Benchmark Metric | Value |
| :--- | :--- |
| **Test Environment** | Pure CPU Local Execution |
| **Input Clip** | 89 frames (~3 seconds @ 30 FPS) |
| **Active Targets** | 2 Simultaneous Track IDs |
| **MobileSAM Refinements** | **13 Calls** (Reduced from 89 calls via motion caching) |
| **Processing Speed** | **~0.13 - 0.18 FPS** |
| **Total Processing Time** | **~486.9 - 665.9 seconds** |

> **Technical Trade-off Note:**  
> MobileSAM's Vision Transformer (ViT) encoder requires processing full $1024 \times 1024$ image arrays, taking ~35–50 seconds per pass on standard x86 CPU architectures. By implementing displacement-based caching, **Veritas reduced raw transformer execution calls by over 85%**, enabling functional edge anonymization without requiring CUDA GPUs.

---

## ⚡ Tech Stack

- **Backend Framework:** FastAPI, Uvicorn
- **Computer Vision:** OpenCV (`cv2`), NumPy, FFmpeg
- **Inference Engine:** ONNX Runtime (`onnxruntime`)
- **Machine Learning Models:** YOLO (Detection), MobileSAM (Segmentation)

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and accessible in your system PATH.

---

### Local Installation & Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/veritas.git
cd veritas
```

#### 2. Create and Activate a Virtual Environment

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Verify ONNX Model Weights

Ensure the required ONNX model files are placed in the root directory (or as configured in `perception.py`):

```text
yolo11n.onnx
sam_encoder.onnx
sam_decoder.onnx
```

#### 5. Run the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## ⚠️ Known Limitations

1. **CPU Speed Bottleneck:** Standard CPU execution limits real-time throughput ($<0.2\text{ FPS}$) due to transformer attention layers. It is best suited for offline asynchronous batch processing rather than live streaming without GPU acceleration.

2. **Track ID Drops on Hard Occlusions:** Simple spatial IoU tracking can drop track IDs if a target is fully obscured by an object or exits the frame completely before re-entering.

3. **Extreme Motion Lag:** If a person moves at extremely high speeds while `motion_threshold_px` is set aggressively high, the cached mask can briefly lag behind the actual boundary before triggering a re-segmentation pass.

---

## 🛣️ Future Scope

- [ ] **Hardware Acceleration:** Integrate ONNX Runtime `CUDAExecutionProvider` / TensorRT to boost execution speed from 0.18 FPS to 30+ FPS.
- [ ] **Model Quantization:** Implement dynamic INT8 quantization for the MobileSAM encoder to reduce memory footprint and CPU compute time by ~70%.
- [ ] **DeepSORT / ByteTrack Integration:** Replace basic IoU spatial tracking with Kalman filter-based tracking to handle long-term target occlusions seamlessly.
- [ ] **Point-Prompt Interface:** Allow UI users to manually click specific subjects to generate prompt point embeddings for custom mask tracking.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
