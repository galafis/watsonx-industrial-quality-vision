# Architecture - Industrial Quality Vision

## System Overview

Industrial Quality Vision implements a two-stage computer vision pipeline for automated defect detection and severity classification on manufacturing production lines. The system is designed for edge deployment with centralized governance through IBM Watsonx.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph EdgeDevice["Edge Device (Factory Floor)"]
        CAM[Industrial Camera\n2448x2048 @ 30fps]
        PRE[Image Preprocessor\nLetterbox + Normalize]
        DET[YOLOv8m Detector\nONNX Runtime]
        CLS[EfficientNet-B2\nONNX Runtime]
        ROI[ROI Extractor]

        CAM --> PRE
        PRE --> DET
        DET --> ROI
        ROI --> CLS
    end

    subgraph CloudServices["IBM Cloud"]
        GRN[Granite LLM\nNarrative Reports]
        GOV[Watsonx Governance\nModel Monitoring]
        DB[(Metrics Store)]

        GRN --> GOV
        GOV --> DB
    end

    subgraph Interface["User Interface"]
        API[FastAPI REST\nPort 8080]
        UI[Streamlit Dashboard\nPort 8501]
    end

    CLS --> API
    CLS --> GRN
    API --> UI
    GOV --> UI

    style EdgeDevice fill:#1a1a2e,stroke:#e94560,color:#fff
    style CloudServices fill:#054ADA,stroke:#fff,color:#fff
    style Interface fill:#0f3460,stroke:#e94560,color:#fff
```

## Detection Pipeline

### Stage 1: YOLOv8 Defect Detection

The detection stage uses YOLOv8m (medium) for real-time defect localization across five categories.

```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        IMG[Raw Image] --> LB[Letterbox\n640x640]
        LB --> NORM[Normalize\nImageNet Stats]
    end

    subgraph Backbone["YOLOv8m Backbone"]
        NORM --> CSP1[CSPDarknet\nStage 1-4]
        CSP1 --> SPPF[SPPF\nSpatial Pyramid]
    end

    subgraph Neck["Feature Pyramid"]
        SPPF --> FPN[FPN\nTop-Down]
        FPN --> PAN[PAN\nBottom-Up]
    end

    subgraph Head["Detection Head"]
        PAN --> DH1[P3 Head\nSmall Objects]
        PAN --> DH2[P4 Head\nMedium Objects]
        PAN --> DH3[P5 Head\nLarge Objects]
    end

    subgraph Post["Post-Processing"]
        DH1 --> NMS[NMS\nconf=0.5\niou=0.45]
        DH2 --> NMS
        DH3 --> NMS
        NMS --> OUT[Detections\nboxes + classes + conf]
    end

    style Backbone fill:#0f3460,stroke:#e94560,color:#fff
    style Neck fill:#16213e,stroke:#0f3460,color:#fff
    style Head fill:#1a1a2e,stroke:#e94560,color:#fff
```

**Defect Classes:**

| ID | Class | Description |
|---:|-------|-------------|
| 0 | scratch | Linear surface marks from tool contact |
| 1 | dent | Surface deformation from impact/pressure |
| 2 | crack | Structural fracture lines from stress |
| 3 | discoloration | Color anomalies from heat/chemical exposure |
| 4 | missing_part | Absent components from assembly error |

### Stage 2: EfficientNet-B2 Severity Classification

Each detected defect ROI is extracted, resized, and classified into severity levels.

```mermaid
flowchart LR
    subgraph Extract["ROI Extraction"]
        DET[Detection Box] --> CROP[Crop + Pad 10%]
        CROP --> RSZ[Resize\n224x224]
        RSZ --> NORM2[Normalize\nImageNet Stats]
    end

    subgraph Model["EfficientNet-B2"]
        NORM2 --> BB[Backbone\nMBConv Blocks]
        BB --> GAP[Global Avg\nPooling]
    end

    subgraph ClassHead["Classification Head"]
        GAP --> D1[Dropout 0.3]
        D1 --> FC1[FC 256]
        FC1 --> RELU[ReLU]
        RELU --> D2[Dropout 0.2]
        D2 --> FC2[FC 4]
        FC2 --> SM[Softmax]
    end

    subgraph Output["Output"]
        SM --> SEV[Severity Level\n+ Probability]
    end

    style Model fill:#0f3460,stroke:#e94560,color:#fff
    style ClassHead fill:#16213e,stroke:#0f3460,color:#fff
```

**Severity Levels:**

| ID | Level | Action | SLA |
|---:|-------|--------|-----|
| 0 | cosmetic | Log only | 30 days |
| 1 | minor | Schedule repair | 7 days |
| 2 | major | Stop and rework | 24 hours |
| 3 | critical | Halt production line | Immediate |

## Training Pipeline

```mermaid
flowchart TB
    subgraph DataPrep["Data Preparation"]
        RAW[Raw Images\nCOCO Annotations] --> AUG[Albumentations\nAugmentation]
        RAW --> SYN[Synthetic Defects\nCut-Paste]
        AUG --> LOADER[PyTorch\nDataLoader]
        SYN --> LOADER
    end

    subgraph Phase1["Phase 1: Transfer Learning"]
        LOADER --> FREEZE[Freeze Backbone]
        FREEZE --> HEAD[Train Head\nwith Warmup LR]
    end

    subgraph Phase2["Phase 2: Fine-Tuning"]
        HEAD --> UNFREEZE[Unfreeze Backbone]
        UNFREEZE --> FULL[Full Network\nCosine Annealing]
    end

    subgraph Monitor["Monitoring"]
        FULL --> ES[Early Stopping\npatience=15]
        FULL --> CKPT[Checkpoint\nBest Val Loss]
        ES --> BEST[Best Model]
        CKPT --> BEST
    end

    style Phase1 fill:#0f3460,stroke:#e94560,color:#fff
    style Phase2 fill:#1a1a2e,stroke:#e94560,color:#fff
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| LR scheduler | Cosine Annealing |
| Warmup epochs | 5 |
| Early stopping | 15 epochs patience |
| Optimizer | AdamW |

### Data Augmentation Pipeline

The augmentation pipeline simulates real-world factory imaging conditions:

| Augmentation | Purpose | Probability |
|-------------|---------|------------|
| Horizontal Flip | Orientation invariance | 0.5 |
| Vertical Flip | Orientation invariance | 0.2 |
| Rotation (15 deg) | Camera angle variation | 0.5 |
| Perspective | Lens distortion | 0.3 |
| Brightness/Contrast | Lighting changes | 0.6 |
| CLAHE | Surface detail enhancement | 0.3 |
| Gaussian Noise | Sensor noise | 0.4 |
| Motion Blur | Camera vibration | 0.3 |
| Random Shadow | Shadow casting | 0.3 |

## Edge Deployment

```mermaid
flowchart LR
    subgraph Export["Model Export"]
        PT_D[YOLOv8m.pt] --> ONNX_D[detector.onnx]
        PT_C[EfficientNet.pt] --> ONNX_C[classifier.onnx]
    end

    subgraph Runtime["ONNX Runtime"]
        ONNX_D --> ORT[ONNX Runtime\nCPU/GPU Provider]
        ONNX_C --> ORT
    end

    subgraph Optimization["Optimizations"]
        ORT --> DYN[Dynamic Batch\nAxes]
        ORT --> FOLD[Constant\nFolding]
        ORT --> QUANT[Optional\nQuantization]
    end

    style Export fill:#0f3460,stroke:#e94560,color:#fff
    style Runtime fill:#16213e,stroke:#0f3460,color:#fff
```

### ONNX Export Settings

| Setting | Value |
|---------|-------|
| Opset version | 17 |
| Dynamic axes | Enabled (batch dimension) |
| Constant folding | Enabled |
| Input validation | Enabled |
| Output comparison | PyTorch vs ONNX tolerance 1e-4 |

## Governance Architecture

```mermaid
flowchart TB
    subgraph Inference["Production Inference"]
        IMG2[Input Image] --> PIPE[Detection +\nClassification]
        PIPE --> RES[Results]
    end

    subgraph Monitoring["Watsonx Governance"]
        RES --> LOG[Metric Logger]
        LOG --> METRICS[mAP, Precision\nRecall, F1, Latency]
        METRICS --> WINDOW[Sliding Window\nn=100]
        WINDOW --> SHIFT{Degradation\n> 10%?}
        SHIFT -->|Yes| ALERT[Alert System]
        SHIFT -->|No| OK[Continue]
    end

    subgraph Reporting["Narrative Reports"]
        RES --> GRANITE[IBM Granite\n13B Chat]
        GRANITE --> REPORT[Inspection\nReport]
    end

    style Monitoring fill:#054ADA,stroke:#fff,color:#fff
    style Reporting fill:#533483,stroke:#e94560,color:#fff
```

### Governance Thresholds

| Metric | Threshold |
|--------|-----------|
| mAP | >= 0.85 |
| Precision | >= 0.80 |
| Recall | >= 0.80 |
| Shift window | 100 samples |
| Degradation alert | > 10% drop |

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Detection | YOLOv8m (Ultralytics) | Real-time defect localization |
| Classification | EfficientNet-B2 (timm) | Severity assessment |
| Edge Runtime | ONNX Runtime | Low-latency inference |
| Augmentation | Albumentations | Training data augmentation |
| Training | PyTorch + AdamW | Model training pipeline |
| Reports | IBM Granite | Natural language inspection reports |
| Governance | IBM Watsonx | Model performance tracking |
| API | FastAPI + Uvicorn | REST endpoints |
| UI | Streamlit | Interactive dashboard |
| Infrastructure | Docker Compose | Service orchestration |

## Infrastructure

```mermaid
flowchart LR
    subgraph Docker["Docker Compose"]
        API[FastAPI API\nPort 8080]
        STR[Streamlit UI\nPort 8501]
    end

    subgraph Storage["Data"]
        DATA[data/\nImages + Annotations]
        MODELS[models/\nWeights + ONNX]
        LOGS[logs/\nGovernance]
    end

    API --> DATA
    API --> MODELS
    API --> LOGS
    STR --> API

    style Docker fill:#0f3460,stroke:#e94560,color:#fff
```

### Container Architecture

| Service | Port | Base Image | Purpose |
|---------|------|-----------|---------|
| api | 8080 | python:3.12-slim | REST API for inference |
| ui | 8501 | python:3.12-slim | Streamlit dashboard |

## Module Structure

```
src/
├── config.py              # Dataclass configurations
│   ├── DetectorConfig     # YOLOv8 parameters
│   ├── ClassifierConfig   # EfficientNet parameters
│   ├── TrainingConfig     # Hyperparameters
│   ├── AugmentationConfig # Pipeline settings
│   ├── InferenceConfig    # ONNX export settings
│   ├── GovernanceConfig   # Monitoring thresholds
│   └── WatsonxSettings    # IBM Cloud credentials
├── data/
│   ├── dataset.py         # DefectDataset (COCO format)
│   ├── augmentation.py    # Albumentations pipelines
│   ├── preprocessing.py   # Image normalization/ROI
│   └── synthetic_defects.py # Cut-paste augmentation
├── models/
│   ├── detector.py        # DefectDetector (YOLOv8)
│   ├── classifier.py      # SeverityClassifierNet
│   ├── trainer.py         # ClassifierTrainer
│   └── export_onnx.py     # ONNX export utilities
├── governance/            # Watsonx governance integration
├── inference/             # ONNX inference engine
├── narrative/             # Granite report generation
├── api/                   # FastAPI endpoints
└── ui/                    # Streamlit dashboard
```
