# Watsonx Industrial Quality Vision

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![IBM Watsonx](https://img.shields.io/badge/IBM-Watsonx-054ADA?logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen?logo=githubactions&logoColor=white)]()

**[English](#english)** | **[Portugues](#portugues)**

---

<a name="english"></a>

## Overview

Industrial Quality Vision is a two-stage computer vision system for automated defect detection and severity classification on manufacturing production lines. The pipeline combines **YOLOv8** for real-time defect localization with **EfficientNet-B2** for fine-grained severity assessment, deployed at the edge via **ONNX Runtime** and governed through **IBM Watsonx**.

The system processes camera feeds from factory-floor inspection stations, identifies five defect categories across manufactured surfaces, classifies each defect into four severity levels, generates natural language inspection reports using IBM Granite, and tracks model performance through the Watsonx governance dashboard.

### Architecture

```mermaid
flowchart LR
    A[Camera Feed] --> B[Image Preprocessing]
    B --> C[YOLOv8 Defect Detection]
    C --> D{Defects Found?}
    D -->|Yes| E[ROI Extraction]
    D -->|No| K[Pass Report]
    E --> F[EfficientNet-B2\nSeverity Classification]
    F --> G[ONNX Edge\nDeployment]
    G --> H[Granite Narrative\nReport]
    H --> I[Watsonx Governance\nDashboard]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style F fill:#0f3460,stroke:#e94560,color:#fff
    style G fill:#16213e,stroke:#0f3460,color:#fff
    style H fill:#533483,stroke:#e94560,color:#fff
    style I fill:#054ADA,stroke:#fff,color:#fff
```

### Pipeline Detail

```mermaid
flowchart TB
    subgraph Stage1["Stage 1 - Detection"]
        A1[Input Image\n640x640] --> A2[YOLOv8m Backbone]
        A2 --> A3[Feature Pyramid\nNetwork]
        A3 --> A4[Detection Head]
        A4 --> A5[NMS Filtering\nconf=0.5 iou=0.45]
    end

    subgraph Stage2["Stage 2 - Classification"]
        B1[Crop Defect ROI] --> B2[Resize 224x224]
        B2 --> B3[EfficientNet-B2\nBackbone]
        B3 --> B4[Global Avg Pool]
        B4 --> B5[FC 256 → ReLU]
        B5 --> B6[FC 4 classes]
        B6 --> B7[Softmax]
    end

    A5 --> B1

    style Stage1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style Stage2 fill:#0f3460,stroke:#e94560,color:#fff
```

## Defect Classes

| Class | ID | Description | Typical Cause |
|-------|---:|-------------|---------------|
| `scratch` | 0 | Linear surface marks | Tool contact, handling |
| `dent` | 1 | Surface deformation | Impact, pressure |
| `crack` | 2 | Structural fracture lines | Stress, fatigue |
| `discoloration` | 3 | Color/surface anomalies | Heat, chemical exposure |
| `missing_part` | 4 | Absent component/feature | Assembly error |

## Severity Levels

| Level | ID | Action Required | SLA |
|-------|---:|-----------------|-----|
| `cosmetic` | 0 | Log only | 30 days |
| `minor` | 1 | Schedule repair | 7 days |
| `major` | 2 | Stop & rework | 24 hours |
| `critical` | 3 | Halt production line | Immediate |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8m (Ultralytics) |
| Classification | EfficientNet-B2 (timm) |
| Deep Learning | PyTorch 2.1+ |
| Edge Deployment | ONNX Runtime |
| Data Augmentation | Albumentations |
| Image Processing | OpenCV |
| Narrative Reports | IBM Granite via Watsonx |
| Governance | IBM Watsonx AI Governance |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Infrastructure | Docker, Docker Compose |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- IBM Watsonx credentials (for governance features)

### Installation

```bash
# Clone repository
git clone https://github.com/galafis/watsonx-industrial-quality-vision.git
cd watsonx-industrial-quality-vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing)
pip install -r requirements-dev.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your credentials:
# WATSONX_API_KEY=your_api_key
# WATSONX_PROJECT_ID=your_project_id
# WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

### Docker

```bash
# Build and start all services
docker-compose up -d

# API available at http://localhost:8080
# UI available at http://localhost:8501
```

## Project Structure

```
watsonx-industrial-quality-vision/
├── config/
│   └── settings.yaml              # Model and pipeline configuration
├── data/
│   ├── annotations/               # COCO-format annotation files
│   └── sample_images/             # Sample inspection images
├── docs/
│   └── architecture.md            # System architecture documentation
├── notebooks/
│   └── 01_quality_vision_demo.ipynb  # Interactive demo notebook
├── src/
│   ├── __init__.py
│   ├── config.py                  # Dataclass configurations
│   ├── api/                       # FastAPI REST endpoints
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentation.py        # Albumentations pipelines
│   │   ├── dataset.py             # PyTorch Dataset (COCO format)
│   │   ├── preprocessing.py       # Normalization, ROI, color-space
│   │   └── synthetic_defects.py   # Cut-paste augmentation
│   ├── governance/                # Watsonx governance integration
│   ├── inference/                 # ONNX Runtime inference engine
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py          # EfficientNet severity classifier
│   │   ├── detector.py            # YOLOv8 defect detector
│   │   ├── export_onnx.py         # ONNX export utilities
│   │   └── trainer.py             # Training pipeline
│   ├── narrative/                 # Granite report generation
│   └── ui/                        # Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── test_augmentation.py
│   ├── test_classifier.py
│   ├── test_detector.py
│   ├── test_export_onnx.py
│   └── test_preprocessing.py
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Detector architecture | YOLOv8m |
| Classifier architecture | EfficientNet-B2 |
| Detector input size | 640 x 640 |
| Classifier input size | 224 x 224 |
| Epochs | 100 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.0005 |
| Early stopping patience | 15 epochs |
| LR scheduler | Cosine Annealing + Warmup |
| Warmup epochs | 5 |
| ONNX opset version | 17 |

## Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="portugues"></a>

## Visao Geral

Industrial Quality Vision e um sistema de visao computacional em dois estagios para deteccao automatizada de defeitos e classificacao de severidade em linhas de producao industriais. O pipeline combina **YOLOv8** para localizacao de defeitos em tempo real com **EfficientNet-B2** para avaliacao de severidade, implantado na borda via **ONNX Runtime** e governado pelo **IBM Watsonx**.

O sistema processa feeds de camera de estacoes de inspecao no chao de fabrica, identifica cinco categorias de defeitos em superficies manufaturadas, classifica cada defeito em quatro niveis de severidade, gera relatorios de inspecao em linguagem natural usando IBM Granite e monitora a performance do modelo atraves do dashboard de governanca Watsonx.

### Arquitetura

```mermaid
flowchart LR
    A[Feed de Camera] --> B[Pre-processamento]
    B --> C[Deteccao YOLOv8]
    C --> D{Defeitos?}
    D -->|Sim| E[Extracao ROI]
    D -->|Nao| K[Relatorio OK]
    E --> F[Classificacao\nEfficientNet-B2]
    F --> G[Deploy ONNX\nna Borda]
    G --> H[Relatorio\nGranite]
    H --> I[Dashboard\nWatsonx]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style F fill:#0f3460,stroke:#e94560,color:#fff
    style G fill:#16213e,stroke:#0f3460,color:#fff
    style H fill:#533483,stroke:#e94560,color:#fff
    style I fill:#054ADA,stroke:#fff,color:#fff
```

## Classes de Defeitos

| Classe | ID | Descricao | Causa Tipica |
|--------|---:|-----------|--------------|
| `scratch` | 0 | Marcas lineares na superficie | Contato com ferramentas |
| `dent` | 1 | Deformacao na superficie | Impacto, pressao |
| `crack` | 2 | Linhas de fratura estrutural | Estresse, fadiga |
| `discoloration` | 3 | Anomalias de cor/superficie | Calor, exposicao quimica |
| `missing_part` | 4 | Componente ausente | Erro de montagem |

## Niveis de Severidade

| Nivel | ID | Acao Requerida | SLA |
|-------|---:|----------------|-----|
| `cosmetic` | 0 | Apenas registrar | 30 dias |
| `minor` | 1 | Agendar reparo | 7 dias |
| `major` | 2 | Parar e retrabalhar | 24 horas |
| `critical` | 3 | Parar linha de producao | Imediato |

## Inicio Rapido

```bash
# Clonar repositorio
git clone https://github.com/galafis/watsonx-industrial-quality-vision.git
cd watsonx-industrial-quality-vision

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Executar testes
make test

# Iniciar com Docker
docker-compose up -d
```

## Estrutura do Projeto

```
watsonx-industrial-quality-vision/
├── config/settings.yaml           # Configuracao de modelos e pipeline
├── data/                          # Dados de anotacoes e imagens
├── docs/architecture.md           # Documentacao de arquitetura
├── notebooks/                     # Notebooks de demonstracao
├── src/
│   ├── config.py                  # Configuracoes (dataclasses)
│   ├── data/                      # Dataset, augmentacao, pre-processamento
│   ├── models/                    # Detector YOLOv8, Classificador EfficientNet
│   ├── governance/                # Integracao Watsonx Governance
│   ├── inference/                 # Engine de inferencia ONNX
│   ├── narrative/                 # Geracao de relatorios Granite
│   └── ui/                        # Dashboard Streamlit
├── tests/                         # Testes unitarios
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

## Licenca

Este projeto esta licenciado sob a Licenca MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
