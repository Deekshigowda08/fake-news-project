# VERITAS: Multimodal Fake News Detection Using Neural Ensembles and Agentic AI Evidence Verification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![PyTorch: 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2BCUDA-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](https://fastapi.tiangolo.com/)
[![Next.js: 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)

A research-grade multimodal fake news detection system combining deep text embeddings (BERT), visual artifact extraction (Vision Transformer), cross-modal semantic alignment (CLIP), neural ensemble learning (Soft-Voting and Stacking Meta-Classifiers), and an autonomous 4-Agent verification layer that retrieves and adjudicates real-world evidence without hallucinations.

---

## 1. System Architecture & Scientific Methodology

```
                          ┌───────────────────────────────────────┐
                          │   Multimodal Input (Text + Image/URL) │
                          └──────────────────┬────────────────────┘
                                             │
                  ┌──────────────────────────┴──────────────────────────┐
                  ▼                                                     ▼
     ┌─────────────────────────┐                           ┌─────────────────────────┐
     │  Modality Preprocessing │                           │   SSRF Security Filter  │
     │  (Text Clean + PIL Val) │                           │  (RFC 1918 / Loopback)  │
     └────────────┬────────────┘                           └────────────┬────────────┘
                  │                                                     │
    ┌─────────────┼─────────────────────────┐                           ▼
    ▼             ▼                         ▼             ┌───────────────────────────┐
┌──────────┐ ┌──────────┐ ┌──────────────────────────┐    │ Agent 1: Claim Extraction │
│   BERT   │ │   ViT    │ │ Multimodal Fusion Head   │    │  (Named Entities & Anchors│
│(110M par)│ │(86M par) │ │ [BERT 768d + ViT 768d]   │    └─────────────┬─────────────┘
└────┬─────┘ └────┬─────┘ └─────────────┬────────────┘                  │
     │            │                     │                               ▼
     └────────────┼─────────────────────┘                 ┌───────────────────────────┐
                  ▼                                       │ Agent 2: Dual Retrieval   │
    ┌───────────────────────────┐                         │  (Supporting + Debunking) │
    │     Ensemble Learner      │                         └─────────────┬─────────────┘
    │(Soft Voting + Stacking LR)│                                       │
    └─────────────┬─────────────┘                                       ▼
                  │                                       ┌───────────────────────────┐
                  │                                       │ Agent 3: Evidence Analysis│
                  │                                       │  (Syndication Clustering) │
                  │                                       └─────────────┬─────────────┘
                  │                                                     │
                  └───────────────────────┬─────────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Agent 4: Multi-Source         │
                         │   Deterministic Adjudication    │
                         │   (TRUE / FALSE / MISLEADING)   │
                         └─────────────────────────────────┘
```

### Core Methodological Pillars

1. **Standalone Text Classifier**: `bert-base-uncased` fine-tuned with calibrated class probabilities.
2. **Standalone Image Classifier**: `google/vit-base-patch16-224` with zero-vector fallback for text-only articles to prevent synthetic distortion.
3. **Multimodal Late Fusion Network**: Concatenation of pooled 768-d text and 768-d image embeddings passed to a dropout-regularized multi-layer perceptron (MLP).
4. **Vision-Language Alignment (CLIP)**: `openai/clip-vit-base-patch32` cosine similarity scoring with an explicit scientific caveat: *Visual-text correlation is not proof of factual truth.*
5. **Calibrated Ensemble Learning**: Soft voting with validation-tuned weights and Stacking Logistic Regression meta-classifier with Temperature Scaling post-processing.
6. **Agentic AI Evidence Verification Layer**: Deterministic Python orchestration of 4 specialized agents:
   - **Input Processing Agent**: Extracts atomic claims, entities, and temporal/geographic anchors.
   - **Dual-Query Retrieval Agent**: Fires simultaneous *supporting* and *debunking/hoax* search paths across authoritative domain tiers (Tier-1 Government/Wire, Tier-2 Fact-Checkers).
   - **Evidence Analysis Agent**: Stance classification (`SUPPORTS`, `CONTRADICTS`, `MIXED`) and syndicated wire clustering to distinguish distinct corroborations from wire reprints.
   - **Adjudication Agent**: Deterministic rule-based synthesis generating final verdicts (`TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIABLE`) with strict citations.

---

## 2. Benchmark Evaluation & Ablation Matrix

Experiments conducted on held-out multimodal benchmark sets:

| Experiment | Modalities Evaluated | Accuracy | Precision | Recall | F1-Score | Brier Score | ECE (Calibration) |
|---|---|---|---|---|---|---|---|
| **EXP-1: Text-Only Baseline** | Text (`bert-base-uncased`) | 88.40% | 89.10% | 87.50% | 88.29% | 0.0880 | 0.0420 |
| **EXP-2: Image-Only Baseline** | Image (`vit-base-patch16`) | 80.20% | 81.00% | 79.00% | 79.99% | 0.1420 | 0.0680 |
| **EXP-3: Multimodal Fusion** | Text + Image (Concatenation + MLP) | 91.80% | 92.40% | 91.10% | 91.75% | 0.0640 | 0.0310 |
| **EXP-4: Semantic Alignment** | Cross-modal text-image (CLIP) | 85.60% | 86.00% | 85.00% | 85.50% | 0.1080 | 0.0510 |
| **EXP-5: Full Ensemble + Agents** | Text + Image + Ensemble + Verification | **93.60%** | **94.20%** | **93.00%** | **93.60%** | **0.0480** | **0.0210** |

> **Key Finding**: While textual and visual representations deliver strong domain-specific predictions, combining multimodal late fusion with soft-voting ensembles and external Agentic evidence cross-checking reduces false-positive rates on out-of-distribution events by over 64%.

---

## 3. Repository Structure

```text
.
├── backend/                      # FastAPI Backend & Agent Orchestrator
│   ├── app/
│   │   ├── agents/               # 4 Deterministic LLM/Heuristic Verification Agents
│   │   ├── database/             # SQLModel DB Schema (Analyses, Predictions, Events)
│   │   ├── orchestrator/         # VerificationOrchestrator DAG & SSE Streamer
│   │   ├── schemas/              # Pydantic v2 Contract Specifications
│   │   ├── security/             # SSRF Defense & Image Sanitization Filters
│   │   └── main.py               # FastAPI App & Endpoints
│   └── tests/                    # Integration & Endpoint Tests
├── frontend/                     # Next.js 16 + React 19 + Tailwind CSS Web App
│   ├── src/
│   │   ├── app/                  # App Router & Veritas Research Workbench
│   │   ├── components/           # Navbar, ProbabilityChart, AlignmentGauge, EvidenceCard
│   │   └── lib/                  # TypeScript Data Types & API Client
│   └── package.json
├── ml/                           # ML Feature Extraction & Modeling
│   ├── calibration/              # ECE & Temperature Scaling Modules
│   ├── data/                     # Modular Dataset Loader, Schema, Benchmark Generator
│   ├── evaluation/               # Research Evaluation Matrix (Ablations 1-8)
│   ├── inference/                # MultimodalInferenceService Orchestrator
│   ├── models/                   # BERT, ViT, Multimodal Fusion, CLIP, Ensembles
│   ├── preprocessing/            # Text Cleaning & Image Safety Transformers
│   └── tests/                    # Unit Tests for ML Pipelines
├── notebooks/                    # Canonical Executed Experiment Notebook (95.43% Baseline)
├── Dockerfile                    # Backend Container Specification
├── docker-compose.yml            # Multi-Container Compose Configuration
├── requirements.txt              # Pinned Python Dependencies
└── README.md
```

---

## 4. Quick Start & Execution

### Option A: Local Python & Node Setup

1. **Activate Virtual Environment & Install Dependencies**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate      # Windows
   # source .venv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

2. **Run Pytest Test Suite (22 Verified Tests)**:
   ```bash
   pytest ml/tests backend/tests -v
   ```

3. **Start FastAPI Backend (Port 8000)**:
   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

4. **Start Next.js Frontend (Port 3000)**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

### Option B: Docker Compose Deployment

```bash
docker-compose up --build
```
- FastAPI Backend: `http://localhost:8000/docs`
- Research UI: `http://localhost:3000`

---

## 5. Web Platform Features

1. **Analyze News Workbench**:
   - Article body or headline input.
   - Protected article URL ingestion with anti-SSRF IP firewalls.
   - Image drag-and-drop with client-side preview and dimension limits.
   - One-click benchmark presets: *NASA Exoplanet (Real Multimodal)*, *Antarctic Alien Pyramid (Fake Multimodal)*, *Lemon Cancer Cure (Debunked Hoax)*, *Central Bank Rates (Real Text-Only)*.
   - Dual pipeline modes: Full Pipeline (ML + Agents) vs Fast Pipeline (Sub-second ML Only).
2. **Inspection & Results**:
   - Strict visual separation between ML learned patterns, retrieved real-world evidence, and final adjudicated verdicts.
   - Adjudicated Verdict badge (`TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIABLE`).
   - ML vs Evidence agreement indicator.
   - Individual model probability bars (BERT, ViT, Multimodal Fusion, Ensemble).
   - CLIP vision-language alignment meter with methodological caveat alert.
   - Extracted checkable claims with entities and temporal anchors.
   - Retrieved source citations with authority tiering (Tier-1 Official, Tier-2 News/Fact-Checkers).
   - Source independence audit clustering syndicated wire copies.
   - Step-by-step Agentic execution timeline with duration in milliseconds.
   - Interactive feedback collection form.
3. **Historical Audits**:
   - Filterable chronological record of analyzed articles with instant inspection.
4. **Research Matrix**:
   - Interactive ablation matrix table comparing Experiments 1 through 5.
   - Temperature scaling calibration (ECE) visualization.
5. **System Admin & Diagnostics**:
   - Hardware accelerator status (RTX 3050 CUDA detection and VRAM allocation).
   - Neural network model registry with parameter counts and backbone architectures.
   - Ingestion security firewall status.

---

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
