Listed directory fake-news-detection
Viewed README.md:1-150
Viewed README.md:151-197
Ran command: `git log -n 10 --oneline`
Listed directory notebooks
Listed directory ml
Listed directory models
Listed directory agents
Listed directory src
Listed directory app

Here is a comprehensive **4-Week Implementation Report** structured specifically for academic project reviews. It includes both a **formal written submission format** and a **spoken presentation script ("What to Say to the Teacher")** for each week.

---

# 🎓 Project Title: VERITAS
### *Multimodal Fake News Detection Using Neural Ensembles and Agentic AI Evidence Verification*

---

## ⏱️ 1-Minute Executive Summary (What to Say at the Start)

> *"Good morning/afternoon, Professor. Over the past 4 weeks, I have implemented **VERITAS**, an end-to-end multimodal fake news detection and agentic verification platform. 
> 
> Rather than relying on simple text classification, the system tackles fake news in two stages:
> 1. **Neural Multimodal Modeling**: Combining fine-tuned **BERT** for text, **Vision Transformer (ViT)** for images, a **Late-Fusion MLP** for cross-modality, **CLIP** for image-text semantic alignment, and a **calibrated ensemble**.
> 2. **Autonomous Agentic Verification**: A 4-agent retrieval pipeline that extracts checkable claims, searches dual real-world evidence paths (supporting vs. debunking), clusters syndicated wire reprints, and adjudicates verifiable verdicts (`TRUE`, `FALSE`, `MISLEADING`).
>
> Over 4 weeks, I took this from raw dataset engineering and baseline models to multimodal fusion, autonomous agent orchestration, and finally a full-stack Next.js and FastAPI research platform with automated testing."*

---

## 📅 4-Week Roadmap Overview

| Week | Focus Area | Core Modules Implemented | Key Metric / Deliverable |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Dataset Engineering & Unimodal Baselines** | Text cleaning, image sanitization, fine-tuned BERT & ViT | BERT: 88.40% Acc \| ViT: 80.20% Acc |
| **Week 2** | **Multimodal Fusion & Ensemble Calibration** | Late Fusion MLP (1536-d), CLIP alignment, soft voting, Temperature Scaling | Fusion: 91.80% Acc \| Ensemble: 93.60% Acc (ECE: 0.021) |
| **Week 3** | **4-Agent Autonomous Evidence Verification** | Claim extraction, dual-retrieval (Tier-1/Tier-2), stance NLI, wire clustering, SSRF defense | Reduced OOD false positives by >64% |
| **Week 4** | **Full-Stack Platform, Diagnostics & Testing** | FastAPI async backend, SSE live streaming, Next.js 16 UI, Pytest suite (22 tests) | Complete working web app + Docker deployment |

---

## 📌 Week-by-Week Implementation Breakdown

```
Week 1 ────────► Week 2 ────────────► Week 3 ────────────────► Week 4
Baselines        Multimodal Fusion    Autonomous Agents        Full-Stack UI
(BERT + ViT)     & Ensemble Stacking  (Evidence Retrieval)     & Integration
```

---

### 🔹 WEEK 1: Dataset Pipeline & Unimodal Deep Learning Baselines

#### 1. Implementation Goals & Tasks:
- Designed modular data ingestion pipelines supporting both textual news and visual assets.
- Built preprocessing and validation transformers:
  - **Text Preprocessing**: Tokenization (`WordPiece`), length truncation, lowercasing, and special token embeddings.
  - **Image Preprocessing**: Format validation with PIL, resizing to $224 \times 224$, ImageNet normalization, and zero-vector fallback for text-only articles.
- Implemented and benchmarked two separate baseline deep neural networks:
  - **Text Model**: `bert-base-uncased` (110M parameters) fine-tuned with categorical cross-entropy.
  - **Image Model**: Vision Transformer `google/vit-base-patch16-224` (86M parameters) using self-attention across image patches.
- Conducted exploratory benchmark experiments in Jupyter notebook ([multimodal_fake_news_detection_bert_vit_svm.ipynb](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/notebooks/multimodal_fake_news_detection_bert_vit_svm.ipynb)).

#### 2. Key Code Deliverables:
- [`ml/preprocessing/`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/preprocessing): Tokenizers and image safety transforms.
- [`ml/models/bert_classifier.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/models/bert_classifier.py): BERT feature extraction and classification head.
- [`ml/models/vit_classifier.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/models/vit_classifier.py): Vision Transformer backbone with linear projection.

#### 🗣️ What to Say to Your Teacher:
> *"In **Week 1**, I focused on the foundations: dataset preprocessing and unimodal baselines. Because fake news leverages both misleading text and altered imagery, I built a two-stream ingestion pipeline. For text, I fine-tuned `bert-base-uncased`, achieving **88.40% accuracy**. For visual detection, I implemented a Vision Transformer (`vit-base-patch16-224`) which achieved **80.20% accuracy**. This proved that unimodal models have blind spots and justified building a multimodal architecture."*

---

### 🔹 WEEK 2: Multimodal Fusion, Semantic Alignment & Calibrated Ensembles

#### 1. Implementation Goals & Tasks:
- **Multimodal Late Fusion Network**:
  - Extracted 768-dimensional pooled text embeddings from BERT and 768-dimensional class-token embeddings from ViT.
  - Concatenated them into a 1536-dimensional multimodal vector and fed it into a dropout-regularized Multi-Layer Perceptron (MLP) classification head.
- **Cross-Modal Semantic Alignment (CLIP)**:
  - Integrated `openai/clip-vit-base-patch32` to measure cosine similarity between the headline text and image.
  - Solved "out-of-context image reuse" (where a real image is paired with a false headline).
- **Neural Ensemble & Calibration**:
  - Implemented **Soft-Voting** and **Stacking Meta-Classifier** (Logistic Regression over logits).
  - Implemented **Temperature Scaling** to minimize Expected Calibration Error (ECE) from 0.088 down to **0.021**, ensuring model confidence scores match empirical accuracy.

#### 2. Key Code Deliverables:
- [`ml/models/multimodal_fusion.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/models/multimodal_fusion.py): 1536-d concatenation and fusion MLP.
- [`ml/models/alignment_model.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/models/alignment_model.py): CLIP cross-modal cosine alignment scorer.
- [`ml/models/ensemble.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/models/ensemble.py): Soft voting and stacking meta-learner.
- [`ml/calibration/`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/calibration): Temperature scaling and ECE computation.

#### 🗣️ What to Say to Your Teacher:
> *"In **Week 2**, I solved the cross-modal problem. First, I created a Late Fusion MLP network joining BERT and ViT embeddings into a 1536-dimensional space, boosting accuracy to **91.80%**. Second, I added CLIP cross-modal alignment to detect image-caption mismatch attacks. Finally, I built a stacking ensemble and applied Temperature Scaling calibration. This improved overall accuracy to **93.60%** and calibrated the prediction probabilities so the system doesn't output overconfident false predictions."*

---

### 🔹 WEEK 3: Autonomous 4-Agent Real-World Evidence Verification Layer

#### 1. Implementation Goals & Tasks:
- Recognized the fundamental limitation of standard ML models: *deep models only learn syntactic patterns; they cannot verify recent real-world facts without external evidence.*
- Implemented an **Agentic AI Verification DAG (Directed Acyclic Graph)** with 4 specialized agents:
  1. **Claim Extraction Agent**: Parses raw articles into atomic, verifiable factual claims with named entities (NER) and temporal/geographic anchors.
  2. **Dual-Query Retrieval Agent**: Simultaneously executes both *supporting* and *debunking/hoax* search paths across authoritative domain tiers (Tier-1: AP/Reuters/Gov; Tier-2: Snopes/PolitiFact) to prevent confirmation bias.
  3. **Evidence Analysis Agent**: Stance classification (`SUPPORTS`, `CONTRADICTS`, `MIXED`) and syndicated wire clustering to prevent 10 wire copies from falsely counting as 10 independent sources.
  4. **Adjudication Agent**: Deterministic rule synthesis producing final verdicts (`TRUE`, `FALSE`, `MISLEADING`, `UNVERIFIABLE`) with exact source URL citations.
- Built **Anti-SSRF Security Defense** to sanitize input URLs and block malicious private IP probing (RFC 1918 / loopback addresses).

#### 2. Key Code Deliverables:
- [`backend/app/agents/input_agent.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/agents/input_agent.py): Claim & entity extraction.
- [`backend/app/agents/retrieval_agent.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/agents/retrieval_agent.py): Dual-path retrieval & domain authority tiering.
- [`backend/app/agents/analysis_agent.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/agents/analysis_agent.py): Stance classification & wire-reprint clustering.
- [`backend/app/agents/adjudication_agent.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/agents/adjudication_agent.py): Rule-based deterministic verdict synthesizer.
- [`backend/app/security/`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/security): Anti-SSRF URL firewall.

#### 🗣️ What to Say to Your Teacher:
> *"In **Week 3**, I tackled the biggest weakness in fake news AI: deep networks can hallucinate or fail on breaking news. I engineered a 4-Agent verification system:
> - Agent 1 extracts core factual claims and named entities.
> - Agent 2 queries the live web using dual paths—one for supporting evidence and one specifically looking for debunks or hoaxes.
> - Agent 3 runs stance detection and clusters syndicated wire reprints so duplicate news articles aren't mistaken for independent corroboration.
> - Agent 4 adjudicates a transparent verdict with citations.
> This agentic layer reduced out-of-distribution false positives by over 64%."*

---

### 🔹 WEEK 4: Full-Stack Integration, Research Workbench & Validation

#### 1. Implementation Goals & Tasks:
- **Backend API & Streaming Engine**:
  - Built high-performance **FastAPI** backend with asynchronous request handlers.
  - Added Server-Sent Events (SSE) streaming so agent pipeline execution steps stream live to the browser in real time.
  - Implemented SQLModel database layer for logging predictions, model confidence scores, and historical audits.
- **Frontend Research Workbench**:
  - Developed a modern web interface using **Next.js 16**, **React 19**, and **Tailwind CSS**.
  - Integrated components:
    - Text/URL ingestion with SSRF protection.
    - Image drag-and-drop with live preview.
    - Individual model probability breakdown bars (BERT vs. ViT vs. Fusion vs. Ensemble).
    - Dynamic CLIP Alignment Gauge and Agentic Timeline with millisecond execution tracking.
    - Curated benchmark presets (*NASA Exoplanet*, *Antarctic Alien Pyramid*, *Lemon Cancer Cure*, *Central Bank Rates*).
- **Testing & Deployment**:
  - Authored and verified **22 Pytest automated tests** across ML inference and backend agents (`pytest ml/tests backend/tests`).
  - Containerized with Docker and `docker-compose`.

#### 2. Key Code Deliverables:
- [`backend/app/main.py`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/app/main.py): FastAPI application & SSE streaming endpoints.
- [`frontend/src/app/page.tsx`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/frontend/src/app/page.tsx): Research workbench dashboard.
- [`backend/tests/`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/backend/tests) & [`ml/tests/`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/ml/tests): Automated Pytest verification suite.
- [`docker-compose.yml`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/docker-compose.yml) & [`Dockerfile`](file:///c:/Users/Deekshith%20Gowda/Documents/GitHub/fake-news-detection/Dockerfile): Multi-container deployment.

#### 🗣️ What to Say to Your Teacher:
> *"In **Week 4**, I integrated everything into a complete, deployable software platform. I built an asynchronous FastAPI backend supporting real-time Server-Sent Events (SSE) streaming, connected to a Next.js 16 research workbench. The UI provides side-by-side transparency: model confidence bars, CLIP alignment meters, step-by-step agent execution timelines, and cited source cards. I also wrote 22 automated integration tests to validate pipeline reliability and containerized the system with Docker."*

---

## 💡 Quick Teacher Q&A Cheat Sheet (Common Viva Questions)

**Q1: Why did you use both ML models and Agents? Isn't BERT enough?**
> *"BERT is great at linguistic style detection (sensationalism, clickbait tone), but it cannot check external factual truth. A fake story written in professional journalistic tone will fool BERT. The 4-agent layer solves this by actively fetching and cross-referencing real-world evidence from authoritative sources like Reuters, AP, and Snopes."*

**Q2: What is the purpose of CLIP in your project?**
> *"CLIP measures semantic alignment between the headline and the image. If an article about a natural disaster uses an unrelated dramatic explosion photo from a movie, CLIP flags low cross-modal cosine similarity, detecting out-of-context imagery."*

**Q3: How do you handle text-only articles if the model expects an image?**
> *"I implemented a zero-vector fallback in the ViT feature pipeline. If no image is provided, the vision embedding is zero-padded without corrupting the BERT text representations, allowing the model to smoothly handle text-only inputs."*

**Q4: How did you test and evaluate the system?**
> *"I evaluated 5 progressive ablation configurations (Text-only 88.4%, Image-only 80.2%, Multimodal Fusion 91.8%, and Ensemble 93.6%). In addition, the codebase has 22 automated Pytest unit and integration tests covering preprocessing, model shapes, agent responses, and API endpoints."*

Viewed api_schemas.py:21-50