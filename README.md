# Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context–Detail Interaction

---

## Our Paper

Accurate detection of clinically meaningful events in healthcare time-series data is crucial for reliable downstream analysis and decision support. However, most existing methods struggle to jointly localize event boundaries and classify event types; even detection transformer (DETR)-based approaches show limited performance when confronted with extremely sparse events typical of clinical recordings.  

To address these challenges, we propose a coarse-to-fine detection framework combining a global context explorer, a local detail inspector, and an adaptive gating module (AGM) that fuses multiple label perspectives. The AGM uses transformed labels—encoding event presence and temporal position—to improve learning on sparse events. This design acts as a switch that selectively activates detailed feature extraction only when an event is likely, thereby reducing noise and improving efficiency in sparse settings.  

We evaluate our framework on diverse healthcare datasets—including arrhythmia detection, emotion recognition, and human-activity monitoring—and demonstrate substantial performance gains over existing DETR-based models, with particularly strong improvements in sparse event detection. With precise and robust event detection, our framework enables interpretation and actionable insights in real-world clinical applications.

---

## Framework Summary

Figure&nbsp;1: Overview of the proposed framework. The input time-series data is first processed by a frozen foundation model (FM), followed by a feed-forward network (FFN). Global and local temporal features are then extracted via the global context explorer (GCE) and local detail inspector (LDI), respectively. The GCE and LDI outputs are fused through the adaptive gating module (AGM), which acts as a dynamic gate to integrate global and local information. The fused representation is subsequently fed into the transformer decoder to predict event types and their temporal boundaries.

![Model Framework](./fig/model.pdf)

---
