---
title: "Self-supervised Preference Learning for Multimodal Foundation Models"
design:
  font_size: XS
Summary: With Akshata Tiwari, Jillian Ross, and Andrew Lo. <br>
 Under Review at **NeurIPS**.

abstract: |
  Preference optimization for multimodal foundation models typically draws its signal from external judgment: human annotations, model-based scoring, or task-specific supervision. We propose a self-supervised alternative based on a simple principle: if two images depict similar underlying data, their descriptions should be similar. We instantiate this idea for time-series data rendered as charts—a setting that is simple, broadly important, and well-suited to clean downstream tasks. Two charts are neighbors if they trace similar patterns in the underlying data; we compute these neighborhoods directly from the data, generate candidate descriptions of each image and its neighbors using the model itself, and rank candidates by their agreement with the neighborhood to form preference pairs. We apply this method to DPO on two open-source foundation models, Qwen2-VL-7B and Gemma-3-4B, across three domains: financial price paths, electrocardiograms, and seismic traces. The learned signal transfers zero-shot to held-out tasks. Relative to the supervised fine-tuning baseline, the detection of large price-moves increases by 31%, crash-risk prediction by 14%, electrocardiogram abnormality detection by 10%, and seismic event detection by 3.3%. Probes show the gains reflect stronger alignment between visual inputs and textual outputs, and ablations confirm that neither generic preference optimization nor learned-feature neighborhoods produce the same gains.

tags:
- Working Paper
date: "2025-05-01T00:00:00Z"
nolink: true
---
