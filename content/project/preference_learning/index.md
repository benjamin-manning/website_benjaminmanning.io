---
title: "Self-supervised Preference Learning for Multimodal Foundation Models"
design:
  font_size: XS
Summary: With Akshata Tiwari, Jillian Ross, and Andrew Lo. <br>
 Under review.

coming_soon: true

abstract: |
  **Preliminary Abstract (subject to change).** Preference optimization for multimodal foundation models typically relies on external judgment, such as human annotations, model-based scoring, or task-specific supervision. We introduce structural-neighbor preference learning, a self-supervised method based on a simple principle: structurally similar inputs should receive similar descriptions. For each image in a data set of interest, we identify others with similar underlying patterns—for example, charts with similar trends and fluctuations—and collect the model’s descriptions of the image and its neighbors. We rank these candidate descriptions by how much they agree with descriptions of similar images and disagree with those of dissimilar images, selecting the highest- and lowest-ranked candidates as preferred and rejected responses. We apply this method to DPO on Qwen2-VL-7B and Gemma-3-4B, testing across financial price paths, electrocardiograms, and seismic traces. We find the preference signals our approach learns transfer zero-shot to held-out tasks: relative to the supervised finetuned baseline, the detection of large price moves increases by 31%, crash-risk prediction by 14%, electrocardiogram abnormality detection by 10%, and seismic event detection by 3.3%. Our method also outperforms image conditioned, self-rewarding, and external-judge preference baselines, and remains effective when similarity between inputs is determined from image-derived representations rather than the underlying signal. We also observe improvements on other types of images, including histograms, scatter plots, and natural images.

tags:
- Working Paper
date: "2025-05-01T00:00:00Z"
nolink: true
---
