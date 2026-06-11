---
title: "Automated Social Science: Language Models as Scientist and Subjects"
Summary: With Kehang Zhu and John J. Horton. <br> 
 Reject and Resubmit at the **Quarterly Journal of Economics**. <br>
 Extended abstract at the **ACM Conference on Economics & Computation (EC '26)**.
# summary: asdf

abstract: |
  We present an approach for automatically generating and testing, in silico, social scientific hypotheses. This automation is made possible by recent advances in large language models (LLM), but the key feature of the approach is the use of structural causal models. Structural causal models provide a language to state hypotheses, a blueprint for constructing LLM-based agents, an experimental design, and a plan for data analysis. The fitted structural causal model becomes an object available for prediction or the planning of follow-on experiments. We demonstrate the approach with several scenarios: a negotiation, a bail hearing, a job interview, and an auction. In each case, causal relationships are both proposed and tested by the system, finding evidence for some and not others. We provide evidence that the insights from these simulations of social interactions are not available to the LLM purely through direct elicitation. When given its proposed structural causal model for each scenario, the LLM is good at predicting the signs of estimated effects, but it cannot reliably predict the magnitudes of those estimates. In the auction experiment, the in silico simulation results closely match the predictions of auction theory, but elicited predictions of the clearing prices from the LLM are inaccurate. However, the LLM's predictions are dramatically improved if the model can condition on the fitted structural causal model. In short, the LLM knows more than it can (immediately) tell.

tags:
- Working Paper
date: "2025-06-03T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: "https://www.nber.org/papers/w32381"
image:
  caption: Photo by rawpixel on Unsplash
  focal_point: Smart

links:
- name: "Press: Marginal Revolution"
  url: "https://marginalrevolution.com/marginalrevolution/2024/03/its-happening-economic-science-edition.html"
url_code: "https://github.com/KeHang-Zhu/lm-automated-social-science/"
url_pdf: "https://www.nber.org/papers/w32381"
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
---
