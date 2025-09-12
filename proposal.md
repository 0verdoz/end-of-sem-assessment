# Fake News Detection for Social Media: Towards an Adaptable

# Model for Ghana

```

# Abstract

The spread of fake news on social media platforms threatens social cohesion, electoral
integrity, and public health in Ghana. Due to resource constraints, this project lever-
ages publicly available English-language datasets (LIAR, FakeNewsNet) to develop a
transformer-based natural language processing (NLP) model for fake news detection,
with the goal of creating a prototype adaptable to the Ghanaian context. Using BERT
and potentially AfriBERTa, the model will be fine-tuned and evaluated using accuracy,
F1-score, and ROC-AUC. Expected outcomes include a model achieving an F1-score
greater-than (>) 0.8 and insights into its adaptability for Ghanaian social media. This
work aims to support media literacy and misinformation resilience in Ghana, with a
reproducible pipeline shared via GitHub.

# 1 Introduction

The rapid rise of social media in Ghana has made platforms like Twitter (X), Face-
book, and WhatsApp critical channels for news dissemination. However, these platforms
also facilitate the spread of fake news, intentionally false or misleading information that
impacts events like elections and public health campaigns (example., COVID-19 misin-
formation). Unlike Western countries with advanced fake news detection tools, Ghana
lacks automated systems tailored to its cultural and linguistic context, particularly for
low-resource languages like Twi.
This project develops a transformer-based NLP model for fake news detection using
English language datasets, with the aim of creating a prototype adaptable to Ghanaian
social media. Due to restrictions on data mining (example, bans on Facebook data col-
lection), the project relies on open datasets like LIAR and FakeNewsNet. The objectives
are to:

- Develop a transformer-based model to classify fake versus real news.
- Evaluate model performance and explore its adaptability to Ghanaian contexts.
- Provide a reproducible pipeline for future adaptation to Ghana-specific data.

The research question is: How effectively can a transformer-based NLP model, trained
on English datasets, detect fake news, and how can it be adapted for Ghanaian social me-
dia? This work is significant for laying the groundwork for context-aware misinformation
detection in Ghana, enhancing digital media literacy.

# 2 Literature Review

Fake news detection has evolved from traditional feature-based approaches to advanced
deep learning methods. Early work by C. Castillo (1) used linguistic and user-based
features for credibility assessment on Twitter, achieving moderate success. Transformer
models like BERT (2) have since set benchmarks on datasets such as LIAR (3) and
FakeNewsNet (4), which provide labeled English news samples. Multilingual models like
mBERT and XLM-R (5) support low-resource languages, offering potential for African
contexts.


In Africa, NLP research is growing, with initiatives like Masakhane (6) advancing
models for languages like Twi. AfriBERTa (7) demonstrates effective text classification
for African languages with limited data. However, fake news detection in Ghana remains
underexplored. Local fact-checking organizations like GhanaFact and Dubawa rely on
manual verification, underscoring the need for automated tools (8). This project ad-
dresses this gap by developing a model trained on English datasets, with plans to explore
adaptation for Ghanaian contexts in future work.

# 3 Proposed Methodology

## 3.1 Data Sources

Due to constraints on social media data mining (no access to Facebook data and limited
Twitter/X API access), the project relies on:

- Public Datasets: LIAR (3) (political statements with veracity labels) and Fake-
    NewsNet (4) (news articles with social context).
- Justification: These datasets are publicly available, ethically sourced, and widely
    used in NLP research. While not Ghana-specific, they provide a robust foundation
    for training, with future adaptation planned for Ghanaian data when accessible.

## 3.2 Data Preprocessing

- Clean text by removing URLs, punctuation, and non-standard characters.
- Tokenize using HuggingFace’s tokenizer for BERT-based models.
- Create balanced train/test splits (80/20) with stratified sampling to maintain class
    distribution.

## 3.3 Models and Techniques

- Baseline: Logistic regression with TF-IDF features to establish a benchmark.
- Main Model: Fine-tune:
    - BERT (2) for English text classification.
    - AfriBERTa (7) to explore potential adaptation for African linguistic patterns.
- Use HuggingFace Transformers with PyTorch for training.
- Apply transfer learning: Pre-train on LIAR/FakeNewsNet, with plans to fine-tune
    on Ghana-specific data in future iterations.


## 3.4 Evaluation Metrics

- Primary Metrics: Accuracy, Precision, Recall, F1-score.
- Additional Metrics: Confusion matrix for error analysis; ROC-AUC for probabilistic
    outputs.
- Analysis: Assess model generalizability and discuss adaptation strategies for Ghana-
    ian contexts.

## 3.5 Tools and Resources

- Software: Python (NumPy, pandas, scikit-learn, matplotlib), HuggingFace Trans-
    formers, PyTorch.
- Hardware: Google Colab (free tier with GPU) for model training.
- Data Storage: Use cloud storage (Google Drive) and Git/GitHub for versioning.
- Contingency: If Colab resources are limited, switch to a personal laptop (16GB
    RAM, standard CPU).
- Declaration: I accept full responsibility for my machine and have contingency plans
    to protect against hardware/software failure.

# 4 Expected Results and Significance

The project aims to produce:

- A transformer-based model with an F1-score > 0.8 on English fake news datasets.
- A reproducible pipeline (via GitHub) for future adaptation to Ghana-specific data.
- Insights into model generalizability and strategies for local context adaptation.

```

Significance:

```
- Academic: Contributes to NLP research by testing transformer models in a new
    application context.
- Societal: Lays the foundation for combating misinformation in Ghana, supporting
    fact-checkers and policymakers.
- Practical: Provides a prototype for future deployment in Ghana’s digital ecosystem,
    enhancing media literacy.

# 5 Work Plan and Timeline

A one-week buffer is included for unexpected challenges.


```

Weeks Milestone
Weeks 1 Conduct literature review; download and explore LIAR, FakeNews-
Net datasets. Preprocess data; implement baseline model and initial
BERT/AfriBERTa training.
Week 2 Fine-tune model; conduct preliminary experiments. Evaluate performance;
analyze results and adaptation potential.
Week 3 Draft final report; prepare GitHub repository with code and documentation.

```

```

Table 1: Project Timeline

```
# 6 Ethical Considerations

The project adheres to ethical standards:

- Data Privacy: Uses only publicly available, ethically sourced datasets (LIAR, Fak-
    eNewsNet) compliant with data usage policies.
- Ethics Approval: If future Ghana-specific data collection is pursued, an Ethics
    Committee request will be submitted, detailing privacy safeguards.
- Transparency: Code and methodology will be shared via GitHub for reproducibility
    and scrutiny.
- Bias Mitigation: Analyze model outputs for potential biases (e.g., cultural or lin-
    guistic) during evaluation.

# 7 References

# References

```

[1] C. Castillo, M. Mendoza, and B. Poblete, “Information credibility on Twitter,” in
Proceedings of the 20th International Conference on World Wide Web, 2011, pp.
675–684.

```

```

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep
bidirectional transformers for language understanding,” in Proceedings of NAACL-
HLT, 2019, pp. 4171–4186.

```

```

[3] W. Y. Wang, ““Liar, Liar Pants on Fire”: A new benchmark dataset for fake news
detection,” in Proceedings of the 55th Annual Meeting of the ACL, 2017, pp. 422– 426.

```

```

[4] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, “FakeNewsNet: A data repository
with news content, social context, and spatiotemporal information for studying fake
news on social media,” arXiv preprint arXiv:1809.01286, 2017.

```

```

[5] A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in
Proceedings of the 58th Annual Meeting of the ACL, 2020, pp. 8440–8451.

```

[6] D. I. Adelani et al., “Masakhane: Building NLP for African languages,” in Findings
of the ACL, 2021.

[7] K. Ogueji, Y. Zhu, and J. Lin, “Small data? No problem! Exploring the viability
of pretrained multilingual language models for low-resourced languages,” in Pro-
ceedings of the 1st Workshop on Multilingual Representation Learning, 2021, pp.
210–218.

[8] GhanaWeb, “GhanaFact and Dubawa: The rise of fact-checking in Ghana,”
GhanaWeb, 2023. [Online]. Available: https://www.ghanaweb.com.


```
