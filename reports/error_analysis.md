# Error & Limitations Analysis

Fake News Detection Pipeline (Logistic Regression + TF‑IDF Baseline, AfriBERTa Prototype)

## 1. Purpose of This Document

This report:

- Analyzes model performance (quantitative + qualitative)
- Explains failure cases, instability observed in tests, and root causes
- Details dataset, pipeline, and methodological limitations
- Assesses bias / ethical risks for adaptation to Ghana / broader African contexts
- Recommends prioritized improvements (engineering + ML)

---

## 2. Datasets & Representations

| Aspect              | Current State                                                                | Risks / Limitations                                                                                 |
| ------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Primary Dataset     | English-only (e.g., amalgam of fact-checked claims/news)                     | Limited socio-linguistic coverage for Ghana (code-switching, dialect, Pidgin, Akan, Ewe, Ga, Hausa) |
| Label Schema        | Binary: Fake vs Real (assumed)                                               | Real-world misinformation includes satire, partially true, misleading framing                       |
| Preprocessing       | Lowercasing, basic cleaning, stopword removal, stemming (if applied), TF‑IDF | Potential loss of semantic nuance (negation, stance, sarcasm)                                       |
| Splits              | Single random split                                                          | No stratified multi-seed cross-validation → fragile estimates                                       |
| Class Distribution  | Appears near-balanced (inferred from confusion matrix)                       | Real deployments often imbalanced → metrics may degrade                                             |
| Data Quality Issues | CSV parsing errors (ParserError: variable columns)                           | Inconsistent quoting / commas inside text fields → preprocessing fragility                          |
| Domain Drift        | Source likely U.S./international political claims                            | Ghana-specific entities, local idioms not represented                                               |

---

## 3. Models Compared

| Model                                   | Text Representation                                     | Approx Params                     | Training Time (relative) | Inference Speed | Strengths                                       | Weaknesses                                 |
| --------------------------------------- | ------------------------------------------------------- | --------------------------------- | ------------------------ | --------------- | ----------------------------------------------- | ------------------------------------------ |
| Logistic Regression                     | TF‑IDF (word, likely unigrams)                          | ~ (features) × 2 (weights + bias) | Very Fast (seconds)      | Very Fast       | Interpretable weights; robust on sparse signals | Misses context, long dependencies, sarcasm |
| Logistic Regression (Potential Upgrade) | TF‑IDF + char 3–5 n‑grams                               | Larger                            | Slightly slower          | Fast            | Better on OOV, misspellings, dialect drift      | Still bag-of-features                      |
| AfriBERTa (Base)                        | Subword embeddings + transformer layers                 | ~110M                             | Much Slower              | Moderate        | Captures morphology, multilingual adaptability  | Risk of overfitting small dataset; heavier |
| Hybrid (Future)                         | TF‑IDF lexical + Transformer CLS embedding concatenated | TF‑IDF + transformer              | Medium                   | Moderate        | Complements lexical + contextual                | Added complexity                           |

---

## 4. Quantitative Results (Observed & Prototype)

Baseline confusion matrix (as printed by evaluation script):

Confusion Matrix (assumed ordering [[TN, FP], [FN, TP]]):
[[2004, 73],
 [  14, 2069]]

| Metric                     | Logistic Regression (Observed)   | AfriBERTa (Prototype Example\*) | Notes                            |
| -------------------------- | -------------------------------- | ------------------------------- | -------------------------------- |
| Accuracy                   | 0.9791                           | 0.985 (hypothetical)            | Transformer modest lift expected |
| Precision                  | 0.9659                           | 0.978                           | Fewer false positives            |
| Recall                     | 0.9933                           | 0.992                           | Baseline already very high       |
| F1 Score                   | 0.9794                           | 0.985                           | Incremental improvement          |
| False Positive Rate (FPR)  | 0.0351                           | ~0.026                          | 73 / (2004+73)                   |
| False Negative Rate (FNR)  | 0.0067                           | ~0.008                          | 14 / (14+2069)                   |
| Balanced / Macro F1        | ~≈ Micro (classes near-balanced) | Slightly ↑                      | Confirm via per-class metrics    |
| Inference Latency / sample | <2 ms                            | 20–40 ms (CPU)                  | Dependent on hardware            |

\* AfriBERTa numbers are illustrative; actual experiments should be documented with seeds, hardware, and exact training config. DO NOT treat as final without reproducible logs.

---

## 5. Statistical Reliability

Total samples: 4160  
Accuracy = 4073 / 4160 = 0.97909  
Approximate 95% Confidence Interval (Wilson): 0.9747 – 0.9834  
Implication: Gains <0.5% absolute must be validated across multiple random seeds / folds to rule out variance.

Recommended: Perform 5× stratified CV → report mean ± std for F1, precision, recall.

---

## 6. Interpretation of Confusion Matrix

| Type                 | Count | Share | Observed Pattern (Likely)                                     |
| -------------------- | ----- | ----- | ------------------------------------------------------------- |
| True Negatives (TN)  | 2004  | 48.1% | Correctly filtered real news                                  |
| False Positives (FP) | 73    | 1.75% | Possibly sensational but factual headlines; satire misflagged |
| False Negatives (FN) | 14    | 0.34% | Subtle fabrication; plausible linguistic style                |
| True Positives (TP)  | 2069  | 49.7% | Fake items with strong lexical markers                        |

Key Imbalance in Errors: Model is stricter on recall (low FN) but tolerates some FP inflation. This may be acceptable for safety-first moderation but raises risk of wrongly suppressing legitimate content (precision trade-off).

---

## 7. Error Taxonomy (Qualitative Hypotheses)

| Error Class                       | Description                                        | Example Pattern (Hypothetical)            | Remediation                                              |
| --------------------------------- | -------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| Satire Misclassification (FP)     | Satirical sources treated as fake                  | Exaggerated political parody              | Add satire label or domain whitelist                     |
| Emerging Event Uncertainty (FP)   | Real breaking news flagged (lack of corroboration) | Early crisis reports w/ limited context   | Incorporate temporal external knowledge                  |
| Subtle Fabrication (FN)           | Minimal lexical signals; fabricated statistics     | Neutral tone + invented numbers           | Add numeric pattern / hallucination detector             |
| Quoted Speech Ambiguity (FN / FP) | Claims within quotes misparsed                     | "Minister said X" w/out stance resolution | Add stance & source credibility features                 |
| Entity Novelty (FN)               | New local names, Ghana-centric actors              | New politician, local slang               | Fine-tune with regional corpora                          |
| Code-switch / Pidgin (FN)         | Mixed English + local dialect                      | “Gov dey plan…”                           | Multilingual / dialect-aware model (AfriBERTa fine-tune) |
| Adversarial Paraphrase (FN)       | Rewording to avoid trigger words                   | Soft hedging verbs                        | Data augmentation (back-translation, paraphrase sets)    |
| Clickbait Headlines (FP)          | Legit headlines with sensational framing           | “You won’t believe…”                      | Add document body context (beyond headline)              |

---

## 8. Representation Limitations

1. Bag-of-Words TF‑IDF:
   - Ignores order, sarcasm, negation (“not confirmed” vs “confirmed”).
   - High-dimensional sparse vectors → risk of spurious correlations.
2. Stemming / Stopword Removal:
   - Removes function words important for factual nuance (“not”, “no longer”).
3. Lack of Entity Normalization:
   - Misses cross-document consistency signals.
4. Lack of Temporal Modeling:
   - Truth value of claims can change over time (dynamic knowledge).

---

## 9. Model-Specific Limitations

| Model                 | Limitation                                                   | Impact                                |
| --------------------- | ------------------------------------------------------------ | ------------------------------------- |
| Logistic Regression   | Linear decision boundary                                     | Complex semantic deception missed     |
| Logistic Regression   | Feature leakage risk (source names)                          | Overestimates real-world performance  |
| AfriBERTa (Prototype) | Overfitting small dataset (if few epochs w/out reg)          | Inflated validation metrics           |
| AfriBERTa             | Subword vocab may underrepresent Ghanaian code-switch tokens | Token fragmentation → degraded recall |
| Both                  | Binary label oversimplification                              | Loss of nuanced interventions         |

---

## 10. Engineering & Pipeline Issues Observed

| Issue                              | Evidence from Logs                                          | Root Cause                                                                                             | Fix                                                                                        |
| ---------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Intermittent F1 = 0.0              | Early failing test: "F1 score too low: 0.0"                 | metrics.json possibly unwritten or keys mismatch (expected `f1_macro`, script outputs `f1_score`)      | Align metric keys; assert file atomic write                                                |
| Missing vectorizer.joblib          | FileNotFoundError in eval                                   | Training script may not save vectorizer or path mismatch; eval hard-coded `./models/vectorizer.joblib` | Pass `--model_dir`; ensure consistent path join                                            |
| ParserError (CSV)                  | `Expected 3 fields... saw 4`                                | Unescaped commas/quotes in text                                                                        | Use `pd.read_csv(..., quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")` or proper delimiter |
| ModuleNotFoundError (`project`)    | Running `python project/eval.py`                            | Direct script path vs module; sys.path not set                                                         | Use `python -m project.eval` or add package init                                           |
| Non-deterministic pass/fail timing | Pipeline test sometimes passes after multiple runs          | Race or residual state (cached models, partial writes)                                                 | Clean directories robustly; add `flush + fsync`                                            |
| Unregistered pytest marker warning | `PytestUnknownMarkWarning`                                  | Marker not declared in `pytest.ini`                                                                    | Add `[pytest] markers = pipeline: ...`                                                     |
| Hard-coded paths                   | `./models/...`                                              | Limited portability                                                                                    | Use argparse + Pathlib                                                                     |
| Conflicting CLI flags              | Provided `--data_dir` with no value in failing test variant | Arg parsing likely not validating                                                                      | Use `argparse` required arguments & tests                                                  |

---

## 11. Bias & Ethical Considerations

| Risk                       | Description                                        | Mitigation                                        |
| -------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| Cultural Misclassification | Model learned Western political discourse patterns | Curate Ghanaian corpora; human-in-loop            |
| Overblocking Real Content  | High recall bias may silence valid dissent         | Calibrate threshold for platform policy           |
| Satire & Humor Mislabeling | Cultural satire penalized                          | Multi-class expansion (satire class)              |
| Language Exclusion         | Non-English misinformation undetected              | Multilingual fine-tuning (AfriBERTa, MasakhaNEWS) |
| Source Familiarity Bias    | Overtrusting well-known outlets                    | Source-agnostic scoring + cross-checking content  |
| Model Drift                | Elections / crises change discourse patterns       | Scheduled re-training; drift detectors            |

---

## 12. Root Cause Analysis of Key Failures

| Symptom                            | Most Probable Root Cause                                        | Verification Step                                                                                          |
| ---------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| F1 = 0.0 in test                   | metrics key mismatch or empty predictions                       | Inspect metrics.json content                                                                               |
| Missing vectorizer.joblib          | Train run skipped saving vectorizer or path cleaned before eval | Confirm train script: does it call `joblib.dump(vectorizer, os.path.join(model_dir,"vectorizer.joblib"))`? |
| ParserError                        | Non-escaped commas inside quoted text                           | Try `pd.read_csv(..., engine="python")` with quoting                                                       |
| Intermittent passing after retries | Timing / dependency on residual artifacts                       | Run inside isolated temp dir                                                                               |
| ModuleNotFoundError                | Packaged incorrectly                                            | Ensure `__init__.py` in `project/`                                                                         |

---

## 13. Prioritized Recommendations

### Immediate (High Impact / Low Effort)

1. Standardize metric names (`f1_score` vs `f1_macro`) & write all metrics (micro, macro, per-class).
2. Add `project/__init__.py` and robust argparse to `train.py` / `eval.py` with explicit `--data_dir`, `--model_dir`, `--reports_dir`.
3. Save vectorizer path deterministically; fail fast if missing.
4. Add `pytest.ini`:
   ```
   [pytest]
   markers =
       pipeline: end-to-end training + evaluation
   ```
5. Wrap CSV load with defensive parsing:
   ```python
   pd.read_csv(path, on_bad_lines="skip", quoting=1)
   ```
6. Set all random seeds (`numpy`, `random`, `sklearn`, `transformers`) for reproducibility.
7. Log dataset size & class distribution at train time.

### Near-Term

1. Implement k-fold stratified CV with aggregated metrics.
2. Add sample-level error log (store misclassified examples).
3. Add calibration (Platt scaling or isotonic) for probability thresholds.
4. Introduce char n-grams to reduce OOV false negatives.
5. Add basic language / dialect detector to flag out-of-domain inputs.

### Medium-Term

1. Fine-tune AfriBERTa on combined English + Ghanaian code-switch corpora.
2. Multi-class expansion: {Real, Fake, Satire, Unverified}.
3. Incorporate external knowledge signals (fact-check DB fuzzy match).
4. Deploy drift monitoring (KL divergence on embedding distributions).

### Long-Term

1. Semi-supervised adaptation: self-training on unlabeled Ghanaian news.
2. Stance detection auxiliary head (improves factual context modeling).
3. Claim decomposition + evidence retrieval module (RAG architecture).
4. Human feedback interface for continuous improvement.

---

## 14. Suggested Metric Enhancements

Add per-class metrics:

| Class | Precision                                  | Recall | F1     | Support |
| ----- | ------------------------------------------ | ------ | ------ | ------- |
| Real  | (TN / (TN+FN?)\* depends on label mapping) | ...    | ...    | 2077    |
| Fake  | (2069 / (2069+73)) = 0.9659                | 0.9933 | 0.9794 | 2083    |

Clarify label ordering: store explicit `labels = ["real","fake"]` in metrics JSON to avoid ambiguity.

---

## 15. Example Improved Metrics JSON Schema

```json
{
  "schema_version": "1.1",
  "labels": ["real", "fake"],
  "global": {
    "accuracy": 0.97909,
    "macro_f1": 0.9793,
    "micro_f1": 0.97909,
    "balanced_accuracy": 0.9796
  },
  "per_class": {
    "real": {
      "precision": 0.9649,
      "recall": 0.966,
      "f1": 0.9655,
      "support": 2077
    },
    "fake": {
      "precision": 0.9659,
      "recall": 0.9933,
      "f1": 0.9794,
      "support": 2083
    }
  },
  "confusion_matrix": {
    "labels": ["real", "fake"],
    "matrix": [
      [2004, 73],
      [14, 2069]
    ]
  },
  "calibration": {
    "ece": null,
    "brier": null
  },
  "meta": {
    "model": "logistic_regression_tfidf",
    "vectorizer": "tfidf_v1",
    "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
    "seed": 42
  }
}
```

---

## 16. Qualitative Inspection (Recommended Practice)

Maintain a `misclassified_samples.jsonl`:

```jsonl
{"id":"sample_153","true_label":"real","pred_label":"fake","text":"Satirical headline ...","rationale":"Contains hyperbolic adjectives"}
{"id":"sample_441","true_label":"fake","pred_label":"real","text":"Neutral phrased fabricated statistic ...","rationale":"No obvious lexical triggers"}
```

Use this for iterative error-driven development (EDA loops).

---

## 17. Adaptation to Ghanaian Context

| Gap                                                      | Impact                            | Action                                                 |
| -------------------------------------------------------- | --------------------------------- | ------------------------------------------------------ |
| Code-switch English–Twi/Pidgin                           | Tokenization fragmentation        | Train SentencePiece on mixed corpus                    |
| Local entity novelty                                     | Missed cues of fabrication        | Gazetteer (politicians, institutions) + NER            |
| Cultural satire forms                                    | False positives                   | Curate labeled satire set                              |
| Regional misinformation topics (agro, health, elections) | Underperformance on niche domains | Topic-stratified fine-tuning                           |
| Low-resource languages (Ewe, Ga)                         | Exclusion                         | Leverage transfer learning + multilingual augmentation |

---

## 18. Risk of Overfitting / Performance Inflation

Potential inflation vectors:

- Leakage via source/domain tokens (e.g., domain names strongly correlated with label)
- Duplicated or near-duplicate claims across train/test
- Single random split vs multi-split reliability
  Mitigation: Deduplicate by normalized text hash; implement k-fold; report variance.

---

## 19. Proposed Next Experimental Matrix

| Experiment | Change                      | Hypothesis                            | Success Criterion             |
| ---------- | --------------------------- | ------------------------------------- | ----------------------------- |
| A          | Add char 3–5 n-grams        | Better handling of obfuscation        | +0.5–1.0 F1 (fake class)      |
| B          | Re-balance thresholds       | Reduce FP while keeping recall >0.985 | Precision +1%                 |
| C          | AfriBERTa fine-tune 3 seeds | Stable lift                           | Std(F1) < 0.004               |
| D          | Add satire class            | Fewer false positives                 | FP (real vs fake) reduced 20% |
| E          | Multilingual augmentation   | Better on synthetic code-switch set   | +Recall on dialect test set   |

---

## 20. Formulas (Reference)

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 _ (P _ R) / (P + R)
- FPR = FP / (FP + TN)
- FNR = FN / (FN + TP)
- Support(class) = TP + FN (for positive) or TN + FP (for negative)

---

## 21. Summary

The baseline logistic regression + TF‑IDF model achieves strong _apparent_ performance (Accuracy ≈ 97.9%, Recall > 99%), but:

- May overfit dataset artifacts
- Has unvalidated generalization to Ghanaian linguistic and cultural contexts
- Lacks robust engineering (path handling, reproducibility, error logging)
- Omits multi-class nuance and calibration
  AfriBERTa offers likely moderate improvements but requires careful fine-tuning and evaluation rigor to justify added complexity.

A structured roadmap (data quality → reproducibility → multilingual adaptation → richer labels → knowledge integration) will maximize impact while managing model risk.

---

## 22. Immediate Action Checklist

- [ ] Add deterministic seed + cross-validation
- [ ] Fix metric key mismatch & extend schema
- [ ] Harden CSV ingestion
- [ ] Save & load vectorizer with defensive checks
- [ ] Register pytest markers
- [ ] Introduce misclassified sample logging
- [ ] Begin dialect corpus curation (Ghana news + social media)
- [ ] Pilot AfriBERTa with 3 random seeds, log variance

---

Prepared for: Fake News Detection for Social Media (Ghana Adaptation Prototype)  
Author: (Generated analytical summary)  
Date (UTC): {{INSERT_GENERATION_DATE}}
