# end-of-sem-assessment

# Fake News Detection for Social Media: Towards an Adaptable Model for Ghana

## Overview

This project implements and evaluates reproducible machine learning models for fake news detection, focusing on English-language news/social media datasets, with the goal of providing a prototype adaptable to the Ghanaian context and beyond. The pipeline supports classic ML (logistic regression) as well as transformer-based approaches (including AfriBERTa) and is extensible to multilingual African fake news detection.

- **Proposal**: See [`proposal.md`](./proposal.md)
- **Colab Notebook Example**: See [`fakereal_news.ipynb`](./notebooks/fakereal_news.ipynb)
- **Metrics & Report**: See [`/reports/metrics.json`](./reports/metrics.json), [`/reports/fake_news_lnc_report.pdf`](./reports/fake_news_lnc_report.pdf)
- **Error/Bias Analysis**: See [`/reports/error_analysis.md`](./reports/error_analysis.md)

## Project Structure

```
/data/             # Place raw csv and preprocessed npy files here (see dataset links below)
/models/           # Saved model files (logistic_regression_model.joblib, vectorizer.joblib, AfriBERTa, etc.)
/reports/          # Metrics, error analysis, final report, plots
/project/          # Source code (preprocess.py, train.py, eval.py, utils.py)
/tests/            # Unit tests (pytest)
README.md
requirements.txt
proposal.md
fakereal_news.ipynb
```

## Data

**Due to dataset size, raw data files are not included in this repo. Please download and place them as instructed below.**

- **English datasets**: Place `train.csv` or similar in `/data/`.
  - [GDrive folder containing sample train.csv / test.csv](https://drive.google.com/drive/folders/1DucxJ_Jp35Ipf6tZj87E_kAimnLHivGH?usp=sharing)
- **AfriBERTa Model & African datasets**:
  - AfriBERTa base model: [HuggingFace castorini/afriberta_base](https://huggingface.co/castorini/afriberta_base)
  - For future adaptation to Ghanaian or African languages, see [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news), [Lacuna Fund](https://github.com/africanlp/africanlp-public-datasets), and [Mul-FaD](https://huggingface.co/datasets/afrisenti/mul-fad).

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   _Python 3.8+ recommended. (Tested with Python 3.13, scikit-learn 1.4+, transformers 4.56+.)_

2. **Download Data:**
   - Download the appropriate `train.csv` from [GDrive](https://drive.google.com/drive/folders/1DucxJ_Jp35Ipf6tZj87E_kAimnLHivGH?usp=sharing) or other sources and place in `/data/`.
   - Make sure the dataset has the following column labels `id, title, author, text, label`.
   - For AfriBERTa or multilingual experiments, download the model from [https://huggingface.co/castorini/afriberta_base](https://huggingface.co/castorini/afriberta_base).

## How To Run

All scripts are in `/project/` and can be run as Python modules.

### 1. Preprocessing

```bash
python project/preprocess.py
```

## OR

```bash
python -m project.preprocess --data_dir data
```

- Loads `train.csv` from `/data/`, processes and splits data, saves `x_train.npy`, `x_test.npy`, `y_train.npy`, `y_test.npy` in `/data/`, and saves the vectorizer to `/models/`.

### 2. Training

```bash
python -m project.train
```

## OR

```bash
python -m project.train --data_dir data --out_dir models
```

- Loads processed data, trains logistic regression, saves model to `/models/logistic_regression_model.joblib`.

### 3. Evaluation

```bash
python -m project.eval
```

## OR

```bash
python -m project.eval --data_dir data --model_dir models --out reports/metrics.json
```

- Loads test data and model, outputs metrics and confusion matrix, saves metrics to `/reports/metrics.json`.

### 4. AfriBERTa (Transformer) Experiments

The notebook [`fakereal_news.ipynb`](./fakereal_news.ipynb) contains full code for:

- Fine-tuning and evaluating AfriBERTa on the same dataset
- (Optional) Adapting to African languages with MasakhaNEWS, Lacuna Fund, or Mul-FaD (requires manual download, see links above)

### 5. Unit Tests

```bash
python -m pytest -v
```

Or for pipeline tests only:

```bash
python -m pytest -m pipeline
```

Or for preprocess tests only:

```bash
python -m pytest -q -m "not pipeline"
```

(See `/tests/` for test coverage.)

## Results

Typical evaluation metrics (logistic regression baseline on English data):

```json
{
  "accuracy": 0.979,
  "precision": 0.966,
  "recall": 0.993,
  "f1_score": 0.979,
  "confusion_matrix": [
    [2004, 73],
    [14, 2069]
  ]
}
```

See `/reports/metrics.json` for full results.

AfriBERTa achieves even higher F1 on the same split (see notebook for details).

## Error/Bias Analysis

See `/reports/error_analysis.md` for detailed analysis and recommendations.

## Adapting to Ghanaian and African Languages

- To extend to Twi, Ga, Ewe, Hausa, Igbo, etc., fine-tune AfriBERTa using [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news), [Lacuna Fund](https://github.com/africanlp/africanlp-public-datasets), or [Mul-FaD](https://huggingface.co/datasets/afrisenti/mul-fad).
- For instructions, see the relevant section in [`fakereal_news.ipynb`](./fakereal_news.ipynb).

## Acknowledgements

- [AfriBERTa model (castorini/afriberta_base)](https://huggingface.co/castorini/afriberta_base)
- [My google drive, dataset from Kaggle](https://drive.google.com/drive/folders/1DucxJ_Jp35Ipf6tZj87E_kAimnLHivGH?usp=sharing)
- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [scikit-learn](https://scikit-learn.org/)
- [nltk](https://www.nltk.org/)
- [MasakhaNEWS](https://github.com/masakhane-io/masakhane-news), [Lacuna Fund](https://github.com/africanlp/africanlp-public-datasets), [Mul-FaD](https://huggingface.co/datasets/afrisenti/mul-fad)

## License

MIT License

---

## Example Pipeline Output

```bash
$ python -m project.preprocess
[nltk_data] Downloading package stopwords to ...
[nltk_data]   Package stopwords is already up-to-date!
Preprocessing complete. Data and vectorizer saved.

$ python -m project.train
Training complete. Model saved to ./models/logistic_regression_model.joblib

$ python -m project.eval
[nltk_data] Downloading package stopwords to ...
[nltk_data]   Package stopwords is already up-to-date!
Evaluation complete. Metrics saved to ./reports/metrics.json
{'accuracy': 0.9790865384615385, 'precision': 0.9659197012138189, 'recall': 0.9932789246279404, 'f1_score': 0.9794082840236686, 'confusion_matrix': [[2004, 73], [14, 2069]]}
Ghanaian News Prediction: Fake

$ pytest -v
========================================== test session starts ===========================================
platform ... -- Python 3.13.7, pytest-8.4.1 ...
collected 3 items

tests/test_pipeline.py::test_pipeline_training_and_evaluation PASSED
tests/test_preprocess.py::test_normalize_text_basic PASSED
tests/test_preprocess.py::test_normalize_text_empty PASSED

================================= 3 passed, 1 warning in ...s ================================
```

---

For more details, see the [proposal](./proposal.md) and [notebook](./fakereal_news.ipynb).
