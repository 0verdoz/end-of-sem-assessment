import pytest
import os
import json
import subprocess

@pytest.mark.pipeline
def test_pipeline_training_and_evaluation():
    data_dir = "data"
    model_dir = "models"
    metrics_path = "reports/metrics.json"

    # Clean up old model + metrics if needed
    if os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    # Run preprocessing
    preprocess_cmd = ["python", "project/preprocess.py"]
    preprocess_result = subprocess.run(preprocess_cmd, capture_output=True, text=True)
    assert preprocess_result.returncode == 0, f"Preprocessing failed:\n{preprocess_result.stderr}"


    # Run training (no CLI args)
    train_cmd = ["python", "-m", "project.train"]
    train_result = subprocess.run(train_cmd, capture_output=True, text=True)
    assert train_result.returncode == 0, f"Training failed:\n{train_result.stderr}"

    # Check model exists
    model_file = os.path.join(model_dir, "logistic_regression_model.joblib")
    assert os.path.isfile(model_file), "Trained model file not found"

    # Run evaluation (no CLI args)
    eval_cmd = ["python", "-m", "project.eval"]
    eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
    assert eval_result.returncode == 0, f"Evaluation failed:\n{eval_result.stderr}"

    # Check metrics
    assert os.path.isfile(metrics_path), "metrics.json file not found"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    f1 = float(metrics.get("f1_score", 0))
    assert f1 >= 0.40, f"F1 score too low: {f1}"