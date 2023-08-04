from core.NeuralNet.MLP import MLP
from core.experiments.binary_classification.binary_classification_experiment import (
    BinaryClassificationExperiment,
)
from config.config import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    GRAPH_EMBEDDINGS_PATH,
)
from core.NeuralNet.MLP import MLP
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# EXPERIMENT 1
data_context = {
    "TRIPLES_PATH": TRIPLES_PATH,
    "ENTITIES_LABELS_PATH": ENTITIES_LABELS_PATH,
    "PROPERTIES_LABELS_PATH": PROPERTIES_LABELS_PATH,
    "GRAPH_EMBEDDINGS_PATH": GRAPH_EMBEDDINGS_PATH,
}
training_context = {
    "NUM_EPOCHS": 1000,
    "LEARNING_RATE": 0.001,
    "DIM_HIDDEN_LAYER": 12,
}
experiment_1 = BinaryClassificationExperiment(
    training_context=training_context,
    data_context=data_context,
    labeler=(
        lambda x: 1 if "APQC" in x else 0
    ),  # Task: predict whether a given node belongs to "APQC"
    model_type=MLP,
)
experiment_1.train()
# experiment_1.evaluate()


model = experiment_1.model
data = experiment_1.data


def evaluate_model(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct_predictions_mask = pred[data.test_mask] == data.y[data.test_mask]
    incorrect_predictions_mask = pred[data.test_mask] != data.y[data.test_mask]
    TP = sum(pred[data.test_mask][correct_predictions_mask] == 1)
    FP = sum(pred[data.test_mask][incorrect_predictions_mask] == 1)
    FN = sum(pred[data.test_mask][incorrect_predictions_mask] == 0)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1


precision, recall, F1 = evaluate_model(model, data)
print(precision, recall, F1)
