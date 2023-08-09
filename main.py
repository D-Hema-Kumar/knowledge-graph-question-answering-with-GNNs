from core.NeuralNet.MLP import MLP
from core.experiments.utils import (save_experiment_results_to_file,evaluate_model)
from core.experiments.ContextClasses import (DataContext, TrainingContext)
from core.experiments.binary_classification.binary_classification_experiment import (
    BinaryClassificationExperiment,
)
from core.experiments.multi_class_classification.multi_class_classification_experiment import (
    MultiClassificationExperiment,
)
from config.config import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    GRAPH_EMBEDDINGS_PATH,
)
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

# EXPERIMENT 1
'''
training_context = TrainingContext(num_epochs = 1000,
                                   learning_rate = 0.01,
                                   dim_hidden_layer = 16
                                    )

data_context = DataContext(triples_path = TRIPLES_PATH,
                           entities_labels_path = ENTITIES_LABELS_PATH,
                           properties_labels_path = PROPERTIES_LABELS_PATH,
                           graph_embeddings_path = GRAPH_EMBEDDINGS_PATH)

experiment_1 = BinaryClassificationExperiment(
    training_context=training_context,
    data_context=data_context,
    labeler=(
        lambda x: 1 if "APQC" in x else 0
    ),  # Task: predict whether a given node belongs to "APQC"
    model_type=MLP,
)
experiment_1.train()

model = experiment_1.model
data = experiment_1.data

## EVALUATE
logger.info("Evaluating")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}")

# Log Experiment data
evaluation_results = {}
evaluation_results["precision"] = precision
evaluation_results["recall"] = recall
evaluation_results["F1"] =  F1
save_experiment_results_to_file(file_name = "experiments.csv",
                                experiment_context = "Task: predict whether a given node belongs to APQC",
                                data_context=data_context,
                                training_context=training_context,
                                experiment=experiment_1,
                                evaluation_results=evaluation_results
                                )
'''

# EXPERIMENT 2
'''
training_context = TrainingContext(num_epochs = 1000,
                                   learning_rate = 0.01,
                                   dim_hidden_layer = 16
                                    )

data_context = DataContext(triples_path = TRIPLES_PATH,
                           entities_labels_path = ENTITIES_LABELS_PATH,
                           properties_labels_path = PROPERTIES_LABELS_PATH,
                           graph_embeddings_path = GRAPH_EMBEDDINGS_PATH)

experiment_2 = BinaryClassificationExperiment(
    training_context=training_context,
    data_context=data_context,
    labeler=(
        lambda x: 1 if "APQC" in x else 0
    ),  # Task: predict whether a given node belongs to "APQC"
    model_type=GCN,
)
# TRAIN
experiment_2.train()
model = experiment_2.model
data = experiment_2.data

## EVALUATE
logger.info("Evaluating")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}")

# TRACK
evaluation_results = {"precision":precision,"recall":recall,"F1":F1}
save_experiment_results_to_file(file_name = "experiments.csv",
                                experiment_context = "Task: predict whether a given node belongs to APQC",
                                data_context=data_context,
                                training_context=training_context,
                                experiment=experiment_2,
                                evaluation_results=evaluation_results
                                )
'''
# Multi class classification Experiment

training_context = TrainingContext(num_epochs = 1000,
                                   learning_rate = 0.01,
                                   num_layers=3,
                                   dim_hidden_layer = 256
                                    )

data_context = DataContext(triples_path = TRIPLES_PATH,
                           entities_labels_path = ENTITIES_LABELS_PATH,
                           properties_labels_path = PROPERTIES_LABELS_PATH,
                           graph_embeddings_path = GRAPH_EMBEDDINGS_PATH)

multiCls_experiment = MultiClassificationExperiment(
    training_context=training_context,
    data_context=data_context,
    labeler=None,
    model_type=GCN,
)
# TRAIN
multiCls_experiment.train()
model = multiCls_experiment.model
data = multiCls_experiment.data

# EVALUATE & SAVE
multiCls_experiment.eval()
multiCls_experiment.save_model()

# TRACK
evaluation_results = {"accuracy":'{:.2%}'.format(multiCls_experiment.accuracy.item()),"precision":"","recall":"","F1":""}
print("eval accuracy",evaluation_results)
save_experiment_results_to_file(file_name = "experiments.csv",
                                experiment_context = "Task: multiclass node type prediction",
                                data_context=data_context,
                                training_context=training_context,
                                experiment=multiCls_experiment,
                                evaluation_results=evaluation_results
                                )