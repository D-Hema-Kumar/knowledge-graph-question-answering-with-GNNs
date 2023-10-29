from core.NeuralNet.MLP import MLP
from core.experiments.utils import (save_experiment_results_to_file,evaluate_model, QAEvaluationMetrcis)
from core.experiments.ContextClasses import (DataContext, TrainingContext, QADataContext)
from core.experiments.binary_classification.binary_classification_experiment import (
    BinaryClassificationExperiment,
)
from core.experiments.multi_class_classification.multi_class_classification_experiment import (
    MultiClassificationExperiment,
)

from core.experiments.qa.qa_experiment import (QAExperiment, MetaQAExperiment)
from config.config import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    GRAPH_EMBEDDINGS_PATH,
    KG_EMBEDDINGS_PATH
)

from config.config import (
    
    TRIPLES_PATH_OLD,
    ENTITIES_LABELS_PATH_OLD,
    PROPERTIES_LABELS_PATH_OLD,
    GRAPH_EMBEDDINGS_PATH_OLD,
    QUESTIONS_CONCEPTS_ANSWERS_PATH,
    GRAPH_EMBEDDINGS_PATH_OLD,
    QUESTIONS_EMBEDDINGS_PATH,
    QA_TRAINING_FILE_PATH,
    QA_TESTING_FILE_PATH,
    MetaQA_CONFIG,
    MetaQA_KG_EMBEDDINGS_PATH
)
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN, RGCN, RGAT
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

GNN_MODELS = [GCN,RGCN, RGAT]
kg_embedding_models = KG_EMBEDDINGS_PATH.keys()

# Binary Classification Experiment
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
'''
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
'''

# QA on VAD Experiments

#1 Hypothesis 1 -- Embeddings with LLM
'''
training_context = TrainingContext(info = "Task: QA with GNN",
                                   num_epochs = 20,
                                   learning_rate = 0.01,
                                   num_layers=2,
                                   dim_hidden_layer = 16,
                                   num_bases= None
                                    )

data_context = QADataContext(triples_path = TRIPLES_PATH_OLD,
                           entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                           properties_labels_path = PROPERTIES_LABELS_PATH_OLD,
                           graph_embeddings_path = GRAPH_EMBEDDINGS_PATH_OLD,
                           training_questions_concepts_answers_file_path=QA_TRAINING_FILE_PATH,
                           testing_questions_concepts_answers_file_path=QA_TESTING_FILE_PATH,
                           questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH)

for i in range(20):
    qa_experiment = QAExperiment(
        training_context=training_context,
        data_context=data_context,
        model_type=GCN
    )
    # TRAIN, SAVE & EVAL
    qa_experiment.run() 

'''
#2 Hypothesis 2 -- Embeddings with KGE Models
'''
training_context = TrainingContext(info = "Task: QA with GCN; node embeddings initialized with complex",
                                   num_epochs = 20,
                                   learning_rate = 0.01,
                                   num_layers=2,
                                   dim_hidden_layer = 16,
                                   num_bases= None
                                    )

data_context = QADataContext(triples_path = TRIPLES_PATH_OLD,
                           entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                           properties_labels_path = PROPERTIES_LABELS_PATH_OLD,
                           LM_embeddings_path = None,
                           KG_embeddings_path = KG_EMBEDDINGS_PATH['complex'],
                           training_questions_concepts_answers_file_path=QA_TRAINING_FILE_PATH,
                           testing_questions_concepts_answers_file_path=QA_TESTING_FILE_PATH,
                           training_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH,
                           testing_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH)


qa_experiment = QAExperiment(
    training_context=training_context,
    data_context=data_context,
    model_type=GCN
)
# TRAIN, SAVE & EVAL
qa_experiment.run() 
'''

'''
for gnn_model in GNN_MODELS:
    for kg_model  in KG_EMBEDDINGS_PATH.keys():
        
        training_context = TrainingContext(info = f"Task: QA with {gnn_model.__name__}; node embeddings initialized with {kg_model}",
                                    num_epochs = 20,
                                    learning_rate = 0.01,
                                    num_layers=2,
                                    dim_hidden_layer = 16,
                                    num_bases= None
                                    )

        data_context = QADataContext(triples_path = TRIPLES_PATH_OLD,
                            entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                            properties_labels_path = PROPERTIES_LABELS_PATH_OLD,
                            LM_embeddings_path = None,
                            KG_embeddings_path = KG_EMBEDDINGS_PATH[kg_model],
                            training_questions_concepts_answers_file_path=QA_TRAINING_FILE_PATH,
                            testing_questions_concepts_answers_file_path=QA_TESTING_FILE_PATH,
                            training_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH,
                            testing_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH)


        qa_experiment = QAExperiment(
                        training_context=training_context,
                        data_context=data_context,
                        model_type=gnn_model
                    )

        # TRAIN, SAVE & EVAL
        qa_experiment.run() 
'''
#3 Hypothesis 3 -- Embeddings with LM + KGE Models
'''
training_context = TrainingContext(info = "Task: QA with GCN; node embeddings initialized with RoBERTa+DistMult",
                                   num_epochs = 20,
                                   learning_rate = 0.01,
                                   num_layers=2,
                                   dim_hidden_layer = 16,
                                   num_bases= None
                                    )

data_context = QADataContext(triples_path = TRIPLES_PATH_OLD,
                           entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                           properties_labels_path = PROPERTIES_LABELS_PATH_OLD,
                           LM_embeddings_path = GRAPH_EMBEDDINGS_PATH_OLD,
                           KG_embeddings_path = KG_EMBEDDINGS_PATH['distmult'],
                           training_questions_concepts_answers_file_path=QA_TRAINING_FILE_PATH,
                           testing_questions_concepts_answers_file_path=QA_TESTING_FILE_PATH,
                           training_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH,
                           testing_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH)


qa_experiment = QAExperiment(
    training_context=training_context,
    data_context=data_context,
    model_type=GCN
)
# TRAIN, SAVE & EVAL
qa_experiment.run() 

'''

'''
for gnn_model in GNN_MODELS:
    for kg_model  in KG_EMBEDDINGS_PATH.keys():
        

        training_context = TrainingContext(info = f"Task: QA with {gnn_model.__name__}; node embeddings initialized with {kg_model}+RoBERTa",
                                   num_epochs = 20,
                                   learning_rate = 0.01,
                                   num_layers=2,
                                   dim_hidden_layer = 16,
                                   num_bases= None
                                    )

        data_context = QADataContext(triples_path = TRIPLES_PATH_OLD,
                           entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                           properties_labels_path = PROPERTIES_LABELS_PATH_OLD,
                           LM_embeddings_path = GRAPH_EMBEDDINGS_PATH_OLD,
                           KG_embeddings_path = KG_EMBEDDINGS_PATH[kg_model],
                           training_questions_concepts_answers_file_path=QA_TRAINING_FILE_PATH,
                           testing_questions_concepts_answers_file_path=QA_TESTING_FILE_PATH,
                           training_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH,
                           testing_questions_embeddings_path = QUESTIONS_EMBEDDINGS_PATH)


        qa_experiment = QAExperiment(
                        training_context=training_context,
                        data_context=data_context,
                        model_type=gnn_model
                    )
        
        # TRAIN, SAVE & EVAL
        qa_experiment.run() 

'''
# QA MetaQA Experiments
'''
training_context = TrainingContext(info = "Task: MetaQA with GNN",
                                   num_epochs = 2,
                                   learning_rate = 0.01,
                                   num_layers=2,
                                   dim_hidden_layer = 16,
                                   num_bases= None
                                    )

data_context = QADataContext(   triples_path=MetaQA_CONFIG['KB_PATH'],
                                entities_labels_path=MetaQA_CONFIG['ENTITIES_LABELS_PATH'],
                                properties_labels_path=MetaQA_CONFIG['PROPERTIES_LABELS_PATH'],
                                graph_embeddings_path=MetaQA_CONFIG['GRAPH_EMBEDDINGS_PATH'],
                                training_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_DEV_PATH'],
                                testing_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_DEV_PATH'],
                                training_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_DEV_PATH'],
                                testing_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_DEV_PATH'],
                                training_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_DEV_FILE_PATH'],
                                testing_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_DEV_FILE_PATH'],
                                is_vad_kb=False
                            )

qa_experiment = MetaQAExperiment(
        training_context=training_context,
        data_context=data_context,
        model_type=GCN
    )
# TRAIN, SAVE & EVAL
qa_experiment.run()
'''

# Hypothesis 2 MetaQA with only KGE embeddings



for gnn_model in GNN_MODELS:
    for kge_model in [k for k in MetaQA_KG_EMBEDDINGS_PATH.keys() if k != 'roberta']:
        
        training_context = TrainingContext(info = f"Task: 1-hop MetaQA with {gnn_model.__name__}; node embeddings initialized with {kge_model}",
                                        num_epochs = 2,
                                        learning_rate = 0.01,
                                        num_layers=2,
                                        dim_hidden_layer = 16,
                                        num_bases= None
                                            )

        data_context = QADataContext(   triples_path=MetaQA_CONFIG['KB_PATH'],
                                        entities_labels_path=MetaQA_CONFIG['ENTITIES_LABELS_PATH'],
                                        properties_labels_path=MetaQA_CONFIG['PROPERTIES_LABELS_PATH'],
                                        KG_embeddings_path=MetaQA_KG_EMBEDDINGS_PATH[kge_model],
                                        LM_embeddings_path=None,
                                        training_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TRAIN_PATH'],
                                        testing_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TEST_PATH'],
                                        training_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_TRAIN_PATH'],
                                        testing_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_TEST_PATH'],
                                        training_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_TRAIN_FILE_PATH'],
                                        testing_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_TEST_FILE_PATH'],
                                        is_vad_kb=False
                                    )

        qa_experiment = MetaQAExperiment(
                training_context=training_context,
                data_context=data_context,
                model_type=gnn_model
            )
        # TRAIN, SAVE & EVAL
        qa_experiment.run()
        



# Hypothesis 3 MetaQA with KGE+RoBERTa embeddings

for gnn_model in GNN_MODELS:
    for kge_model in [k for k in MetaQA_KG_EMBEDDINGS_PATH.keys() if k != 'roberta']:
        
        training_context = TrainingContext(
                                info = f"Task: 1-hop MetaQA with {gnn_model.__name__}; node embeddings initialized with {kge_model}+RoBERTa",
                                num_epochs = 2,
                                learning_rate = 0.01,
                                num_layers=2,
                                dim_hidden_layer = 16,
                                num_bases= None
                                    )

        data_context = QADataContext(   
                                triples_path=MetaQA_CONFIG['KB_PATH'],
                                entities_labels_path=MetaQA_CONFIG['ENTITIES_LABELS_PATH'],
                                properties_labels_path=MetaQA_CONFIG['PROPERTIES_LABELS_PATH'],
                                LM_embeddings_path=MetaQA_KG_EMBEDDINGS_PATH['roberta'],
                                KG_embeddings_path=MetaQA_KG_EMBEDDINGS_PATH[kge_model],
                                training_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TRAIN_PATH'],
                                testing_questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TEST_PATH'],
                                training_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_TRAIN_PATH'],
                                testing_questions_embeddings_path = MetaQA_CONFIG['QUESTIONS_EMBEDDINGS_1_HOP_TEST_PATH'],
                                training_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_TRAIN_FILE_PATH'],
                                testing_subgraphs_file_path = MetaQA_CONFIG['SUBGRAPHS_1_HOP_TEST_FILE_PATH'],
                                is_vad_kb=False
                                    )

        qa_experiment = MetaQAExperiment(
                training_context=training_context,
                data_context=data_context,
                model_type=gnn_model
            )
        # TRAIN, SAVE & EVAL
        qa_experiment.run()
