TRIPLES_PATH = "./data/source_data/VAD_triples.csv"
ENTITIES_LABELS_PATH = "./data/source_data/VAD_entities.csv"
PROPERTIES_LABELS_PATH = "./data/source_data/VAD_properties.csv"
QUESTIONS_ANSWERS_PATH = "./data/source_data/VAD_questions_answers.csv"

GRAPH_EMBEDDINGS_PATH = "./data/graph_embeddings"
QUESTIONS_EMBEDDINGS_PATH = "./data/question_embeddings"
EXPERIMENT_RESULTS_PATH = {"qa":"./core/experiments/qa/results",
                           "eval_results":"./core/experiments",
                           "binary_classification":"./core/experiments/binary_classification/results",
                           "multi_class_classification":"./core/experiments/multi_class_classification/results"}

EXPERIMENT_TYPES_PATH = {"qa":"./core/experiments/qa/",
                         "binary_classification":"./core/experiments/binary_classification/",
                         "multi_class_classification":"./core/experiments/multi_class_classification/"}

GRAPH_EMBEDDINGS_WITH_COMMENT_PATH = "./data/graph_embeddings_with_comment"

KG_EMBEDDINGS_PATH = {
    "complex":"./data/graph_embeddings/complex",
    "distmult":"./data/graph_embeddings/distmult",
    "transe":"./data/graph_embeddings/transe",
}

LM_EMBEDDINGS_PATH = {"roberta":"./data/graph_embeddings/roberta"
}

# old VAD
TRIPLES_PATH_OLD = "./data/source_data_old/VAD_triples.csv"
ENTITIES_LABELS_PATH_OLD = "./data/source_data_old/VAD_entities_labels.csv"
PROPERTIES_LABELS_PATH_OLD = "./data/source_data_old/VAD_properties_labels.csv"
QUESTIONS_CONCEPTS_ANSWERS_PATH = "./data/source_data_old/resource_type_questions_concepts_answers.csv"
QA_TRAINING_FILE_PATH = "./data/source_data_old/qa_training_data.csv"
QA_TESTING_FILE_PATH = "./data/source_data_old/qa_testing_data.csv"
GRAPH_EMBEDDINGS_PATH_OLD = "./data/graph_embeddings_old"
SOURCE_DATA_PATH_OLD = "./data/source_data_old/"

VAD_KGQA_GoldStandard = "./data/source_data_old/OPAL_KGQA_GoldStandard.json"

#MetaQA
MetaQA_CONFIG = {"KB_PATH":"./data/MetaQA/source_data/kb.txt",
                "ENTITIES_LABELS_PATH":"./data/MetaQA/source_data/kb_entities.csv",
                 "PROPERTIES_LABELS_PATH": "./data/MetaQA/source_data/kb_properties.txt",
                 "GRAPH_EMBEDDINGS_PATH":"./data/MetaQA/graph_embeddings",

                 "QUESTIONS_CONCEPTS_ANSWERS_1_HOP_FOLDER_PATH": "data/MetaQA/source_data/one-hop/",

                 "QUESTIONS_CONCEPTS_ANSWERS_1_HOP_DEV_PATH": "data/MetaQA/source_data/one-hop/qca_dev.csv",
                 "QUESTIONS_EMBEDDINGS_1_HOP_DEV_PATH":"./data/MetaQA/question_embeddings/one-hop/dev",
                 "SUBGRAPHS_1_HOP_DEV_FILE_PATH":"data/MetaQA/source_data/one-hop/dev_subgraphs.npz",

                 "QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TRAIN_PATH": "data/MetaQA/source_data/one-hop/qca_train.csv",
                 "QUESTIONS_EMBEDDINGS_1_HOP_TRAIN_PATH":"./data/MetaQA/question_embeddings/one-hop/train",
                 "SUBGRAPHS_1_HOP_TRAIN_FILE_PATH":"data/MetaQA/source_data/one-hop/train_subgraphs.npz",

                 "QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TEST_PATH": "data/MetaQA/source_data/one-hop/qca_test.csv",
                 "QUESTIONS_EMBEDDINGS_1_HOP_TEST_PATH":"./data/MetaQA/question_embeddings/one-hop/test",
                 "SUBGRAPHS_1_HOP_TEST_FILE_PATH":"data/MetaQA/source_data/one-hop/test_subgraphs.npz",

                 
                 "QUESTIONS_CONCEPTS_ANSWERS_2_HOP_FOLDER_PATH": "data/MetaQA/source_data/two-hop/",

                 "QUESTIONS_CONCEPTS_ANSWERS_2_HOP_DEV_PATH": "data/MetaQA/source_data/two-hop/qca_dev.csv",
                 "QUESTIONS_EMBEDDINGS_2_HOP_DEV_PATH":"./data/MetaQA/question_embeddings/two-hop/dev",
                 "SUBGRAPHS_2_HOP_DEV_FILE_PATH":"data/MetaQA/source_data/two-hop/dev_subgraphs.npz",

                 "QUESTIONS_CONCEPTS_ANSWERS_2_HOP_TRAIN_PATH": "data/MetaQA/source_data/two-hop/qca_train.csv",
                 "QUESTIONS_EMBEDDINGS_2_HOP_TRAIN_PATH":"./data/MetaQA/question_embeddings/two-hop/train",
                 "SUBGRAPHS_2_HOP_TRAIN_FILE_PATH":"data/MetaQA/source_data/two-hop/train_subgraphs.npz",

                 "QUESTIONS_CONCEPTS_ANSWERS_2_HOP_TEST_PATH": "data/MetaQA/source_data/two-hop/qca_test.csv",
                 "QUESTIONS_EMBEDDINGS_2_HOP_TEST_PATH":"./data/MetaQA/question_embeddings/two-hop/test",
                 "SUBGRAPHS_2_HOP_TEST_FILE_PATH":"data/MetaQA/source_data/two-hop/test_subgraphs.npz",

                 "QUESTIONS_CONCEPTS_ANSWERS_3_HOP_FOLDER_PATH": "data/MetaQA/source_data/three-hop/"
                 }


MetaQA_KG_EMBEDDINGS_PATH = {
    "complex":"./data/MetaQA/graph_embeddings/complex",
    "distmult":"./data/MetaQA/graph_embeddings/distmult",
    "transe":"./data/MetaQA/graph_embeddings/transe",
    "roberta":"./data/graph_embeddings/roberta"
}