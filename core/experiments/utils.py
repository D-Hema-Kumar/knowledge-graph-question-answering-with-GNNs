import csv
import pandas as pd
import numpy as np
import os
from config.config import (
    EXPERIMENT_RESULTS_PATH )
from core.experiments.ContextClasses import (DataContext, TrainingContext)
from torch_geometric.data import Data
from loguru import logger
import datetime
import torch
from tqdm import tqdm
import ast

def evaluate_model(model, data):
    #### Function to evaluate a model and return precision, recall and F1
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

def save_experiment_results_to_file(file_name, experiment_context:str,
                                    data_context:DataContext,
                                    training_context:TrainingContext,
                                    experiment, #TO DO set object type
                                    evaluation_results=None #TO DO pass class objects. for now passing dict object
                                    ):

    # Define the CSV field names for training context and evaluation results
    fieldnames = ["time_stamp","info","Epochs", "Learning Rate", "hidden_layer_dimension","num_layers","num_bases", "Model", 
                  "accuracy", "precision", "recall","F1","model_directory","triples_path","entities_labels_path",
                  "properties_labels_path","graph_embeddings_path"]
    file_path = os.path.join(EXPERIMENT_RESULTS_PATH["eval_results"], file_name)
    # Create a dictionary with the data to be saved
    if evaluation_results == None:
        data = {

        "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        "info":experiment_context,
        #data context
        "triples_path" : data_context.triples_path,
        "entities_labels_path" : data_context.entities_labels_path,
        "properties_labels_path" : data_context.properties_labels_path,
        "graph_embeddings_path" : data_context.graph_embeddings_path,

        # training context
        "Epochs": training_context.num_epochs,
        "Learning Rate": training_context.learning_rate,
        "hidden_layer_dimension": training_context.dim_hidden_layer,
        "num_layers":training_context.num_layers,
        "num_bases":training_context.num_bases,

        "Model": experiment.model,
        "model_directory":experiment.experiment_results_folder_path,

        # evaluation results
        
        "accuracy": None ,
        "precision": None,
        "recall": None ,
        "F1": None,

    
    }
        
    else:
        data = {

            "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "info":experiment_context,
            #data context
            "triples_path" : data_context.triples_path,
            "entities_labels_path" : data_context.entities_labels_path,
            "properties_labels_path" : data_context.properties_labels_path,
            "graph_embeddings_path" : data_context.graph_embeddings_path,

            # training context
            "Epochs": training_context.num_epochs,
            "Learning Rate": training_context.learning_rate,
            "hidden_layer_dimension": training_context.dim_hidden_layer,
            "num_layers":training_context.num_layers,
            "num_bases":training_context.num_bases,

            "Model": experiment.model,
            "model_directory":experiment.experiment_results_folder_path,

            # evaluation results
            
            "accuracy": evaluation_results["accuracy"],
            "precision": evaluation_results["precision"],
            "recall": evaluation_results["recall"],
            "F1":evaluation_results["F1"],

        
        }

    # Check if the file exists to decide whether to write headers or not
    file_exists = False
    try:
        with open(file_path, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    # Open the file in append mode to append new data
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the headers only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the data to the file
        writer.writerow(data)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    #print("device type: ", device)
    return device

def load_model(trained_model_path):
    return torch.load(trained_model_path, map_location=get_device())

def evaluate_qa_model(trained_model_path, qa_data_builder, model_name):
    device = get_device()
    data = qa_data_builder.build_data()
    logger.info("Loading model.")
    trained_model = load_model(os.path.join(trained_model_path,model_name))
    logger.info("Evaluating model.")
    trained_model.eval()
    with torch.no_grad():
        res = []
        for idx in tqdm(qa_data_builder.question_concepts_answers.index):

            row = qa_data_builder.question_concepts_answers.loc[idx]
            q_embedding = qa_data_builder.questions_to_embeddings[row["question"]]
            q_x = qa_data_builder.get_x(to_concat=q_embedding)
            q_edge_mask, q_nodes, q_concept_mask, q_answer_mask, q_answer_and_random_nodes_mask =qa_data_builder.get_concepts_and_masks_for_question(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"])
            q_edge_index = data.edge_index[:,q_edge_mask]
            q_edge_type = data.edge_type[q_edge_mask]
            q_training_x_mask = qa_data_builder.get_question_training_mask_for_x()
            q_y_labels = qa_data_builder.get_question_y_labels()
            q_data = Data(x=q_x,edge_index=q_edge_index,edge_type=q_edge_type,train_mask =q_training_x_mask,y=q_y_labels)
            q_data = q_data.to(device)
            logger.debug(f'Predicting for Q {idx} : {row["question"]}')
            out,_ = trained_model(q_data)
            predicted_answer_nodes = torch.where(out.argmax(dim=1))[0]
            predicted_answer_node_probabilities = out.max(dim=1)[0][predicted_answer_nodes]
            sorted_probability_indices = torch.argsort(predicted_answer_node_probabilities, descending= True)
            count_predicted_nodes =len(predicted_answer_nodes)
            actual_answer_nodes = q_nodes[q_answer_mask].tolist()
            if count_predicted_nodes > 0 and count_predicted_nodes < 50 :
                logger.debug(f"Answers predicted.")
                is_predicted_in_actual_answers = bool(set(actual_answer_nodes) & set(predicted_answer_nodes[sorted_probability_indices].tolist()))
                res.append((idx, actual_answer_nodes, predicted_answer_nodes[sorted_probability_indices].tolist(),predicted_answer_node_probabilities[sorted_probability_indices].tolist(),count_predicted_nodes,is_predicted_in_actual_answers))
            
            elif count_predicted_nodes >= 50:
                logger.debug(f"More than 50 answers found.")
                res.append((idx, actual_answer_nodes, np.nan,np.nan,count_predicted_nodes,False))
            else:
                logger.debug(f"NO answers found.")
                res.append((idx, actual_answer_nodes, np.nan,np.nan,0,False))
        
        
    eval_res = pd.DataFrame.from_records(res,columns=["q_idx","actual_answer_nodes","predicted_answer_nodes","probabilities_of_answer_nodes","count_predicted_nodes","is_predicted_in_actual"])
    eval_res.to_csv(os.path.join(trained_model_path,"evaluation_results.csv"),index=False)
    logger.info("Evaluation results saved.")

def evaluate_MetaQA_model(trained_model_path, qa_data_builder, model_name):

    device = get_device()
    #data = qa_data_builder.build_data()
    logger.info("Loading model.")
    trained_model = load_model(os.path.join(trained_model_path,model_name))
    logger.info("Evaluating model.")
    trained_model.eval()

    with torch.no_grad():
        res = []
        for idx, row in tqdm(qa_data_builder.testing_questions_concepts_answers.iterrows()):

            q_data, question_subgraph_nodes,question_subgraph_answer = qa_data_builder.get_question_data(question=row["question"], training=False)
            q_data = q_data.to(device)
            question_subgraph_nodes = question_subgraph_nodes.to(device)
            question_subgraph_answer = question_subgraph_answer.to(device).tolist()
            #predict
            out,_ = trained_model(q_data)

            predicted_local_answer_nodes = torch.where(out.argmax(dim=1))[0] # local subgraph answer
            predicted_answer_nodes = question_subgraph_nodes[predicted_local_answer_nodes] # remapping local answer to global node number
            predicted_answer_node_probabilities = out.max(dim=1)[0][predicted_local_answer_nodes]
            sorted_probability_indices = torch.argsort(predicted_answer_node_probabilities, descending= True)
            count_predicted_nodes =len(predicted_answer_nodes)

            if count_predicted_nodes > 0:
                #logger.debug(f"answers predicted")
                is_predicted_in_actual_answers = bool(set(question_subgraph_answer) & set(predicted_answer_nodes[sorted_probability_indices].tolist()))
                res.append((row["question"], question_subgraph_answer, predicted_answer_nodes[sorted_probability_indices].tolist(),predicted_answer_node_probabilities[sorted_probability_indices].tolist(),count_predicted_nodes,is_predicted_in_actual_answers))
            
            else:
                #logger.debug(f"NO answers found")
                res.append((row["question"], question_subgraph_answer, np.nan,np.nan,0,False))
            
            #break;
    eval_res = pd.DataFrame.from_records(res,columns=["question","actual_answer_nodes","predicted_answer_nodes","probabilities_of_answer_nodes","count_predicted_nodes","is_predicted_in_actual"])
    eval_res.to_csv(os.path.join(trained_model_path,"evaluation_results.csv"),index=False)
    logger.info("Evaluation results saved.")

    evaluation_metrcis = QAEvaluationMetrcis(trained_model_path)
    hits_1, hits_3, hits_5, mrr, precision_1, precision_3 ,precision_5, recall_1, recall_3,recall_5 = evaluation_metrcis.run_evaluation()
    F1 = evaluation_metrcis.f1	
    logger.info(f'F1 : {np.round(F1,2)} -- hits@1 : {np.round(hits_1,2)} -- hits@3 : {np.round(hits_3,2)} -- hits@5 : {np.round(hits_5,2)} -- MRR : {np.round(mrr,2)}')


class QAEvaluationMetrcis:
    def __init__(self,model_prediction_path:str):

        
        self.evaluation_results = pd.read_csv(os.path.join(model_prediction_path,'evaluation_results.csv'))
        
        # reads columns as lists instead of strings
        list_type_columns = ['actual_answer_nodes','predicted_answer_nodes', 'probabilities_of_answer_nodes']
        for col in list_type_columns:
            self.evaluation_results[col] = self.evaluation_results[col].apply(lambda x : ast.literal_eval(x) if type(x)==str else [])

        #one_answer_mask = self.evaluation_results['actual_answer_nodes'].apply(lambda x : True if len(x)==1 else False )

    def hits_at_k(self,predictions, actual, k):
        hits = 0
        for pred_nodes, actual_node in zip(predictions, actual):
            if any(node in pred_nodes[:k] for node in actual_node):
                hits += 1
        return hits / len(predictions)

    
    def reciprocal_rank(self, predictions, actual):
        ranks = []
        for pred_nodes, actual_node in zip(predictions, actual):
            if any(node in pred_nodes for node in actual_node):
                rank = pred_nodes.index(actual_node[0]) + 1 if actual_node[0] in pred_nodes else 0
                ranks.append(1 / rank if rank > 0 else 0)
        return sum(ranks) / len(predictions)
    
    def precision_at_k(self,predictions, actual, k):
        correct_predictions = 0
        total_predictions = 0
        for pred_nodes, actual_node in zip(predictions, actual):
            correct_predictions += len(set(pred_nodes[:k]) & set(actual_node))
            total_predictions += k
        return correct_predictions / total_predictions
    
    def recall_at_k(self, predictions, actual, k):
        correct_predictions = 0
        total_actual = 0
        for pred_nodes, actual_node in zip(predictions, actual):
            correct_predictions += len(set(pred_nodes[:k]) & set(actual_node))
            total_actual += len(actual_node)
        return correct_predictions / total_actual
    
    def precision_score(self,predictions, ground_truths):

        precision_scores = []
        for pred, truths in zip(predictions, ground_truths):
            if len(pred)==0:
                precision = 0.0
                
            else:
                precision = len(set(pred) & set(truths)) / len(pred)
                
            precision_scores.append(precision)

        return sum(precision_scores) / len(predictions)
    
    def recall_score(self,predictions, ground_truths):

        recall_scores = []

        for pred, truths in zip(predictions, ground_truths):
            if len(pred)==0:
                recall = 0.0
                
            else:
                recall = len(set(pred) & set(truths)) / len(truths)
                
            recall_scores.append(recall)

        return sum(recall_scores) / len(predictions)
    
    def f1_score(self,predictions, ground_truths):

        f1_scores = []
        for pred, truths in zip(predictions, ground_truths):
            if len(pred)==0:
                precision = 0.0
                recall = 0.0
            else:
                precision = len(set(pred) & set(truths)) / len(pred)
                recall = len(set(pred) & set(truths)) / len(truths)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)

        return sum(f1_scores) / len(predictions)

    def run_evaluation(self):

        self.hits_1 = self.hits_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=1)
        self.hits_3 = self.hits_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=3)
        self.hits_5 = self.hits_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=5)
        self.mrr = self.reciprocal_rank(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'])

        self.recall_1 = self.recall_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=1)
        self.recall_3 = self.recall_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=3)
        self.recall_5 = self.recall_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=5)
        self.precision_1 = self.precision_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=1)
        self.precision_3 = self.precision_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=3)
        self.precision_5 = self.precision_at_k(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'], k=5)

        self.f1 = self.f1_score(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'])
        self.recall = self.recall_score(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'])
        self.precision = self.precision_score(self.evaluation_results['predicted_answer_nodes'], self.evaluation_results['actual_answer_nodes'])
               
        return self.hits_1, self.hits_3, self.hits_5, self.mrr, self.precision_1, self.precision_3 ,self.precision_5, self.recall_1, self.recall_3, self.recall_5


