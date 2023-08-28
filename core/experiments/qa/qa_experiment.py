import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core.ToTorch.DataBuilder import QAMaskBuilder
from core.experiments.utils import (QAEvaluationMetrcis)
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN, RGCN
import torch.nn.functional as F
from config.config import (EXPERIMENT_RESULTS_PATH, EXPERIMENT_TYPES_PATH)
import datetime
import os
from tqdm import tqdm
import csv

class QAExperiment:
    def __init__(self, data_context, training_context, model_type: GCN):

        self.training_context = training_context
        self.data_context = data_context
        self.model_type = model_type
        self.experiment_results_folder_path = os.path.join(
            EXPERIMENT_RESULTS_PATH["qa"],
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        self.qa_data_builder = QAMaskBuilder(
                                        triples_path=data_context.triples_path,
                                        entities_labels_path=data_context.entities_labels_path,
                                        properties_labels_path=data_context.properties_labels_path,
                                        embeddings_path=data_context.graph_embeddings_path,
                                        training_questions_concepts_answers_file_path = data_context.training_questions_concepts_answers_file_path,
                                        testing_questions_concepts_answers_file_path = data_context.testing_questions_concepts_answers_file_path,
                                        questions_embeddings_path = data_context.questions_embeddings_path
                                        )
        self.data = self.qa_data_builder.build_data()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print("device type: ", self.device)
        
    
    def train(self):
        logger.info("Training")

        if self.model_type.__name__ == 'GCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features*2,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )
        elif self.model_type.__name__ == 'RGCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features*2,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_relations=len(set(self.data.edge_type.tolist())),
                num_bases = self.training_context.num_bases,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_context.learning_rate,
            weight_decay=5e-4,
        )

        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.training_context.num_epochs):
            shuffled_train_indices = np.random.permutation(self.qa_data_builder.training_questions_concepts_answers.index.tolist())
            for idx in tqdm(shuffled_train_indices):
                row = self.qa_data_builder.training_questions_concepts_answers.loc[idx]
                q_embedding = self.qa_data_builder.questions_to_embeddings[row["question"]]
                q_x = self.qa_data_builder.get_x(to_concat=q_embedding)
                q_edge_mask, q_nodes, q_concept_mask, q_answer_mask, q_answer_and_random_nodes_mask =self.qa_data_builder.get_concepts_and_masks_for_question(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"])
                q_edge_index = self.data.edge_index[:,q_edge_mask]
                q_edge_type = self.data.edge_type[q_edge_mask]
                q_training_x_mask = self.qa_data_builder.get_question_training_mask_for_x()
                q_y_labels = self.qa_data_builder.get_question_y_labels()
                q_data = Data(x=q_x, edge_index=q_edge_index, edge_type=q_edge_type, train_mask=q_training_x_mask, y=q_y_labels)
                q_data = q_data.to(self.device)
                logger.debug(f'Training for Q {idx} : {row["question"]}')
                
                optimizer.zero_grad()
                out,embedding = self.model(q_data)
                loss = F.nll_loss(out[q_data.train_mask], q_data.y[q_data.train_mask],weight = torch.tensor([1.0,20.0]).to(self.device))
                loss.backward()
                optimizer.step()
                #break;
            if epoch%1==0:
                    logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            #break;
    def eval(self):
        logger.info("Evaluating")
        self.model.eval()
        
        with torch.no_grad():
            res = []
            for idx, row in tqdm(self.qa_data_builder.testing_questions_concepts_answers.iterrows()):

                q_embedding = self.qa_data_builder.questions_to_embeddings[row["question"]]
                q_x = self.qa_data_builder.get_x(to_concat=q_embedding)
                q_edge_mask, q_nodes, q_concept_mask, q_answer_mask, q_answer_and_random_nodes_mask =self.qa_data_builder.get_concepts_and_masks_for_question(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"])
                q_edge_index = self.data.edge_index[:,q_edge_mask]
                q_edge_type = self.data.edge_type[q_edge_mask]
                q_training_x_mask = self.qa_data_builder.get_question_training_mask_for_x()
                q_y_labels = self.qa_data_builder.get_question_y_labels()
                q_data = Data(x=q_x,edge_index=q_edge_index,edge_type=q_edge_type,train_mask =q_training_x_mask,y=q_y_labels)
                q_data = q_data.to(self.device)
                logger.debug(f'Predicting for Q {idx} : {row["question"]}')
                out,_ = self.model(q_data)
                predicted_answer_nodes = torch.where(out.argmax(dim=1))[0]
                predicted_answer_node_probabilities = out.max(dim=1)[0][predicted_answer_nodes]
                sorted_probability_indices = torch.argsort(predicted_answer_node_probabilities, descending= True)
                count_predicted_nodes =len(predicted_answer_nodes)
                actual_answer_nodes = q_nodes[q_answer_mask].tolist()
                if count_predicted_nodes > 0 and count_predicted_nodes < 50 :
                    logger.debug(f"answers predicted")
                    is_predicted_in_actual_answers = bool(set(actual_answer_nodes) & set(predicted_answer_nodes[sorted_probability_indices].tolist()))
                    res.append((idx, actual_answer_nodes, predicted_answer_nodes[sorted_probability_indices].tolist(),predicted_answer_node_probabilities[sorted_probability_indices].tolist(),count_predicted_nodes,is_predicted_in_actual_answers))
                
                elif count_predicted_nodes >= 50:
                    res.append((idx, actual_answer_nodes, np.nan,np.nan,count_predicted_nodes,False))
                else:
                    logger.debug(f"NO answers found")
                    res.append((idx, actual_answer_nodes, np.nan,np.nan,0,False))
                
                #break;
        eval_res = pd.DataFrame.from_records(res,columns=["q_idx","actual_answer_nodes","predicted_answer_nodes","probabilities_of_answer_nodes","count_predicted_nodes","is_predicted_in_actual"])
        eval_res.to_csv(os.path.join(self.experiment_results_folder_path,"evaluation_results.csv"),index=False)
        logger.info("Evaluation results saved.")

    
    def save_model(self):
        '''This function saves the model, its weights
          the embeddings and the experiment results '''
        logger.info("Saving Model")

        #make folder
        os.makedirs(self.experiment_results_folder_path, exist_ok=True)
        #model
        torch.save(self.model, os.path.join(self.experiment_results_folder_path,f'{self.model_type.__name__}model.pt'))
        #model weights
        torch.save(self.model.state_dict(),os.path.join(self.experiment_results_folder_path,f'{self.model_type.__name__}weights.pt'))
        logger.info("Model data saved.")
    
    def save_qa_experiment_metadata(self):

        # Define the CSV field names for training context and evaluation results
        fieldnames = ["time_stamp","info","Epochs", "Learning Rate", "hidden_dimension","num_layers","num_bases", "Model",
                      "hits@1", "hits@3", "hits@5", "mrr", "precision@1", "precision@3" ,"precision@5", "recall@1", 
                      "recall@3", "recall@5","model_directory","triples_path","entities_labels_path",
                    "properties_labels_path","graph_embeddings_path"]
        
        file_path = os.path.join(EXPERIMENT_TYPES_PATH["qa"], "qa_experiments_masterdata.csv")
        
        qa_experiment_metadata = {

            "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "info":self.training_context.info,

            #data context
            "triples_path" : self.data_context.triples_path,
            "entities_labels_path" : self.data_context.entities_labels_path,
            "properties_labels_path" : self.data_context.properties_labels_path,
            "graph_embeddings_path" : self.data_context.graph_embeddings_path,

            # training context
            "Epochs": self.training_context.num_epochs,
            "Learning Rate": self.training_context.learning_rate,
            "hidden_dimension": self.training_context.dim_hidden_layer,
            "num_layers":self.training_context.num_layers,
            "num_bases":self.training_context.num_bases,

            "Model": self.model,
            "model_directory":self.experiment_results_folder_path,

            # QA evaluation results
            "hits@1": self.hits_1,
            "hits@3": self.hits_3,
            "hits@5": self.hits_5,
            "mrr":self.mrr,
            "precision@1": self.precision_1,
            "precision@3": self.precision_3,
            "precision@5": self.precision_5,
            "recall@1": self.recall_1, 
            "recall@3": self.recall_3,
            "recall@5": self.recall_5
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
            writer.writerow(qa_experiment_metadata)
        logger.info('QA experiment settings appended.')

    def get_evaluation_metrics(self):

        evaluation_metrcis = QAEvaluationMetrcis(self.experiment_results_folder_path)
        self.hits_1, self.hits_3, self.hits_5, self.mrr, self.precision_1, self.precision_3 ,self.precision_5, self.recall_1, self.recall_3, self.recall_5 = evaluation_metrcis.run_evaluation()
        logger.info(f'hits@1 : {np.round(self.hits_1,2)} -- hits@3 : {np.round(self.hits_3,2)} -- hits@5 : {np.round(self.hits_5,2)} -- MRR : {np.round(self.mrr,2)}')
    

    def run(self):
        '''train, evaluate, save model and experiment results.'''

        self.train()
        self.save_model()
        self.eval()
        self.get_evaluation_metrics()
        self.save_qa_experiment_metadata()