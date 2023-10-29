import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from core.ToTorch.DataBuilder import QAMaskBuilder, QADataBuilder
from core.experiments.utils import (QAEvaluationMetrcis)
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN, RGCN, RGAT
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
                                        LM_embeddings_path=data_context.LM_embeddings_path,
                                        KG_embeddings_path=data_context.KG_embeddings_path,
                                        training_questions_concepts_answers_file_path = data_context.training_questions_concepts_answers_file_path,
                                        testing_questions_concepts_answers_file_path = data_context.testing_questions_concepts_answers_file_path,
                                        training_questions_embeddings_path = data_context.training_questions_embeddings_path,
                                        testing_questions_embeddings_path = data_context.testing_questions_embeddings_path,
                                        is_vad_kb=data_context.is_vad_kb
                                        )
        self.data = self.qa_data_builder.build_data()
        self.setup_logger()
        self.setup_device()
    
    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f'Device type: {self.device}')

        
    def setup_logger(self):
        logger.add(os.path.join(self.experiment_results_folder_path,'experimentLog.log'),rotation="1 MB", level="DEBUG")
    
    def setup_model(self):

        if self.model_type.__name__ == 'GCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )
        elif self.model_type.__name__ == 'RGCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_relations=len(set(self.data.edge_type.tolist())),
                num_bases = self.training_context.num_bases,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )
        elif self.model_type.__name__ == 'RGAT':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_relations=len(set(self.data.edge_type.tolist())),
                num_bases = self.training_context.num_bases,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )


    def train(self):
        logger.info("Training")

        if self.model_type.__name__ == 'GCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )
        elif self.model_type.__name__ == 'RGCN':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
                dim_hidden_layer=self.training_context.dim_hidden_layer,
                num_relations=len(set(self.data.edge_type.tolist())),
                num_bases = self.training_context.num_bases,
                num_layers=self.training_context.num_layers,
                num_classes=2,
            )
        elif self.model_type.__name__ == 'RGAT':
            self.model = self.model_type(
                num_node_features=self.data.num_node_features+768,
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
        num_train_samples = self.qa_data_builder.training_questions_concepts_answers.shape[0]
        for epoch in range(self.training_context.num_epochs):
            shuffled_train_indices = np.random.permutation(self.qa_data_builder.training_questions_concepts_answers.index.tolist())
            total_epoch_loss = 0.0
            for idx in tqdm(shuffled_train_indices):
                row = self.qa_data_builder.training_questions_concepts_answers.loc[idx]
                q_data = self.qa_data_builder.get_question_data(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"], training=True)
                q_data = q_data.to(self.device)
                
                optimizer.zero_grad()
                out,embedding = self.model(q_data)
                loss = F.nll_loss(out[q_data.train_mask], q_data.y[q_data.train_mask],weight = torch.tensor([1.0,20.0]).to(self.device))
                loss.backward()
                optimizer.step()
                total_epoch_loss +=loss.item()
                #break;
            logger.debug(f"Epoch: {epoch:03d}, Loss: {(total_epoch_loss/num_train_samples):.4f}")
            #break;
    def eval(self):
        logger.info("Evaluating")
        self.model.eval()
        
        with torch.no_grad():
            res = []
            for idx, row in tqdm(self.qa_data_builder.testing_questions_concepts_answers.iterrows()):

                q_data = self.qa_data_builder.get_question_data(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"], training=False)
                q_data = q_data.to(self.device)
                question_subgraph_all_nodes = self.qa_data_builder.question_subgraph_all_nodes.to(self.device)
                question_subgraph_answer_mask = self.qa_data_builder.question_subgraph_answer_mask.to(self.device)
                out,_ = self.model(q_data)
                #During evaluation
                predicted_local_answer_nodes = torch.where(out.argmax(dim=1))[0] # local subgraph answer
                predicted_answer_nodes = question_subgraph_all_nodes[predicted_local_answer_nodes] # remapping local answer to global node number
                predicted_answer_node_probabilities = out.max(dim=1)[0][predicted_local_answer_nodes]
                sorted_probability_indices = torch.argsort(predicted_answer_node_probabilities, descending= True)
                count_predicted_nodes =len(predicted_answer_nodes)
                actual_answer_nodes = self.qa_data_builder.question_specific_answer_nodes.to(self.device).tolist()
                #actual_answer_nodes = question_subgraph_all_nodes[question_subgraph_answer_mask].tolist()
                if count_predicted_nodes > 0 and count_predicted_nodes < 50:
                    #logger.debug(f"answers predicted")
                    is_predicted_in_actual_answers = bool(set(actual_answer_nodes) & set(predicted_answer_nodes[sorted_probability_indices].tolist()))
                    res.append((idx, actual_answer_nodes, predicted_answer_nodes[sorted_probability_indices].tolist(),predicted_answer_node_probabilities[sorted_probability_indices].tolist(),count_predicted_nodes,is_predicted_in_actual_answers))
                
                elif count_predicted_nodes >= 50:
                    res.append((idx, actual_answer_nodes, np.nan,np.nan,count_predicted_nodes,False))
                else:
                    #logger.debug(f"NO answers found")
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
                      "F1","precision","recall","hits@1", "hits@3", "hits@5", "mrr", 
                      "precision@1", "precision@3" ,"precision@5", "recall@1", "recall@3", "recall@5",
                      "model_directory","triples_path","entities_labels_path","properties_labels_path",
                      "LM_embeddings_path", "KG_embeddings_path"]
        
        file_path = os.path.join(EXPERIMENT_TYPES_PATH["qa"], "qa_experiments_masterdata.csv")
        
        qa_experiment_metadata = {

            "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "info":self.training_context.info,

            #data context
            "triples_path" : self.data_context.triples_path,
            "entities_labels_path" : self.data_context.entities_labels_path,
            "properties_labels_path" : self.data_context.properties_labels_path,
            "LM_embeddings_path" : self.data_context.LM_embeddings_path,
            "KG_embeddings_path" : self.data_context.KG_embeddings_path,


            # training context
            "Epochs": self.training_context.num_epochs,
            "Learning Rate": self.training_context.learning_rate,
            "hidden_dimension": self.training_context.dim_hidden_layer,
            "num_layers":self.training_context.num_layers,
            "num_bases":self.training_context.num_bases,

            "Model": self.model,
            "model_directory":self.experiment_results_folder_path,

            # QA evaluation results
            "F1": self.F1,
            "precision": self.precision_score ,
            "recall": self.recall_score,
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
        logger.info('QA experiment metadata appended to masterdata file.')

    def get_evaluation_metrics(self):

        evaluation_metrcis = QAEvaluationMetrcis(self.experiment_results_folder_path)
        self.hits_1, self.hits_3, self.hits_5, self.mrr, self.precision_1, self.precision_3 ,self.precision_5, self.recall_1, self.recall_3, self.recall_5 = evaluation_metrcis.run_evaluation()
        self.F1 = evaluation_metrcis.f1	
        self.precision_score = evaluation_metrcis.precision
        self.recall_score = evaluation_metrcis.recall	
        logger.info(f'F-score : {np.round(self.F1,2)} -- hits@1 : {np.round(self.hits_1,2)} -- hits@3 : {np.round(self.hits_3,2)} -- hits@5 : {np.round(self.hits_5,2)} -- MRR : {np.round(self.mrr,2)}')
    

    def run(self):
        '''train, evaluate, save model and experiment results.'''

        self.train()
        self.save_model()
        self.eval()
        self.get_evaluation_metrics()
        self.save_qa_experiment_metadata()

class MetaQAExperiment(QAExperiment):
    def __init__(self, data_context, training_context, model_type: GCN):

        self.training_context = training_context
        self.data_context = data_context
        self.model_type = model_type
        self.experiment_results_folder_path = os.path.join(
            EXPERIMENT_RESULTS_PATH["qa"],
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        #print('KG Embedding path inside MetaQAExperiment class -- :',data_context.KG_embeddings_path)
        self.qa_data_builder = QADataBuilder(
                                        triples_path=data_context.triples_path,
                                        entities_labels_path=data_context.entities_labels_path,
                                        properties_labels_path=data_context.properties_labels_path,
                                        LM_embeddings_path=data_context.LM_embeddings_path,
                                        KG_embeddings_path=data_context.KG_embeddings_path,
                                        training_questions_concepts_answers_file_path = data_context.training_questions_concepts_answers_file_path,
                                        testing_questions_concepts_answers_file_path = data_context.testing_questions_concepts_answers_file_path,
                                        training_questions_embeddings_path = data_context.training_questions_embeddings_path,
                                        testing_questions_embeddings_path = data_context.testing_questions_embeddings_path,
                                        is_vad_kb=data_context.is_vad_kb,
                                        training_subgraphs_file_path= data_context.training_subgraphs_file_path,
                                        testing_subgraphs_file_path=data_context.testing_subgraphs_file_path,
                                    )
        
        self.data = self.qa_data_builder.build_data()
        self.setup_logger()
        self.setup_device()
        self.setup_model()

    def train(self):
        logger.info("Training")
        
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_context.learning_rate,
            weight_decay=5e-4,
        )

        self.model.train()
        num_train_samples = self.qa_data_builder.training_questions_concepts_answers.shape[0]
        for epoch in range(self.training_context.num_epochs):
            shuffled_train_indices = np.random.permutation(self.qa_data_builder.training_questions_concepts_answers.index.tolist())
            total_epoch_loss = 0.0
            for idx in tqdm(shuffled_train_indices):
                row = self.qa_data_builder.training_questions_concepts_answers.loc[idx]
                q_data = self.qa_data_builder.get_question_data(question=row["question"], training=True)
                q_data = q_data.to(self.device)
                optimizer.zero_grad()
                out,embedding = self.model(q_data)
                loss = F.nll_loss(out[q_data.train_mask], q_data.y[q_data.train_mask],weight = torch.tensor([1.0,20.0]).to(self.device))
                loss.backward()
                optimizer.step()
                total_epoch_loss +=loss.item()
            logger.debug(f"Epoch: {epoch:03d}, Loss: {(total_epoch_loss/num_train_samples):.4f}")
    
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            res = []
            for idx, row in tqdm(self.qa_data_builder.testing_questions_concepts_answers.iterrows()):

                q_data, question_subgraph_nodes,question_subgraph_answer = self.qa_data_builder.get_question_data(question=row["question"], training=False)
                q_data = q_data.to(self.device)
                question_subgraph_nodes = question_subgraph_nodes.to(self.device)
                question_subgraph_answer = question_subgraph_answer.to(self.device).tolist()
                #predict
                out,_ = self.model(q_data)

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
        eval_res.to_csv(os.path.join(self.experiment_results_folder_path,"evaluation_results.csv"),index=False)
        logger.info("Evaluation results saved.")

    def save_MetaQA_experiment_metadata(self):

        # Define the CSV field names for training context and evaluation results
        fieldnames = ["time_stamp","info","Epochs", "Learning Rate", "hidden_dimension","num_layers","num_bases", "Model",
                      "F1","precision","recall","hits@1", "hits@3", "hits@5", "mrr", 
                      "precision@1", "precision@3" ,"precision@5", "recall@1", "recall@3", "recall@5"
                      ,"model_directory","triples_path","entities_labels_path", "properties_labels_path",
                      "LM_embeddings_path", "KG_embeddings_path","training_questions_concepts_answers_file_path",
                    "testing_questions_concepts_answers_file_path"]
        
        file_path = os.path.join(EXPERIMENT_TYPES_PATH["qa"], "MetaQA_experiments_masterdata.csv")
        
        qa_experiment_metadata = {

            "time_stamp":datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
            "info":self.training_context.info,

            #data context
            "triples_path" : self.data_context.triples_path,
            "entities_labels_path" : self.data_context.entities_labels_path,
            "properties_labels_path" : self.data_context.properties_labels_path,
            "LM_embeddings_path" : self.data_context.LM_embeddings_path,
            "KG_embeddings_path" : self.data_context.KG_embeddings_path,
            "training_questions_concepts_answers_file_path":self.data_context.training_questions_concepts_answers_file_path,
            "testing_questions_concepts_answers_file_path":self.data_context.testing_questions_concepts_answers_file_path,

            # training context
            "Epochs": self.training_context.num_epochs,
            "Learning Rate": self.training_context.learning_rate,
            "hidden_dimension": self.training_context.dim_hidden_layer,
            "num_layers":self.training_context.num_layers,
            "num_bases":self.training_context.num_bases,

            "Model": self.model,
            "model_directory":self.experiment_results_folder_path,

            # QA evaluation results

            "F1": self.F1,
            "hits@1": self.hits_1,
            "hits@3": self.hits_3,
            "hits@5": self.hits_5,
            "mrr":self.mrr,

            "precision@1": self.precision_1,
            "precision@3": self.precision_3,
            "precision@5": self.precision_5,
            "recall@1": self.recall_1, 
            "recall@3": self.recall_3,
            "recall@5": self.recall_5,

            "precision": self.precision_score ,
            "recall": self.recall_score

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
        logger.info(f'MetaQA experiment metadata appended to masterdata file {file_path}.')

    def get_evaluation_metrics(self):

        evaluation_metrcis = QAEvaluationMetrcis(self.experiment_results_folder_path)
        self.hits_1, self.hits_3, self.hits_5, self.mrr, self.precision_1, self.precision_3 ,self.precision_5, self.recall_1, self.recall_3, self.recall_5 = evaluation_metrcis.run_evaluation()
        self.F1 = evaluation_metrcis.f1	
        self.precision_score = evaluation_metrcis.precision
        self.recall_score = evaluation_metrcis.recall	
        logger.info(f'F1 : {np.round(self.F1,2)} -- hits@1 : {np.round(self.hits_1,2)} -- hits@3 : {np.round(self.hits_3,2)} -- hits@5 : {np.round(self.hits_5,2)} -- MRR : {np.round(self.mrr,2)}')


    def run(self):
        '''train, evaluate, save model and experiment results.'''

        self.train()
        self.save_model()
        self.eval()
        logger.info(f'MetaQA experiment model and evaluation results saved to: {self.experiment_results_folder_path}.')
        self.get_evaluation_metrics()
        self.save_MetaQA_experiment_metadata()

