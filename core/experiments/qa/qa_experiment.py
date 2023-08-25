import torch
import numpy as np
import pandas as pd
from core.ToTorch.DataBuilder import QAMaskBuilder
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN, RGCN
import torch.nn.functional as F
from config.config import EXPERIMENT_RESULTS_PATH
import datetime
import os
from tqdm import tqdm

class QAExperiment:
    def __init__(self, data_context, training_context, model_type: GCN):
        self.qa_data_builder = QAMaskBuilder(
            triples_path=data_context.triples_path,
            entities_labels_path=data_context.entities_labels_path,
            properties_labels_path=data_context.properties_labels_path,
            embeddings_path=data_context.graph_embeddings_path,
            questions_concepts_answers_path = data_context.questions_concepts_answers_path,
            questions_embeddings_path = data_context.questions_embeddings_path
            )
        
        self.data = self.qa_data_builder.build_data()
        self.training_context = training_context
        self.model_type = model_type
        self.experiment_results_folder_path = os.path.join(
            EXPERIMENT_RESULTS_PATH["qa"],
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        
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

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print("device type: ", self.device)

        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.training_context.num_epochs):
            self.shuffled_indices = np.random.permutation(self.qa_data_builder.question_concepts_answers.index)
            for idx in tqdm(self.shuffled_indices):
                row = self.qa_data_builder.question_concepts_answers.loc[idx]
                q_embedding = self.qa_data_builder.questions_to_embeddings[row["question"]]
                q_x = self.qa_data_builder.get_x(to_concat=q_embedding)
                q_edge_mask, q_nodes, q_concept_mask, q_answer_mask, q_answer_and_random_nodes_mask =self.qa_data_builder.get_concepts_and_masks_for_question(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"])
                q_edge_index = self.data.edge_index[:,q_edge_mask]
                q_edge_type = self.data.edge_type[q_edge_mask]
                q_training_x_mask = self.qa_data_builder.get_question_training_mask_for_x()
                q_y_labels = self.qa_data_builder.get_question_y_labels()
                q_data = Data(x=q_x, edge_index=q_edge_index, edge_type=q_edge_type, train_mask=q_training_x_mask, y=q_y_labels)
                q_data = q_data.to(self.device)
                print(f'Training for Q {idx} : {row["question"]}')
                
                optimizer.zero_grad()
                out,embedding = self.model(q_data)
                loss = F.nll_loss(out[q_data.train_mask], q_data.y[q_data.train_mask],weight = torch.tensor([1.0,20.0]).to(self.device))
                loss.backward()
                optimizer.step()
                #break;
            if epoch%5==0:
                    logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
            #break;
    def eval(self):
        logger.info("Evaluating")
        self.model.eval()
        
        with torch.no_grad():
            res = []
            for idx in tqdm(self.shuffled_indices):

                row = self.qa_data_builder.question_concepts_answers.loc[idx]
                q_embedding = self.qa_data_builder.questions_to_embeddings[row["question"]]
                q_x = self.qa_data_builder.get_x(to_concat=q_embedding)
                q_edge_mask, q_nodes, q_concept_mask, q_answer_mask, q_answer_and_random_nodes_mask =self.qa_data_builder.get_concepts_and_masks_for_question(question =row["question"], concept_uri= row["concepts"], answer_uri= row["answers"])
                q_edge_index = self.data.edge_index[:,q_edge_mask]
                q_edge_type = self.data.edge_type[q_edge_mask]
                q_training_x_mask = self.qa_data_builder.get_question_training_mask_for_x()
                q_y_labels = self.qa_data_builder.get_question_y_labels()
                q_data = Data(x=q_x,edge_index=q_edge_index,edge_type=q_edge_type,train_mask =q_training_x_mask,y=q_y_labels)
                q_data = q_data.to(self.device)
                print(f'Predicting for Q {idx} : {row["question"]}')
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
        logger.debug("Evaluation results saved.")

    
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
