import torch
from core.ToTorch.DataBuilder import NodeTypeDataBuilder
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
from core.NeuralNet.GNN import GCN
import torch.nn.functional as F
from config.config import EXPERIMENT_RESULTS_PATH
import datetime
import os

class MultiClassificationExperiment:
    def __init__(self, data_context, training_context, labeler, model_type: MLP):
        data_builder = NodeTypeDataBuilder(
            triples_path=data_context.triples_path,
            entities_labels_path=data_context.entities_labels_path,
            properties_labels_path=data_context.properties_labels_path,
            embeddings_path=data_context.graph_embeddings_path,
            labeler=labeler,
        )
        x = data_builder.get_x()
        edge_index = data_builder.get_edge_index()
        y = data_builder.get_y()
        train_mask, val_mask, test_mask = data_builder.get_entities_masks()
        self.data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        self.training_context = training_context
        self.model_type = model_type
        self.experiment_results_folder_path = os.path.join(
            EXPERIMENT_RESULTS_PATH["multi_class_classification"],
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    def train(self):
        logger.info("Training")
        self.model = self.model_type(
            num_node_features=self.data.num_node_features,
            dim_hidden_layer=self.training_context.dim_hidden_layer,
            num_layers=self.training_context.num_layers,
            num_classes=len(set(self.data.y.tolist())),
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_context.learning_rate,
            weight_decay=5e-4,
        )
        self.model.train()
        for epoch in range(self.training_context.num_epochs):
            optimizer.zero_grad()
            out,_ = self.model(self.data)
            loss = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask]
            )
            loss.backward()
            optimizer.step()
            if epoch%100==0:
                logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        
    def eval(self):
        logger.info("Evaluating")
        self.model.eval()
        pred_, _ = self.model(self.data)
        pred = pred_.argmax(dim=1)
        correct_predictions_mask = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
        incorrect_predictions_mask = pred[self.data.test_mask] != self.data.y[self.data.test_mask]
        self.accuracy = sum(correct_predictions_mask)/len(correct_predictions_mask)
        logger.info(f"{self.model_type} accuracy: {'{:.2%}'.format(self.accuracy.item())}")

    def get_new_embeddings(self):
        '''to get the new embedings after training'''
        self.model.eval()
        with torch.no_grad():
            _,self.experiment_model_embeddings = self.model(self.data)
        return self.experiment_model_embeddings
    
    def save_model(self):
        '''This function saves the model, its weights
          the embeddings and the experiment results '''
        logger.info("Saving Model")

        #make folder
        os.makedirs(self.experiment_results_folder_path, exist_ok=True)
        #model
        torch.save(self.model, os.path.join(self.experiment_results_folder_path,"model.pt"))
        #model weights
        torch.save(self.model.state_dict(),os.path.join(self.experiment_results_folder_path,"weights.pt"))
        #embeddings
        torch.save(self.get_new_embeddings(), os.path.join(self.experiment_results_folder_path,"embeddings.pt"))
        #evaluation results -- TO DO
