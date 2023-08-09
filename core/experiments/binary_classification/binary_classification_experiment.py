import torch
from core.ToTorch.DataBuilder import DataBuilder
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
import torch.nn.functional as F


class BinaryClassificationExperiment:
    def __init__(self, data_context, training_context, labeler, model_type: MLP):
        data_builder = DataBuilder(
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

    def train(self):
        logger.info("Training")
        self.model = self.model_type(
            num_node_features=self.data.num_node_features,
            dim_hidden_layer=self.training_context.dim_hidden_layer,
            num_classes=2,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_context.learning_rate,
            weight_decay=5e-4,
        )
        self.model.train()
        #print("Type of model:",type(self.model))
        for epoch in range(self.training_context.num_epochs):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask]
            )
            loss.backward()
            optimizer.step()
            logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")
        #model=self.model
        #return model

    def evaluate(self):
        logger.info("Evaluating")

        self.model.eval()
        pred = self.model(self.data).argmax(dim=1)
        correct_predictions_mask = pred[self.data.test_mask] == self.data.y[self.data.test_mask]
        incorrect_predictions_mask = pred[self.data.test_mask] != self.data.y[self.data.test_mask]
        TP = sum(pred[self.data.test_mask][correct_predictions_mask] == 1)
        FP = sum(pred[self.data.test_mask][incorrect_predictions_mask] == 1)
        FN = sum(pred[self.data.test_mask][incorrect_predictions_mask] == 0)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return precision, recall, F1
     
