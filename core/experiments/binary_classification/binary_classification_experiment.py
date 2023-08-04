import torch
from core.ToTorch.DataBuilder import DataBuilder
from torch_geometric.data import Data
from loguru import logger
from core.NeuralNet.MLP import MLP
import torch.nn.functional as F


class BinaryClassificationExperiment:
    def __init__(self, data_context, training_context, labeler, model_type: MLP):
        data_builder = DataBuilder(
            triples_path=data_context["TRIPLES_PATH"],
            entities_labels_path=data_context["ENTITIES_LABELS_PATH"],
            properties_labels_path=data_context["PROPERTIES_LABELS_PATH"],
            embeddings_path=data_context["GRAPH_EMBEDDINGS_PATH"],
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
            dim_hidden_layer=self.training_context["DIM_HIDDEN_LAYER"],
            num_classes=2,
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_context["LEARNING_RATE"],
            weight_decay=5e-4,
        )
        self.model.train()

        for epoch in range(self.training_context["NUM_EPOCHS"]):
            optimizer.zero_grad()
            out = self.model(self.data)
            loss = F.nll_loss(
                out[self.data.train_mask], self.data.y[self.data.train_mask]
            )
            loss.backward()
            optimizer.step()
            logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    def evaluate(self):
        logger.info("Evaluating")
        self.model.eval()
        precision, recall, F1 = self.model.evaluate(self.data)
        logger.info(
            f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}"
        )
