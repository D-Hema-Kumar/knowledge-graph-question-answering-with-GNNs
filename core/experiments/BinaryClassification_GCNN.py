from torch_geometric.data import Data
from core.ToTorch.DataBuilder import DataBuilder
from config.config import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    GRAPH_EMBEDDINGS_PATH,
    NUM_EPOCHS,
)
import torch
import torch.nn.functional as F
from NeuralNet.GNN import GCN, MLP, evaluate_model
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

## CREATE DATA
logger.info("Creating Data object")

data_builder = DataBuilder(
    triples_path=TRIPLES_PATH,
    entities_labels_path=ENTITIES_LABELS_PATH,
    properties_labels_path=PROPERTIES_LABELS_PATH,
    embeddings_path=GRAPH_EMBEDDINGS_PATH,
    labeler=(lambda x: 1 if "APQC" in x else 0),
)

x = data_builder.get_x()
edge_index = data_builder.get_edge_index()
y = data_builder.get_y()
train_mask, val_mask, test_mask = data_builder.get_entities_masks()

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)

## TRAIN MLP
logger.info("Training MLP")
model = MLP(
    num_node_features=data.num_node_features, dim_hidden_layer=16, num_classes=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

## EVALUATE
logger.info("Evaluating")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}")

## TRAIN GNN
logger.info("Training GNN")

model = GCN(
    num_node_features=data.num_node_features, dim_hidden_layer=16, num_classes=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

## EVALUATE
logger.info("Evaluating")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}")
