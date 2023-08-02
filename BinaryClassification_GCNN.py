from torch_geometric.data import Data
from DataBuilder import DataBuilder
from constants import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    EMBEDDINGS_PATH,
)
import torch
import torch.nn.functional as F
from GNN import GCN, evaluate_model
from loguru import logger

## CREATE DATA
logger.info("Creating data.")

data_builder = DataBuilder(
    triples_path=TRIPLES_PATH,
    entities_labels_path=ENTITIES_LABELS_PATH,
    properties_labels_path=PROPERTIES_LABELS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
    labeler=(lambda x: 1 if "APQC" in x else 0),
)

x = data_builder.get_x()
edge_index = data_builder.get_edge_index()
y = data_builder.get_y()
train_mask, val_mask, test_mask = data_builder.get_masks()

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask,
)

## TRAIN GNN
logger.info("Training GNN.")

NUM_EPOCHS = 1000

model = GCN(
    num_node_features=data.num_node_features, num_hidden_layers=16, num_classes=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

## EVALUATE
logger.info("Evaluating.")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1: {F1:.4f}")
