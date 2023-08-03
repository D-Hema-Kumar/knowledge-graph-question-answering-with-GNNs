from torch_geometric.data import Data
from DataBuilder import QADataBuilder
from constants import (
    TRIPLES_PATH,
    ENTITIES_LABELS_PATH,
    PROPERTIES_LABELS_PATH,
    GRAPH_EMBEDDINGS_PATH,
    QUESTIONS_ANSWERS_PATH,
    QUESTIONS_EMBEDDINGS_PATH,
    NUM_EPOCHS,
)
import torch
import torch.nn.functional as F
from GNN import GCN, MLP, evaluate_model
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")

## CREATE DATA
logger.info("Creating Data object")

qa_data_builder = QADataBuilder(
    triples_path=TRIPLES_PATH,
    entities_labels_path=ENTITIES_LABELS_PATH,
    properties_labels_path=PROPERTIES_LABELS_PATH,
    embeddings_path=GRAPH_EMBEDDINGS_PATH,
    questions_answers=QUESTIONS_ANSWERS_PATH,
    questions_embeddings_path=QUESTIONS_EMBEDDINGS_PATH,
)

x = qa_data_builder.get_x()
edge_index = qa_data_builder.get_edge_index()

## TRAIN MLP
logger.info("Training MLP")
model = MLP(
    num_node_features=(2 * x.shape[1]), num_hidden_layers=16, num_classes=2
)  # we multiply x.shape by two so as to account for question embedding
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

train_mask, test_mask, val_mask = qa_data_builder.get_questions_masks()

for question, embedding in qa_data_builder.questions_embeddings_masked(
    train_mask
):  # call the questions_iterator from the instance
    x = torch.cat(x, torch.from_numpy(embedding), dim=0)
    y = qa_data_builder.get_y(question=question)
    data = Data(x=x, edge_index=edge_index, y=y)

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        logger.debug(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

## EVALUATE
logger.info("Evaluating")
precision, recall, F1 = evaluate_model(model, data)
logger.info(f"Precision: {precision:.4f} --  Recall: {recall:.4f} -- F1: {F1:.4f}")
