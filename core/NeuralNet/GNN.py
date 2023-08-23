import torch
from torch_geometric.nn import GCNConv,MessagePassing
import torch.nn.functional as F
from torch_geometric.data import Data
from loguru import logger
from torch_geometric.nn import GCNConv, RGCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, dim_hidden_layer,num_layers, num_classes):
        super().__init__()
        #self.conv1 = GCNConv(num_node_features, dim_hidden_layer)
        #self.conv2 = GCNConv(dim_hidden_layer, num_classes)
        layers = []
        input_dim, output_dim = num_node_features, dim_hidden_layer

        for _ in range(num_layers-1):
            layers = layers+[GCNConv(in_channels=input_dim,out_channels=output_dim),torch.nn.ReLU(),torch.nn.Dropout(p=0.2)]
            
            input_dim = dim_hidden_layer
        layers = layers+[GCNConv(in_channels=input_dim,out_channels=num_classes)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        embeddings = []
        for layer in self.layers:
            if isinstance(layer,MessagePassing):
                x = layer(x,edge_index)
                embeddings.append(x.clone())
            else:
                x= layer(x)
            
        return F.log_softmax(x, dim=1), embeddings
    
class RGCN(torch.nn.Module):

    def __init__(self, num_node_features, dim_hidden_layer, num_relations, num_layers, num_classes):
        super().__init__()
        
        layers = []
        input_dim, output_dim = num_node_features, dim_hidden_layer

        for _ in range(num_layers-1):
            layers = layers+[RGCNConv(in_channels=input_dim,out_channels=output_dim, num_relations=num_relations),torch.nn.ReLU(),torch.nn.Dropout(p=0.2)]
            
            input_dim = dim_hidden_layer
        layers = layers+[RGCNConv(in_channels=input_dim,out_channels=num_classes, num_relations=num_relations)]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        embeddings = []
        for layer in self.layers:
            if isinstance(layer,MessagePassing):
                x = layer(x,edge_index,edge_type)
                embeddings.append(x.clone())
            else:
                x= layer(x)
            
        return F.log_softmax(x, dim=1), embeddings



def _predict_answer(model, data):
    """
    Returns the predicted answer and node index.
    """
    pred, _ = model(data)
    return pred.max(dim=1)[0].argmax().item()


def evaluate_qa_model(model, qa_data_builder, mask):
    model.eval()
    correct_predictions = 0
    for q_index, question_embedding in enumerate(
        qa_data_builder.questions_embeddings_masked(mask)
    ):
        question, q_embedding = question_embedding
        q_x = qa_data_builder.get_x(
            to_concat=q_embedding
        )  # adding the question embedding to the node embeddings
        q_y = qa_data_builder.get_y(question=question)
        data = Data(x=q_x, edge_index=qa_data_builder.get_edge_index(), y=q_y)
        pred_node_idx = _predict_answer(model, data)
        actual_node_idx = qa_data_builder.get_node_index_for_question_answer(question)
        if pred_node_idx == actual_node_idx:
            logger.debug(f"Correctly predicted answer to question {question}.")
            correct_predictions += 1
        elif pred_node_idx != torch.tensor(0):
            logger.debug(
                f"Question: {question}. Predicted answer = {qa_data_builder.index_to_entity[pred_node_idx]}, Actual answer: {qa_data_builder.index_to_entity[actual_node_idx]}"
            )
        else:
            logger.debug(f"Could not predict any answer")
    return correct_predictions / (q_index + 1)
