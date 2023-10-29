import torch
import pandas as pd
import numpy as np
import os
from loguru import logger
from tqdm import tqdm

from config.config import ( 
    TRIPLES_PATH_OLD,
    ENTITIES_LABELS_PATH_OLD,
    PROPERTIES_LABELS_PATH_OLD,
    PROPERTIES_LABELS_PATH, 
    QUESTIONS_ANSWERS_PATH, 
    GRAPH_EMBEDDINGS_PATH, 
    ENTITIES_LABELS_PATH,
    QUESTIONS_CONCEPTS_ANSWERS_PATH,
    MetaQA_CONFIG,
    KG_EMBEDDINGS_PATH,
    METAQA_KG_EMBEDDINGS_PATH
)
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import ComplEx, DistMult, TransE

KGE_MODEL_MAP = {'transe':TransE,
                 'complex':ComplEx,
                 'distmult': DistMult
                 }



class KGEncoder:

    '''Class to generate KG embeddings based on a given model 
    (TransE, ComplEx, DistMult)'''

    def __init__(
            self, 
            triples_path,
            entities_labels_path,
            properties_labels_path, 
            is_vad_kb=True) -> None:

        # Read from the triples file
        self.is_vad_kb = is_vad_kb
        if is_vad_kb: # if it is VAD KG
            self.triples_data = pd.read_csv(triples_path,header=None)
            self.entities_data = pd.read_csv(entities_labels_path)
            mask = self.entities_data.iloc[:, 1].isna()
            self.entities_data.loc[mask, "label"] = (
                self.entities_data[self.entities_data.iloc[:, 1].isna()]
                .iloc[:, 0]
                .apply(lambda x: x.split("/")[-1])
            )  # deal with missing labels
        
        else: # MetaQA's triples and entities files come with separate delimiters
            self.triples_data = pd.read_csv(triples_path, delimiter='|', header=None)
            self.entities_data = pd.read_csv(entities_labels_path, delimiter='\t')

        self.properties_data = pd.read_csv(properties_labels_path)

        # Build Index for entities and relations

        self.entity_to_index = {
            entity: index
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        self.index_to_entity = {
            index: entity
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        if is_vad_kb:
            self.index_to_entity_label = {
            index: entity_label
            for index, entity_label in enumerate(self.entities_data.iloc[:, 1].to_list())
            }
        else:
            self.index_to_entity_label = self.index_to_entity


        self.property_to_index = {
                string: index
                for index, string in enumerate(self.properties_data.iloc[:, 0].to_list())
            }
        # Set_up device
        self.setup_device()

        # Load data
        self.edge_index = self.get_edge_index()
        self.edge_type = self.get_edge_type()
        self.data = Data(edge_index=self.edge_index, 
                         edge_type=self.edge_type,
                         num_nodes = len(self.entity_to_index),
                         num_edge_types = len(self.entity_to_index)).to(self.device)
        



    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info(f'Device type: {self.device}')

    
    def get_edge_type(self):
        """
        Return edge type list.
        """
        properties = self.triples_data.iloc[:, 1].map(self.property_to_index)
        if self.is_vad_kb:
            return torch.tensor(properties, dtype=torch.long)
        
        else: # if dataset is MetaQA, add the inverse relation types because they don't have them in the KB

            inverse_properties = np.array(properties) + len(self.property_to_index)
            
            return torch.tensor(properties.tolist() + inverse_properties.tolist(), dtype=torch.long)


    def get_edge_index(self):
        """
        Return edge index list and edge type list.
        """
        if self.is_vad_kb:

            subjects = self.triples_data.iloc[:, 0].map(self.entity_to_index)
            objects = self.triples_data.iloc[:, 2].map(self.entity_to_index)

        else: #'adding inverse connections for MetaQA'
            subjects_ = self.triples_data.iloc[:, 0].map(self.entity_to_index).tolist()
            objects_ = self.triples_data.iloc[:, 2].map(self.entity_to_index).tolist()
            subjects = subjects_ + objects_
            objects =  objects_ + subjects_

        return torch.stack(
            (
                torch.tensor(subjects, dtype=torch.long),
                torch.tensor(objects, dtype=torch.long),
            )
        )
    
    def initiate_model(self,dim):

        # Load KGE model

        self.model = KGE_MODEL_MAP[self.model_name](
            num_nodes=self.data.num_nodes,
            num_relations=self.data.num_edge_types,
            hidden_channels=dim

        ).to(self.device)

        # Load optimizer
        OPTIMIZER_MAP = {
                'transe': optim.Adam(self.model.parameters(), lr=0.01),
                'complex': optim.Adagrad(self.model.parameters(), lr=0.001, weight_decay=1e-6),
                'distmult': optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-6),
    
            }

        

    def train(self):

        #Train
        self.model.train()
        total_loss = total_examples = 0
        for head_index, rel_type, tail_index in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(head_index, rel_type, tail_index)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()
        return total_loss / total_examples
    
    def train_model_and_generate_encodings(self,model_name, save_encodings_to_path,dim=50, epochs = 10, lr = 0.001):

        self.model_name = model_name
        self.initiate_model(dim=dim)
        print(f'Model: {model_name}')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #initiate loader
        self.loader = self.model.loader(
            head_index=self.data.edge_index[0],
            rel_type=self.data.edge_type,
            tail_index=self.data.edge_index[1],
            batch_size=1000,
            shuffle=True
        )

        for epoch in range(1, epochs+1):
            loss = self.train()
            if epoch%5==0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        
        encodings = {}
        self.model.eval()
        for index in range(self.data.num_nodes):
            encodings[self.index_to_entity[index]] = self.model.node_emb(torch.tensor(index)).detach().numpy()

        self.save_encodings(encodings,save_encodings_to_path)

    
    def save_encodings(self, encodings, file_path):
        """
        Save encodings to disk.
        """
        logger.info("Saving encodings.")
        base_path = os.path.join(file_path,self.model_name)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        np.savez(os.path.join(base_path, "entities"), **encodings)


        
if __name__ == "__main__":
    # VAD Data
    '''
    kgencoder = KGEncoder(triples_path = TRIPLES_PATH_OLD,
                           entities_labels_path = ENTITIES_LABELS_PATH_OLD,
                           properties_labels_path = PROPERTIES_LABELS_PATH_OLD, 
            is_vad_kb=True)
    kgencoder.train_model_and_generate_encodings(model_name = 'distmult',
                                                 save_encodings_to_path = GRAPH_EMBEDDINGS_PATH,
                                                 dim = 100,
                                                 epochs = 30,
                                                 lr = 0.001)
    '''
    #MetaQA Data
    for kge_model in KGE_MODEL_MAP.keys():

        kgencoder = KGEncoder(
                        triples_path=MetaQA_CONFIG['KB_PATH'],
                        entities_labels_path=MetaQA_CONFIG['ENTITIES_LABELS_PATH'],
                        properties_labels_path=MetaQA_CONFIG['PROPERTIES_LABELS_PATH'], 
            is_vad_kb=False
            )
        kgencoder.train_model_and_generate_encodings(model_name = kge_model,
                                                 save_encodings_to_path = MetaQA_CONFIG["GRAPH_EMBEDDINGS_PATH"],
                                                 dim = 100,
                                                 epochs = 30,
                                                 lr = 0.001)

