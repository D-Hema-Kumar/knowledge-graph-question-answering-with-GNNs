import csv
import torch
import pandas as pd


class DataBuilder:
    def __init__(self, triples_path, entities_path, relations_path, llm_encoder):
        self.triples_data = pd.read_csv(triples_path)
        self.entities_data = pd.read_csv(entities_path)
        self.relations_data = pd.read_csv(relations_path)
        self.entity_to_id = {
            string: index
            for index, string in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        self.relations_to_id = {
            string: index
            for index, string in enumerate(self.relations_data.iloc[:, 0].to_list())
        }

    def get_x(self):
        """
        Return node feature vectors
        """
        pass

    def get_edge_index(self):
        """
        Return edge index list
        """
        subjects = self.triples_data.iloc[:, 0].map(self.entity_to_id)
        objects = self.triples_data.iloc[:, 2].map(self.entity_to_id)
        return torch.stack((torch.tensor(subjects), torch.tensor(objects)))
