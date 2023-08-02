import torch
import pandas as pd


class DataBuilder:
    def __init__(
        self,
        triples_path,
        entities_labels_path,
        properties_labels_path,
        embeddings_path,
    ):
        self.triples_data = pd.read_csv(triples_path)
        self.entities_data = pd.read_csv(entities_labels_path)
        mask = self.entities_data.iloc[:, 1].isna()
        self.entities_data.loc[mask, "label"] = (
            self.entities_data[self.entities_data.iloc[:, 1].isna()]
            .iloc[:, 0]
            .apply(lambda x: x.split("/")[-1])
        )  # deal with missing labels
        self.properties_data = pd.read_csv(properties_labels_path)
        self.entity_to_id = {
            string: index
            for index, string in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        self.properties_to_id = {
            string: index
            for index, string in enumerate(self.properties_data.iloc[:, 0].to_list())
        }
        # self.llm_encoder = llm_encoder

    def get_x(self):
        """
        Return node feature vectors.
        """
        embeddings = self.entities_data["label"].apply(self.llm_encoder.encode)
        return torch.tensor(embeddings)

    def get_edge_index(self):
        """
        Return edge index list.
        """
        subjects = self.triples_data.iloc[:, 0].map(self.entity_to_id)
        objects = self.triples_data.iloc[:, 2].map(self.entity_to_id)
        return torch.stack((torch.tensor(subjects), torch.tensor(objects)))
