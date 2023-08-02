import torch
import pandas as pd
import numpy as np
import os


class DataBuilder:
    def __init__(
        self,
        triples_path,
        entities_labels_path,
        properties_labels_path,
        embeddings_path,
        labeler=None,
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
        self.entities_label_to_embeddings = {}
        loaded_data = np.load(
            os.path.join(embeddings_path, "entities.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.entities_label_to_embeddings[key] = loaded_data[key]
        self.properties_label_to_embeddings = {}
        loaded_data = np.load(
            os.path.join(embeddings_path, "properties.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.properties_label_to_embeddings[key] = loaded_data[key]
        self.labeler = labeler

    def get_x(self):
        """
        Return node feature vectors.
        """
        embeddings = (
            self.entities_data["label"].map(self.entities_label_to_embeddings).to_list()
        )
        return torch.tensor(embeddings)

    def get_y(self):
        """
        Return ground truth labels vector.
        """
        if self.labeler:
            return torch.tensor(list(map(self.labeler, self.entities_data["uri"])))
        raise Exception("No labeler defined.")

    def get_edge_index(self):
        """
        Return edge index list.
        """
        entity_to_id = {
            string: index
            for index, string in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        subjects = self.triples_data.iloc[:, 0].map(entity_to_id)
        objects = self.triples_data.iloc[:, 2].map(entity_to_id)
        return torch.stack(
            (
                torch.tensor(subjects, dtype=torch.long),
                torch.tensor(objects, dtype=torch.long),
            )
        )

    def get_masks(self, percentage_train=0.7, percentage_val=0.2, percentage_test=0.1):
        """
        Return train, test and validation masks for nodes.
        """
        train_val_test = [0, 1, 2]
        percentages = [percentage_train, percentage_val, percentage_test]
        splits = np.random.choice(
            train_val_test, len(self.entities_data), p=percentages
        )
        train_mask = splits == 0
        val_mask = splits == 1
        test_mask = splits == 2
        return train_mask, val_mask, test_mask
