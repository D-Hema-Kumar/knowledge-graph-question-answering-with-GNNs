import torch
import pandas as pd
import numpy as np
import os
from loguru import logger
import random


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
        self.entity_to_index = {
            entity: index
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }
        self.index_to_entity = {
            index: entity
            for index, entity in enumerate(self.entities_data.iloc[:, 0].to_list())
        }

    def get_x(self, to_concat=None):
        """
        Return node feature vectors, optionally with additional features passed via to_concat.
        """
        embeddings = np.array(
            self.entities_data["label"].map(self.entities_label_to_embeddings).to_list()
        )
        x = torch.tensor(embeddings)
        if to_concat is None:
            return x
        return torch.cat(
            (
                x,
                torch.from_numpy(to_concat.reshape(1, -1).repeat(x.shape[0], axis=0)),
            ),
            dim=1,
        )  # broadcast the question emb to match the x dim

    def get_y(self):
        """
        Return ground truth labels vector.
        """
        return torch.tensor(list(map(self.labeler, self.entities_data["uri"])))

    def get_edge_type(self):
        """
        Return edge type list.
        """
        property_to_id = {
            string: index
            for index, string in enumerate(self.properties_data.iloc[:, 0].to_list())
        }
        properties = self.triples_data.iloc[:, 1].map(property_to_id)
        return torch.tensor(properties, dtype=torch.long)

    def get_edge_index(self):
        """
        Return edge index list and edge type list.
        """
        subjects = self.triples_data.iloc[:, 0].map(self.entity_to_index)
        objects = self.triples_data.iloc[:, 2].map(self.entity_to_index)
        return torch.stack(
            (
                torch.tensor(subjects, dtype=torch.long),
                torch.tensor(objects, dtype=torch.long),
            )
        )

    def _get_masks(
        self,
        size,
        percentage_train,
        percentage_val,
        percentage_test,
    ):
        """
        Return train, test and validation masks.
        """
        train_val_test = [0, 1, 2]
        percentages = [percentage_train, percentage_val, percentage_test]
        splits = np.random.choice(train_val_test, size, p=percentages)
        train_mask = splits == 0
        val_mask = splits == 1
        test_mask = splits == 2
        return train_mask, val_mask, test_mask

    def get_entities_masks(
        self, percentage_train=0.7, percentage_val=0.2, percentage_test=0.1
    ):
        return self._get_masks(
            size=len(self.entities_data),
            percentage_train=percentage_train,
            percentage_val=percentage_val,
            percentage_test=percentage_test,
        )


class QADataBuilder(DataBuilder):
    def __init__(
        self,
        triples_path,
        entities_labels_path,
        properties_labels_path,
        embeddings_path,
        questions_answers_path,
        questions_embeddings_path,
        labeler=None,
    ):
        super().__init__(
            triples_path,
            entities_labels_path,
            properties_labels_path,
            embeddings_path,
            labeler,
        )
        self.question_to_answers = (
            pd.read_csv(questions_answers_path)
            .set_index("question")
            .to_dict()["answer_uri"]
        )
        self.questions_to_embeddings = {}
        loaded_data = np.load(
            os.path.join(questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.questions_to_embeddings[key] = loaded_data[key]

        def labeler(**kwargs):
            try:
                question = kwargs["question"]
            except:
                logger.error("You need to call get_y with a question.")
                raise Exception("You need to call get_y with a question.")
            return lambda x: 1 if self.question_to_answers[question] == x else 0

        self.labeler = labeler

    def get_node_index_for_question_answer(self, question):
        """
        Get the node index of the answer to the given question.
        """
        answer = self.question_to_answers[question]
        return self.entity_to_index[answer]

    def questions_embeddings_masked(self, mask):
        class DictIterator:
            def __init__(self, my_dict, mask):
                self._dict_items = my_dict.items()
                self._iter = iter(self._dict_items)
                self._mask = mask
                self._pos = 0

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    key, value = next(self._iter)
                    if self._mask[self._pos]:
                        self._pos += 1
                        return key, value
                    else:
                        self._pos += 1
                        return self.__next__()
                except StopIteration:
                    raise StopIteration

        return DictIterator(self.questions_to_embeddings, mask)

    def get_questions_masks(
        self, percentage_train=0.7, percentage_val=0.2, percentage_test=0.1
    ):
        return self._get_masks(
            size=len(self.question_to_answers),
            percentage_train=percentage_train,
            percentage_val=percentage_val,
            percentage_test=percentage_test,
        )

    def get_mask_for_nodes_for_question(self, question, size):
        """
        Return mask for nodes of dimension size.
        1 True item in the maks corresponds to answer to the given question
        Remaining (size-1) True items correspond to randomly selected nodes
        Everything else is False.
        """
        answer_node_index = self.get_node_index_for_question_answer(question)
        available_indices = set(range(len(self.entities_data)))
        available_indices.remove(answer_node_index)
        random_nodes = np.random.choice(
            list(available_indices), size - 1, replace=False
        )
        mask = np.zeros(len(self.entities_data), dtype=bool)
        mask[answer_node_index] = True
        mask[random_nodes] = True
        return mask