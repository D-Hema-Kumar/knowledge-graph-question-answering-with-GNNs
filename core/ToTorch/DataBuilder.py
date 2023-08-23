import torch
import pandas as pd
import numpy as np
import os
from loguru import logger
from torch_geometric.data import Data
import random
import ast


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
    
    


class NodeTypeDataBuilder(DataBuilder):
    '''This class is for building the data object necessary for multiclass node type classification task
    '''
    
    def __init__(
        self,
        triples_path,
        entities_labels_path,
        properties_labels_path,
        embeddings_path,
        labeler=None,
    ):
        super().__init__(
            triples_path,
            entities_labels_path,
            properties_labels_path,
            embeddings_path,
            labeler,
        )
    
    def get_y(self):

        #1 get node type records from the triple store (s,p,o)
        triples  = self.triples_data.copy()
        triples.columns = ['subject','predicate','object']
        node_type_triples = triples[triples.predicate=='http://www.w3.org/1999/02/22-rdf-syntax-ns#type'].copy()

        #2 drop the duplicates based on the subject
        node_type_triples.drop_duplicates(subset='subject', keep='first', inplace=True)

        #3 create the object-class ID dictionary and add a column with class ids (object field holds the type information of subject)
        class_id_to_node_type = node_type_triples["object"].unique().tolist()
        node_type_to_class_id = {node : class_id for class_id, node in enumerate(class_id_to_node_type) }
        node_type_triples['class'] = node_type_triples["object"].map(node_type_to_class_id)

        #4 create subject & it's type class id dictionary
        node_to_its_class_label = dict(zip(node_type_triples['subject'].to_list(), node_type_triples['class'].to_list()))

        #5 map the above node type by applying to entities_labels file.
            #a. Nodes that are not in step 4 dict, assign some class and hide them from training/testing/validation
        entities_data = self.entities_data.copy() # copy migh not be needed
        entities_data['class']= entities_data['uri'].apply( lambda x : node_to_its_class_label[x] if x in node_to_its_class_label else len(node_type_to_class_id))
        
        #6 Add the new class to original class-node type list if it is present
        if len(entities_data[entities_data["class"]==len(node_type_to_class_id)]) > 0:
            class_id_to_node_type.append("unknown")
            node_type_to_class_id["unknown"]=len(node_type_to_class_id)

        self.class_id_to_node_type = class_id_to_node_type
        self.node_type_to_class_id = node_type_to_class_id
        return torch.tensor(list(entities_data["class"]))
    
class QAMaskBuilder(DataBuilder):
    '''The class to build the relevant masks given a data objet and questions '''

    def __init__(
    self,
    triples_path,
    entities_labels_path,
    properties_labels_path,
    embeddings_path,
    questions_concepts_answers_path,
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
        self.question_concepts_answers = (
            pd.read_csv(questions_concepts_answers_path)
    
        )
        # reads columns as lists instead of strings
        list_columns = ['concepts', 'answers']
        for col in list_columns:
            self.question_concepts_answers[col] = self.question_concepts_answers[col].apply(ast.literal_eval)
        
        #load question embeddings
        self.questions_to_embeddings = {}
        loaded_data = np.load(
        os.path.join(questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.questions_to_embeddings[key] = loaded_data[key]

        self.x = self.get_x()
        self.edge_index = self.get_edge_index()
        self.edge_type = self.get_edge_type()


    def build_data(self):
        self.data = Data(x=self.x,edge_index=self.edge_index, edge_type=self.edge_type)
        return self.data


    def get_concepts_and_masks_for_question(self, question:str, concept_uri:list, answer_uri:list):
        '''
        given a question, q_node, a_node : builds a list of subgraph concepts and its related masks:
        e.g. 
        question_subgraph_concepts:  tensor([0, 1, 2, 3, 4])
        question_node_mask:  tensor([ True, False, False, False, False])
        answer_node_mask:  tensor([False, False, False, False,  True])
        question_training_nodes_mask:  tensor([False,  True,  True,  True,  True]) '''

        # concept and answer nodes
        question_specific_concept_nodes =  torch.tensor(self.get_concepts(concept_uri))
        question_specific_answer_nodes = torch.tensor(self.get_concepts(answer_uri))

        # 2 hop neighbors from concept nodes
        mask = torch.isin(self.data.edge_index[0], question_specific_concept_nodes)
        one_hop_neighbors = torch.unique(self.data.edge_index[1, mask])
        one_hop_neighbors_with_q_node = torch.unique(torch.cat((one_hop_neighbors,question_specific_concept_nodes)))

        subgraph_mask = torch.zeros_like(self.data.edge_index[0], dtype=torch.bool)
        for node in one_hop_neighbors_with_q_node:
            subgraph_mask = subgraph_mask | (self.data.edge_index[0]==node)
        
        # all concepts in the subgraph
        question_subgraph_all_nodes = torch.unique(self.data.edge_index[:,subgraph_mask])

        # initialize masks: 
        question_subgraph_concept_mask = torch.zeros_like(question_subgraph_all_nodes, dtype=torch.bool)
        question_subgraph_answer_and_random_nodes_mask = torch.zeros_like(question_subgraph_all_nodes, dtype=torch.bool)
        question_subgraph_answer_mask = torch.zeros_like(question_subgraph_all_nodes, dtype=torch.bool)

        question_subgraph_concept_mask = question_subgraph_concept_mask | torch.isin(question_subgraph_all_nodes , question_specific_concept_nodes)
        question_subgraph_answer_mask  =  question_subgraph_answer_mask | torch.isin(question_subgraph_all_nodes , question_specific_answer_nodes)

        # Exclude question_mask and answer_mask items
        valid_indices = torch.where(~question_subgraph_concept_mask & ~question_subgraph_answer_mask)[0]
        n=18 # randomly sample 18 nodes for training
        if n < len(valid_indices):
            random_indices  = random.sample(valid_indices.tolist(), n)
            question_subgraph_answer_and_random_nodes_mask[random_indices] = True
            question_subgraph_answer_and_random_nodes_mask = question_subgraph_answer_and_random_nodes_mask | question_subgraph_answer_mask
        else:
            question_subgraph_answer_and_random_nodes_mask[~question_subgraph_concept_mask] = True
    
        self.subgraph_mask = subgraph_mask
        self.question_subgraph_all_nodes = question_subgraph_all_nodes
        self.question_subgraph_concept_mask = question_subgraph_concept_mask
        self.question_subgraph_answer_mask = question_subgraph_answer_mask
        self.question_subgraph_answer_and_random_nodes_mask = question_subgraph_answer_and_random_nodes_mask

        return subgraph_mask, question_subgraph_all_nodes, question_subgraph_concept_mask, question_subgraph_answer_mask, question_subgraph_answer_and_random_nodes_mask
    
    def get_question_training_mask_for_x(self):
        self.q_training_x_mask = torch.full((self.data.x.size()[0],),False, dtype=torch.bool)
        self.q_training_x_mask[self.question_subgraph_all_nodes[self.question_subgraph_answer_and_random_nodes_mask]] = True
        return self.q_training_x_mask
    
    def get_question_y_labels(self):
        self.q_y_label = torch.zeros((self.data.x.size()[0],), dtype=torch.long)
        self.q_y_label[self.question_subgraph_all_nodes[self.question_subgraph_answer_mask]] = 1
        return self.q_y_label
    
    def get_concepts(self,concepts:list):
        conceptIds = []
        for concept in concepts:
            if concept in self.entity_to_index.keys():
                conceptIds.append(self.entity_to_index[concept])
        return conceptIds


