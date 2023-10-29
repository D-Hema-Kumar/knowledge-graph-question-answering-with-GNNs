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
        is_vad_kb,
        LM_embeddings_path=None,
        KG_embeddings_path=None,
        labeler=None,
        
    ):
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

        self.LM_embeddings_path = LM_embeddings_path
        self.KG_embeddings_path = KG_embeddings_path

        if self.LM_embeddings_path is not None:

            self.entities_lm_embeddings = {}
            loaded_data = np.load(
                os.path.join(self.LM_embeddings_path, "entities.npz"), allow_pickle=True
            )
            for key in loaded_data.keys():
                self.entities_lm_embeddings[key] = loaded_data[key] # LM embeddings are based on 'label' column. Keys are labels, values are the embeddings
            
            self.properties_label_to_embeddings = {}
            loaded_data = np.load(
            os.path.join(LM_embeddings_path, "properties.npz"), allow_pickle=True
            )
            for key in loaded_data.keys():
                self.properties_label_to_embeddings[key] = loaded_data[key]

        if KG_embeddings_path is not None:
            
            self.entities_kge_model_embeddings = {}
            loaded_data = np.load(
                os.path.join(KG_embeddings_path, "entities.npz"), allow_pickle=True
            )
            for key in loaded_data.keys():
                self.entities_kge_model_embeddings[key] = loaded_data[key] # KGE model embeddings have URI's as Keys and values are the embeddings
            
            # Property embeddings are not being used in this system but are READ for future improvements
            self.properties_label_to_embeddings = {}
            loaded_data = np.load(
            os.path.join(KG_embeddings_path, "properties.npz"), allow_pickle=True
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

        if self.LM_embeddings_path is not None and self.KG_embeddings_path is not None:
            
            LM_embeddings = np.array(
            self.entities_data["label"].map(self.entities_lm_embeddings).to_list()
            )

            if self.is_vad_kb:
                KGE_model_embeddings = np.array(
            self.entities_data["uri"].map(self.entities_kge_model_embeddings).to_list()
            )
            else:# MetaQA dataset doesn't have URI
                KGE_model_embeddings = np.array(
            self.entities_data["label"].map(self.entities_kge_model_embeddings).to_list()
            )

            x = torch.cat(
                (
                torch.tensor(LM_embeddings), torch.tensor(KGE_model_embeddings)
            ),
            dim = 1
            )
                          

        elif self.LM_embeddings_path is not None:
            LM_embeddings = np.array(
            self.entities_data["label"].map(self.entities_lm_embeddings).to_list()
            )
            x = torch.tensor(LM_embeddings)

        elif self.KG_embeddings_path is not None:

            if self.is_vad_kb:
                KGE_model_embeddings = np.array(
            self.entities_data["uri"].map(self.entities_kge_model_embeddings).to_list()
            )
            else:# MetaQA dataset doesn't have URI
                KGE_model_embeddings = np.array(
            self.entities_data["label"].map(self.entities_kge_model_embeddings).to_list()
            )
            x = torch.tensor(KGE_model_embeddings)
            
        else:
            raise ValueError("No embeddings provided.") # Scope for improvement: if no embeddings given, then initiate randomly

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

    def get_edge_attr(self):
        '''get edge features'''
        edge_embeddings = np.array(
            self.properties_data["label"].map(self.properties_label_to_embeddings).to_list()
        )
        edge_attr = torch.tensor(edge_embeddings)
       
        return edge_attr

    def get_edge_type(self):
        """
        Return edge type list.
        """
        property_to_id = {
                string: index
                for index, string in enumerate(self.properties_data.iloc[:, 0].to_list())
            }
        properties = self.triples_data.iloc[:, 1].map(property_to_id)
        if self.is_vad_kb:
            return torch.tensor(properties, dtype=torch.long)
        
        else: # if dataset is MetaQA, add the inverse relation types because they don't have them in the KB

            inverse_properties = np.array(properties) + len(property_to_id)
            
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
    '''This class is for MetaQA and other datasets '''

    def __init__(
    self,
    triples_path,
    entities_labels_path,
    properties_labels_path,
    is_vad_kb,
    LM_embeddings_path,
    KG_embeddings_path,
    training_questions_concepts_answers_file_path,
    testing_questions_concepts_answers_file_path,
    training_questions_embeddings_path,
    testing_questions_embeddings_path,
    training_subgraphs_file_path,
    testing_subgraphs_file_path,
    labeler=None,
    ):
        
        super().__init__(
            triples_path=triples_path,
            entities_labels_path=entities_labels_path,
            properties_labels_path=properties_labels_path,
            is_vad_kb=is_vad_kb,
            LM_embeddings_path=LM_embeddings_path,
            KG_embeddings_path=KG_embeddings_path,
            labeler=labeler,
        )
        self.training_questions_concepts_answers = (
            pd.read_csv(training_questions_concepts_answers_file_path)
            )
        self.testing_questions_concepts_answers = (
            pd.read_csv(testing_questions_concepts_answers_file_path)
            )
        # reads columns as lists instead of strings
        list_columns = ['concepts', 'answers']
        for col in list_columns:
            self.training_questions_concepts_answers[col] = self.training_questions_concepts_answers[col].apply(ast.literal_eval)
            self.testing_questions_concepts_answers[col] = self.testing_questions_concepts_answers[col].apply(ast.literal_eval)
        
        #load training_question embeddings
        self.training_questions_to_embeddings = {}
        loaded_data = np.load(
        os.path.join(training_questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.training_questions_to_embeddings[key] = loaded_data[key]

        #load testing_question embeddings
        self.testing_questions_to_embeddings = {}
        loaded_data = np.load(
        os.path.join(testing_questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.testing_questions_to_embeddings[key] = loaded_data[key]
        
        # load training subgraphs
        self.training_questions_to_subgraphs = {}
        loaded_data = np.load(training_subgraphs_file_path, allow_pickle=True)
        for key in loaded_data.keys():
            self.training_questions_to_subgraphs[key] = loaded_data[key]
        
        # load testing subgraphs
        self.testing_questions_to_subgraphs = {}
        loaded_data = np.load(testing_subgraphs_file_path, allow_pickle=True)
        for key in loaded_data.keys():
            self.testing_questions_to_subgraphs[key] = loaded_data[key]

        self.x = self.get_x()
        self.edge_index = self.get_edge_index()
        self.edge_type = self.get_edge_type()


    def build_data(self):
        self.data = Data(x=self.x,edge_index=self.edge_index, edge_type=self.edge_type)
        return self.data
    
    def get_question_training_x_mask(self, question_subgraph_data,n=18):
    
        question_training_mask = torch.zeros_like(question_subgraph_data['question_subgraph_nodes'], dtype=torch.bool)

        question_subgraph_concept_mask = question_subgraph_data['question_subgraph_concept_mask']
        question_subgraph_answer_mask  = question_subgraph_data['question_subgraph_answer_mask']

        # Exclude question_mask and answer_mask items
        valid_indices = torch.where(~question_subgraph_concept_mask & ~question_subgraph_answer_mask)[0]

        if n < len(valid_indices):
            random_indices  = random.sample(valid_indices.tolist(), n)
            question_training_mask[random_indices] = True
            question_training_mask = question_training_mask | question_subgraph_answer_mask
        else:
            question_training_mask[~question_subgraph_concept_mask] = True
        
        return question_training_mask
    
    def get_question_data(self, question:str, training=False):#

        if training:
            question_subgraph_data = self.training_questions_to_subgraphs[question].item()
                
            # q_x
            q_x = torch.cat(
                (
                    self.x[question_subgraph_data['question_subgraph_nodes']],
                    torch.from_numpy(self.training_questions_to_embeddings[question].reshape(1, -1).repeat(self.x[question_subgraph_data['question_subgraph_nodes']].shape[0], axis=0)),
                ),
                dim=1,
            )  # broadcast the question emb to match the x dim

            # q_edge_index
            q_edge_index =  question_subgraph_data['reindexed_question_subgraph_edge_index']

            # q_edge_type
            q_edge_type = question_subgraph_data['question_edge_type']

            # q_training_mask
            q_training_mask = self.get_question_training_x_mask(question_subgraph_data,n=18)

            # q_y
            q_y = question_subgraph_data['question_y']

            return Data(x=q_x, edge_index=q_edge_index, edge_type=q_edge_type, train_mask=q_training_mask, y=q_y)
        
        else:#testing phase

            #load question specific subgraph
            question_subgraph_data = self.testing_questions_to_subgraphs[question].item() 
            
            #q_x
            q_x = torch.cat(
                (
                    self.x[question_subgraph_data['question_subgraph_nodes']],
                    torch.from_numpy(self.testing_questions_to_embeddings[question].reshape(1, -1).repeat(self.x[question_subgraph_data['question_subgraph_nodes']].shape[0], axis=0)),
                ),
                dim=1,
            )

            # q_edge_index
            q_edge_index =  question_subgraph_data['reindexed_question_subgraph_edge_index']

            # q_edge_type
            q_edge_type = question_subgraph_data['question_edge_type'] 

            return Data(x=q_x, edge_index=q_edge_index, edge_type=q_edge_type), question_subgraph_data['question_subgraph_nodes'], question_subgraph_data['question_subgraph_nodes'][question_subgraph_data['question_subgraph_answer_mask']]


    
    


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
    is_vad_kb,
    LM_embeddings_path,
    KG_embeddings_path,
    training_questions_concepts_answers_file_path,
    testing_questions_concepts_answers_file_path,
    training_questions_embeddings_path,
    testing_questions_embeddings_path,
    labeler=None,
    ):
        super().__init__(
            triples_path=triples_path,
            entities_labels_path=entities_labels_path,
            properties_labels_path=properties_labels_path,
            is_vad_kb=is_vad_kb,
            LM_embeddings_path=LM_embeddings_path,
            KG_embeddings_path=KG_embeddings_path,
            labeler=labeler,
        )
        self.training_questions_concepts_answers = (
            pd.read_csv(training_questions_concepts_answers_file_path)
            )
        self.testing_questions_concepts_answers = (
            pd.read_csv(testing_questions_concepts_answers_file_path)
            )
        # reads columns as lists instead of strings
        list_columns = ['concepts', 'answers']
        for col in list_columns:
            self.training_questions_concepts_answers[col] = self.training_questions_concepts_answers[col].apply(ast.literal_eval)
            self.testing_questions_concepts_answers[col] = self.testing_questions_concepts_answers[col].apply(ast.literal_eval)
        
        #load training_question embeddings
        self.training_questions_to_embeddings = {}
        loaded_data = np.load(
        os.path.join(training_questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.training_questions_to_embeddings[key] = loaded_data[key]

        #load testing_question embeddings
        self.testing_questions_to_embeddings = {}
        loaded_data = np.load(
        os.path.join(testing_questions_embeddings_path, "questions.npz"), allow_pickle=True
        )
        for key in loaded_data.keys():
            self.testing_questions_to_embeddings[key] = loaded_data[key]

        self.x = self.get_x()
        self.edge_index = self.get_edge_index()
        self.edge_type = self.get_edge_type()


    def build_data(self):
        self.data = Data(x=self.x,edge_index=self.edge_index, edge_type=self.edge_type)
        return self.data


    def get_question_data(self, question:str, concept_uri:list, answer_uri:list, training=False):
        '''
        given a question, q_node, a_node : builds a list of subgraph concepts and its related masks:
        e.g. 
        question_subgraph_concepts:  tensor([0, 1, 2, 3, 4])
        question_node_mask:  tensor([ True, False, False, False, False])
        answer_node_mask:  tensor([False, False, False, False,  True])
        question_training_nodes_mask:  tensor([False,  True,  True,  True,  True]) '''

        # concept and answer nodes
        question_specific_concept_nodes =  torch.tensor(self.get_concepts(concept_uri))
        self.question_specific_answer_nodes = torch.tensor(self.get_concepts(answer_uri))

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
        question_subgraph_answer_mask  =  question_subgraph_answer_mask | torch.isin(question_subgraph_all_nodes , self.question_specific_answer_nodes)

        # Exclude question_mask and answer_mask items
        valid_indices = torch.where(~question_subgraph_concept_mask & ~question_subgraph_answer_mask)[0]
        n=18 # randomly sample 18 nodes for training
        if n < len(valid_indices):
            random_indices  = random.sample(valid_indices.tolist(), n)
            question_subgraph_answer_and_random_nodes_mask[random_indices] = True
            question_subgraph_answer_and_random_nodes_mask = question_subgraph_answer_and_random_nodes_mask | question_subgraph_answer_mask
        else:
            question_subgraph_answer_and_random_nodes_mask[~question_subgraph_concept_mask] = True
        
        # subgraph's x

        if training:
            #print('*** Shape of X:',self.x[question_subgraph_all_nodes].shape)
            #print('*** Shape of Q:',torch.from_numpy(self.training_questions_to_embeddings[question].reshape(1, -1).repeat(self.x[question_subgraph_all_nodes].shape[0], axis=0)).shape)
            self.q_x = torch.cat(
                (
                    self.x[question_subgraph_all_nodes],
                    torch.from_numpy(self.training_questions_to_embeddings[question].reshape(1, -1).repeat(self.x[question_subgraph_all_nodes].shape[0], axis=0)),
                ),
                dim=1,
            )  # broadcast the question emb to match the x dim
        else: #testing phase
            self.q_x = torch.cat(
                (
                    self.x[question_subgraph_all_nodes],
                    torch.from_numpy(self.testing_questions_to_embeddings[question].reshape(1, -1).repeat(self.x[question_subgraph_all_nodes].shape[0], axis=0)),
                ),
                dim=1,
            )  # broadcast the question emb to match the x dim


        # subgraph's edge_index
        self.question_subgraph_all_nodes_to_index = {sub_node : index for index, sub_node in enumerate (question_subgraph_all_nodes.tolist())}
        self.mapped_q_subgraph_edge_index = torch.tensor(
                                                [
                                                [self.question_subgraph_all_nodes_to_index[head.item()] for head in self.edge_index[:,subgraph_mask][0]],
                                                [self.question_subgraph_all_nodes_to_index[tail.item()] for tail in self.edge_index[:,subgraph_mask][1]]
                                                ],dtype=torch.long
                                            )
        
        # subgraph's edge_type
        self.q_edge_type = self.edge_type[subgraph_mask]
        
        # subgraph's training and y labels
        self.q_training_x_mask = question_subgraph_answer_and_random_nodes_mask
        self.q_y = torch.zeros((len(question_subgraph_all_nodes),), dtype=torch.long)
        self.q_y[question_subgraph_answer_mask]=1
    
        self.subgraph_mask = subgraph_mask
        self.question_subgraph_all_nodes = question_subgraph_all_nodes
        self.question_subgraph_concept_mask = question_subgraph_concept_mask
        self.question_subgraph_answer_mask = question_subgraph_answer_mask
        self.question_subgraph_answer_and_random_nodes_mask = question_subgraph_answer_and_random_nodes_mask

        #return subgraph_mask, question_subgraph_all_nodes, question_subgraph_concept_mask, question_subgraph_answer_mask, question_subgraph_answer_and_random_nodes_mask
        if training:
            return Data(x=self.q_x, edge_index=self.mapped_q_subgraph_edge_index, edge_type=self.q_edge_type, train_mask=self.q_training_x_mask, y=self.q_y)
        return Data(x=self.q_x, edge_index=self.mapped_q_subgraph_edge_index, edge_type=self.q_edge_type)
    
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

    def get_question_2_hop_subgraph_mask(self,concepts:list): #TODO instead of 2 make it K hop in future

        ''' gets you MASK to apply onf data.edge_index to get all the edges k-hops given concept nodes.'''

        self.question_specific_concept_nodes =  torch.tensor(self.get_concepts(concepts))

        # 2 hop neighbors from concept nodes
        mask = torch.isin(self.edge_index[0], self.question_specific_concept_nodes)
        one_hop_neighbors = torch.unique(self.edge_index[1, mask])
        one_hop_neighbors_with_q_node = torch.unique(
                                    torch.cat(
                                        (one_hop_neighbors,self.question_specific_concept_nodes)
                                        )
                                    )

        question_2_hop_subgraph_mask = torch.zeros_like(self.edge_index[0], dtype=torch.bool)
        for node in one_hop_neighbors_with_q_node:
            question_2_hop_subgraph_mask = question_2_hop_subgraph_mask | (self.edge_index[0]==node)
        
        return question_2_hop_subgraph_mask
    
    def get_question_subgraph_nodes(self, question_k_hop_subgraph_mask):
        return torch.unique(self.edge_index[:,question_k_hop_subgraph_mask])

    def get_question_subgraph_concept_mask(self, question_subgraph_nodes:list ):

        question_subgraph_concept_mask = torch.zeros_like(question_subgraph_nodes, dtype=torch.bool)
        question_subgraph_concept_mask = question_subgraph_concept_mask | torch.isin(question_subgraph_nodes , self.question_specific_concept_nodes)
        return question_subgraph_concept_mask
    
    def get_question_subgraph_answer_mask(self, question_subgraph_nodes:list, answer_uri:list):

        self.question_specific_answer_nodes = torch.tensor(self.get_concepts(answer_uri))

        question_subgraph_answer_mask = torch.zeros_like(question_subgraph_nodes, dtype=torch.bool)
        question_subgraph_answer_mask  =  question_subgraph_answer_mask | torch.isin(question_subgraph_nodes , self.question_specific_answer_nodes)
        return question_subgraph_answer_mask

    def get_question_y(self, question_subgraph_nodes, question_subgraph_answer_mask):

        question_y = torch.zeros((len(question_subgraph_nodes),), dtype=torch.long)
        question_y[question_subgraph_answer_mask]=1
        return question_y

    def get_reindexed_question_subgraph_edge_index(self, question_subgraph_nodes, question_k_hop_subgraph_mask ):

        question_subgraph_nodes_to_index = {sub_node : index for index, sub_node in enumerate (question_subgraph_nodes.tolist())}
        reindexed_question_subgraph_edge_index = torch.tensor(
                                                [
                                                [question_subgraph_nodes_to_index[head.item()] for head in self.edge_index[:,question_k_hop_subgraph_mask][0]],
                                                [question_subgraph_nodes_to_index[tail.item()] for tail in self.edge_index[:,question_k_hop_subgraph_mask][1]]
                                                ],dtype=torch.long
                                            )
        return reindexed_question_subgraph_edge_index
    
    def get_question_edge_type(self, question_k_hop_subgraph_mask):
        return self.edge_type[question_k_hop_subgraph_mask]
    
    def get_question_subgraph_dict(self, concept_uri:list, answer_uri:list):

        question_2_hop_subgraph_mask = self.get_question_2_hop_subgraph_mask(concept_uri)
        question_subgraph_nodes = self.get_question_subgraph_nodes(question_2_hop_subgraph_mask)
        question_subgraph_concept_mask = self.get_question_subgraph_concept_mask(question_subgraph_nodes)
        question_subgraph_answer_mask = self.get_question_subgraph_answer_mask(question_subgraph_nodes, answer_uri)
        question_y = self.get_question_y(question_subgraph_nodes, question_subgraph_answer_mask)
        reindexed_question_subgraph_edge_index = self.get_reindexed_question_subgraph_edge_index(question_subgraph_nodes, question_2_hop_subgraph_mask )
        
        return {
                'question_2_hop_subgraph_mask':question_2_hop_subgraph_mask,
                'question_subgraph_nodes':question_subgraph_nodes,
                'question_subgraph_concept_mask':question_subgraph_concept_mask,
                'question_subgraph_answer_mask':question_subgraph_answer_mask,
                'question_y':question_y,
                'reindexed_question_subgraph_edge_index':reindexed_question_subgraph_edge_index
                }
    def save_questions_subgraphs(self,questions_concepts_answers_file_path:str,file_name:str):

        questions_concepts_answers = (
            pd.read_csv(questions_concepts_answers_file_path)
            )
        # reads columns as lists instead of strings
        list_columns = ['concepts', 'answers']
        for col in list_columns:
            questions_concepts_answers[col] = questions_concepts_answers[col].apply(ast.literal_eval)
        
        questions_subgraphs = {}
        for _, row in questions_concepts_answers.iterrows():
            questions_subgraphs[row['question']] = self.get_question_subgraph_dict(concept_uri=row["concepts"]
                                                                                   ,answer_uri=row["answers"])
        
        # Save the dictionary to a NumPy file
        np.save(file_name+'.npy', questions_subgraphs)
