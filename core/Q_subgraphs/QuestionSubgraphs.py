from core.ToTorch.DataBuilder import (DataBuilder)
from config.config import MetaQA_CONFIG
import pandas as pd
import os
import numpy as np
import ast
import torch
from tqdm import tqdm
import pickle
from loguru import logger

class QuestionSubgraphBuilder(DataBuilder):

    def __init__(
    self,
    triples_path,
    entities_labels_path,
    properties_labels_path,
    embeddings_path,
    is_vad_kb,
    labeler=None,
    ):
        super().__init__(
            triples_path,
            entities_labels_path,
            properties_labels_path,
            embeddings_path,
            is_vad_kb,
            labeler,
        )

        self.x = self.get_x()
        self.edge_index = self.get_edge_index()
        self.edge_type = self.get_edge_type()

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
        question_edge_type = self.get_question_edge_type(question_2_hop_subgraph_mask)
        return {
                #'question_2_hop_subgraph_mask':question_2_hop_subgraph_mask.numpy(), # making file very large
                'question_subgraph_nodes':question_subgraph_nodes,
                'question_subgraph_concept_mask':question_subgraph_concept_mask,
                'question_subgraph_answer_mask':question_subgraph_answer_mask,
                'question_y':question_y,
                'reindexed_question_subgraph_edge_index':reindexed_question_subgraph_edge_index,
                'question_edge_type':question_edge_type
                }
    def save_questions_subgraphs(self,questions_concepts_answers_file_path:str,base_path,file_name:str):

        questions_concepts_answers = (
            pd.read_csv(questions_concepts_answers_file_path)
            )
        # reads columns as lists instead of strings
        list_columns = ['concepts', 'answers']
        for col in list_columns:
            questions_concepts_answers[col] = questions_concepts_answers[col].apply(ast.literal_eval)
        
        questions_subgraphs = {}
        for _, row in tqdm(questions_concepts_answers.iterrows()):
            questions_subgraphs[row['question']] = self.get_question_subgraph_dict(concept_uri=row["concepts"]
                                                                                   ,answer_uri=row["answers"])
        
        self.save_subgraphs(questions_subgraphs, os.path.join(base_path,file_name))
    
    def save_subgraphs(self, subgraphs, file_path):
        """
        Save subgraphs to disk.
        """
        logger.info("Saving subgraphs.")
        np.savez(file_path, **subgraphs)

if __name__ == "__main__":
    

    
    subgraph_builder = QuestionSubgraphBuilder(
                                triples_path=MetaQA_CONFIG['KB_PATH'],
                                entities_labels_path=MetaQA_CONFIG['ENTITIES_LABELS_PATH'],
                                properties_labels_path=MetaQA_CONFIG['PROPERTIES_LABELS_PATH'],
                                embeddings_path=MetaQA_CONFIG['GRAPH_EMBEDDINGS_PATH'],
                                is_vad_kb=False
                                 )
    subgraph_builder.save_questions_subgraphs(
        questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_DEV_PATH'],
        base_path = MetaQA_CONFIG["QUESTIONS_CONCEPTS_ANSWERS_1_HOP_FOLDER_PATH"],
        file_name = 'dev_subgraphs'

    )

    subgraph_builder.save_questions_subgraphs(
        questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TRAIN_PATH'],
        base_path = MetaQA_CONFIG["QUESTIONS_CONCEPTS_ANSWERS_1_HOP_FOLDER_PATH"],
        file_name = 'train_subgraphs'

    )

    subgraph_builder.save_questions_subgraphs(
        questions_concepts_answers_file_path = MetaQA_CONFIG['QUESTIONS_CONCEPTS_ANSWERS_1_HOP_TEST_PATH'],
        base_path = MetaQA_CONFIG["QUESTIONS_CONCEPTS_ANSWERS_1_HOP_FOLDER_PATH"],
        file_name = 'test_subgraphs'

    )