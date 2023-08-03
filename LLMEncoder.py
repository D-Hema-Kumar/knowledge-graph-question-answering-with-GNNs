import torch
import pandas as pd
import numpy as np
import os
from loguru import logger
from constants import QUESTIONS_ANSWERS_PATH, GRAPH_EMBEDDINGS_PATH


class LLMEncoder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encode_sentence(self, input_sentence):
        """
        Encode the given sentence using the language model.
        """
        tokens = self.tokenizer.encode(input_sentence, add_special_tokens=True)
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = self.model(input_ids)
        cls_token_representation = outputs.last_hidden_state[:, 0, :]
        normalized_representation = torch.nn.functional.normalize(
            cls_token_representation
        )
        return normalized_representation.numpy()[0]

    def generate_encodings_dict(self, list_input_sentences):
        """
        Generate encodings dict for a list of input sentences.
        """
        logger.info("Generating encodings.")
        res = {}
        for sentence in list_input_sentences:
            res[sentence] = self.encode_sentence(sentence)
        return res

    def save_encodings(self, encodings, file_path):
        """
        Save encodings to disk.
        """
        logger.info("Saving encodings.")
        np.savez(file_path, **encodings)

    def generate_encodings_for_entities_labels(self, entities_labels_path, base_path):
        logger.info("Generating encodings for Entities.")
        entities_data = pd.read_csv(entities_labels_path)
        mask = entities_data.iloc[:, 1].isna()
        entities_data.loc[mask, "label"] = (
            entities_data[entities_data.iloc[:, 1].isna()]
            .iloc[:, 0]
            .apply(lambda x: x.split("/")[-1])
        )  # deal with missing labels
        encodings = self.generate_encodings_dict(entities_data["label"].to_list())
        self.save_encodings(encodings, os.path.join(base_path, "entities"))

    def generate_encodings_for_properties_labels(
        self, properties_labels_path, base_path
    ):
        logger.info("Generating encodings for Properties.")
        properties_data = pd.read_csv(properties_labels_path)
        encodings = self.generate_encodings_dict(properties_data["label"].to_list())
        self.save_encodings(encodings, os.path.join(base_path, "properties"))
    
    def generate_encodings_for_questions(
            self, question_answers_path, base_path
    ):
        logger.info("Generating encodings for Questions.")
        question_answer_data = pd.read_csv(question_answers_path)
        encodings = self.generate_encodings_dict(question_answer_data["question"].to_list())
        self.save_encodings(encodings, os.path.join(base_path, "questions"))


if __name__ == "__main__":
    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    llm_encoder = LLMEncoder(tokenizer, model)
    #llm_encoder.generate_encodings_for_entities_labels(
    #    entities_labels_path="data/VAD_entities_labels.csv", base_path="data/embeddings"
    #)
    #llm_encoder.generate_encodings_for_properties_labels(
    #    properties_labels_path="data/VAD_properties_labels.csv",
    #    base_path="data/embeddings",
    #)
    llm_encoder.generate_encodings_for_questions(question_answers_path=QUESTIONS_ANSWERS_PATH,base_path=GRAPH_EMBEDDINGS_PATH)
