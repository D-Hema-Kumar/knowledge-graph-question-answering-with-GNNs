import torch


class LLMEncoder:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, input_sentence):
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


if __name__ == "__main__":
    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    llm_encoder = LLMEncoder(tokenizer, model)
    print(llm_encoder.encode("hello world"))
