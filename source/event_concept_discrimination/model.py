import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer


class ConceptLinkingBaselineClassifier(nn.Module):
    def __init__(self, model_name, pretrained_tokenizer_path=None):
        super().__init__()
        if pretrained_tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_type = self.model.config.model_type

        try:
            self.emb_size = self.model.config.d_model  # bart
        except:
            self.emb_size = self.model.config.hidden_size  # roberta/bert/deberta/electra

        self.nn = nn.Linear(self.emb_size, 2)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)
            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])

        if 'bart' in self.model_type:
            # embedding of [EOS] (</s>) in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                                      :, -1, :]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        return sentence_representation

    def forward(self, tokens):
        embeddings = self.get_lm_embedding(tokens)  # (batch_size, emb_size)
        return self.nn(embeddings)


class ConceptLinkingClassifier(nn.Module):
    def __init__(self, model_name, batch_norm=False, pretrained_tokenizer_path=None):
        super().__init__()
        if pretrained_tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model_type = self.model.config.model_type
        self.batch_norm = batch_norm

        try:
            self.emb_size = self.model.config.d_model  # bart
        except:
            self.emb_size = self.model.config.hidden_size  # roberta/bert/deberta/electra

        if self.batch_norm:
            self.nn1 = nn.Sequential(nn.Linear(self.emb_size, 512), nn.BatchNorm1d(512), nn.LeakyReLU())
            self.nn2 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU())
            self.nn3 = nn.Sequential(nn.Linear(256, 64), nn.BatchNorm1d(64), nn.LeakyReLU())
            self.nn4 = nn.Linear(64, 2)
        else:
            self.nn1 = nn.Sequential(nn.Linear(self.emb_size, 512), nn.LayerNorm(512), nn.LeakyReLU())
            self.nn2 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU())
            self.nn3 = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.LeakyReLU())
            self.nn4 = nn.Linear(64, 2)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)
            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])

        if 'bart' in self.model_type:
            # embedding of [EOS] (</s>) in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                                      :, -1, :]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        return sentence_representation

    def forward(self, tokens):
        embeddings = self.get_lm_embedding(tokens)  # (batch_size, emb_size)
        y1 = self.nn1(embeddings)
        y2 = self.nn2(y1)
        y3 = self.nn3(y2)
        return self.nn4(y3)  # (batch_size, 2)
