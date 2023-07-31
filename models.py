from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, DistilBertModel


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        import os

        os.environ['TORCH_HOME'] = 'cache'  # hacky workaround to set model dir
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()  # remote last fc
        self.fc = nn.Linear(2048, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x, return_h=False):  # (bs, C, H, W)
        pooled_output = self.resnet50(x)
        logits = self.fc(pooled_output)
        if return_h:
            return logits, pooled_output
        else:
            return logits


class TransformerTokenClassifier(nn.Module):
    def __init__(self, pretrained_model_name_or_path, labels: Union[int, List[str]], dropout: float = 0.0):
        super().__init__()
        self.model_kwargs = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}

        self.num_labels = labels if isinstance(labels, int) else len(labels)

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if pretrained_model_name_or_path.startswith("bert") or pretrained_model_name_or_path.startswith("tartuNLP/EstBERT"):
            # Do not add pooling layer for BERT, otherwise it will not get used in the model and cause problems
            self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config, add_pooling_layer=False)
        else:
            self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden2tag = torch.nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, batch: torch.Tensor, return_h=False):
        # get embeddings for all words in mini-batch
        outputs = self.transformer(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        hidden_state = outputs[0]  # batch_size x sequence_len x dim

        # apply dropout
        hidden_state = self.dropout(hidden_state)

        # pass through linear layer
        logits = self.hidden2tag(hidden_state)

        if return_h:
            return logits, hidden_state
        else:
            return logits


class BertSequenceClassifier(nn.Module):
    def __init__(self, pretrained_model_name_or_path, labels: Union[int, List[str]], dropout: float = 0.0):
        super().__init__()
        self.model_kwargs = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}

        self.num_labels = labels if isinstance(labels, int) else len(labels)

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden2tag = torch.nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, batch: torch.Tensor, return_h=False):
        # get embeddings for all words in mini-batch
        outputs = self.transformer(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pooled_output = outputs[1]  # batch_size x dim

        # apply dropout
        pooled_output = self.dropout(pooled_output)

        # pass through linear layer
        logits = self.hidden2tag(pooled_output)

        if return_h:
            return logits, pooled_output
        else:
            return logits


class DistilBertSequenceClassifier(nn.Module):
    def __init__(self, pretrained_model_name_or_path, labels: Union[int, List[str]], dropout: float = 0.0):
        super().__init__()
        self.model_kwargs = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}

        self.num_labels = labels if isinstance(labels, int) else len(labels)

        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.transformer = DistilBertModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.pre_classifier = nn.Linear(self.config.dim, self.config.dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.config.dim, self.num_labels)

    def forward(self, batch: torch.Tensor, return_h=False):
        # get embeddings for all words in mini-batch
        distilbert_output = self.transformer(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        if return_h:
            return logits, pooled_output
        else:
            return logits
