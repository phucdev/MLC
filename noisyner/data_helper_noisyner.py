from typing import Optional

from transformers import AutoTokenizer
import datasets
import torch
import numpy as np

from torch.utils.data import DataLoader


class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        return batch

def prepare_data(gold_fraction, args):
    configuration = args.configuration
    noisy_ner = NoisyNERDataModule(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, max_seq_length=args.max_seq_length, train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size, num_workers=args.prefetch, pin_memory=True, configuration=configuration,
        gold_fraction=gold_fraction
    )

    train_gold_loader = DataIterator(noisy_ner.train_gold_dataloader())
    train_silver_loader = noisy_ner.train_silver_dataloader()
    val_loader = noisy_ner.val_dataloader()
    test_loader = noisy_ner.test_dataloader()

    return train_gold_loader, train_silver_loader, val_loader, test_loader, noisy_ner.num_classes

class NoisyNERDataModule:
    """DataModule for Noisy NER dataset."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        gold_fraction: float = 0.02,
        configuration: str = "estner_clean"
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.columns = None
        self.tokenizer = None
        self.label2id = {
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        # The Noisy NER dataset actually provides noisy labels for all splits which is not the usual setting
        # So maybe use the clean dataset as the base and only replace the labels of the train silver split with the noisy labels
        noisy_dataset = datasets.load_dataset("phucdev/noisyner", name=configuration)
        self.dataset = datasets.load_dataset("phucdev/noisyner", name="estner_clean")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=True
        )

        if gold_fraction > 0.0:
            gold_silver = noisy_dataset.pop("train").train_test_split(test_size=gold_fraction, seed=42)
            clean_gold_silver = self.dataset.pop("train").train_test_split(test_size=gold_fraction, seed=42)
            self.dataset["train_gold"] = clean_gold_silver["test"]
            self.dataset["train_silver"] = gold_silver["train"]
        else:
            self.dataset["train_silver"] = noisy_dataset.pop("train")


        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["id", "ner_tags", "lemmas", "grammar", "tokens"],
            )
            self.columns = [c for c in self.dataset[split].column_names]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    @property
    def num_classes(self):
        return 7

    def train_gold_dataloader(self):
        if "train_gold" not in self.dataset.keys():
            raise ValueError("No gold training data available.")
        else:
            return DataLoader(
                dataset=self.dataset["train_gold"],
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )

    def train_silver_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train_silver"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def convert_to_features(self, example_batch):
        tokens = example_batch["tokens"]

        # Tokenize the text/text pairs
        features = self.tokenizer(
            tokens,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,  # because text in conll is pre-tokenized
        )

        # Rename label to labels to make it easier to pass to model forward
        labels = []
        for i, label_list in enumerate(example_batch["ner_tags"]):
            word_ids = features.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100, so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label_list[word_idx])

            labels.append(label_ids)
        features["labels"] = labels

        return features
