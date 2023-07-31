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


def add_label_noise(dataset, corruption_prob=0.0, label_column="label", corruption_type="uniform"):
    """Add label noise to a dataset. An alternative approach to: https://github.com/Michael-Tanzer/BERT-mem-lowres/blob/main/proto_train.py#L152

    Args:
        dataset (datasets.Dataset): The dataset to add label noise to.
        corruption_prob (float, optional): Label corruption probability. Defaults to 0.0.
        label_column (str, optional): The column name of the labels. Defaults to "ner_tags".
        corruption_type (str, optional): The type of noise to add. Either "uniform" or "flip". Defaults to "uniform".

    Returns:
        datasets.Dataset: The dataset with label noise added.
    """
    if corruption_prob == 0.0:
        return dataset

    class_labels = dataset.features[label_column].names

    def add_noise(example):
        if np.random.random() <= corruption_prob:
            if corruption_type == "uniform":
                example[label_column] = np.random.randint(0, len(class_labels))
            else:   # flip
                choices = [i for i in range(len(class_labels)) if i != example[label_column]]
                example[label_column] = np.random.choice(choices)
        return example

    # batched=True caused some problems with arrow tables (mix of non-lists and lists)
    return dataset.map(add_noise, batched=False)

def prepare_data(gold_fraction, corruption_prob, args):
    yelp_review_full = YelpReviewsDataModule(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, max_seq_length=args.max_seq_length, train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size, num_workers=args.prefetch, pin_memory=True, corruption_prob=corruption_prob,
        gold_fraction=gold_fraction
    )

    train_gold_loader = DataIterator(yelp_review_full.train_gold_dataloader())
    train_silver_loader = yelp_review_full.train_silver_dataloader()
    val_loader = yelp_review_full.val_dataloader()
    test_loader = yelp_review_full.test_dataloader()

    return train_gold_loader, train_silver_loader, val_loader, test_loader, yelp_review_full.num_classes

class YelpReviewsDataModule:
    """DataModule for Yelp Reviews dataset."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        corruption_prob: float = 0.0,
        gold_fraction: float = 0.02
    ):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.corruption_prob = corruption_prob

        self.columns = None
        self.tokenizer = None
        self.label2id = {
            "1 star": 0,
            "2 stars": 1,
            "3 stars": 2,
            "4 stars": 3,
            "5 stars": 4
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.dataset = datasets.load_dataset("yelp_review_full")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=True
        )

        train_validation = self.dataset.pop("train").train_test_split(test_size=0.05)
        train = train_validation["train"]
        self.dataset["validation"] = train_validation["test"]

        if gold_fraction > 0.0:
            gold_silver = train.train_test_split(test_size=gold_fraction)
            self.dataset["train_gold"] = gold_silver["test"]
            self.dataset["train_silver"] = gold_silver["train"]
        else:
            self.dataset["train_silver"] = train


        for split in self.dataset.keys():
            if split == "train_silver":
                self.dataset[split] = add_label_noise(self.dataset[split], self.corruption_prob)
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["text", "label"],
            )
            self.columns = [c for c in self.dataset[split].column_names]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    @property
    def num_classes(self):
        return 5

    def train_gold_dataloader(self):
        if "train_gold" not in self.dataset.keys():
            raise ValueError("No gold training set available.")
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
        text = example_batch["text"]

        # Tokenize the text/text pairs
        features = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        features["labels"] = example_batch["label"]

        return features
