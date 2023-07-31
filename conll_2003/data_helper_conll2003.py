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

def bio_tags_to_spans(tag_sequence):
    # Inspired by allennlp span utils
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        conll_tag = string_tag[2:]
        if bio_tag == "O":
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


def add_label_noise(dataset, corruption_prob=0.0, label_column="ner_tags", noise_level="token", corruption_type="uniform"):
    """Add label noise to a dataset. An alternative approach to: https://github.com/Michael-Tanzer/BERT-mem-lowres/blob/main/proto_train.py#L152

    Args:
        dataset (datasets.Dataset): The dataset to add label noise to.
        corruption_prob (float, optional): Label corruption probability. Defaults to 0.0.
        label_column (str, optional): The column name of the labels. Defaults to "ner_tags".
        noise_level (str, optional): The level of noise to add. Either "token" or "entity". Defaults to "token".
        corruption_type (str, optional): The type of noise to add. Either "uniform" or "flip". Defaults to "uniform".

    Returns:
        datasets.Dataset: The dataset with label noise added.
    """
    if corruption_prob == 0.0:
        return dataset

    class_labels = dataset.features[label_column].feature.names
    io_labels = ["O"]
    for label in class_labels:
        if label.startswith("B-"):
            io_labels.append(label[2:])

    str2int = {label: i for i, label in enumerate(class_labels)}

    def add_token_level_noise(example):
        labels = example[label_column]
        for i, _ in enumerate(labels):
            if np.random.random() <= corruption_prob:
                if corruption_type == "uniform":
                    labels[i] = np.random.randint(0, len(class_labels))
                else:   # flip
                    choices = [i for i in range(len(class_labels)) if i != labels[i]]
                    labels[i] = np.random.choice(choices)
        return example

    def add_entity_level_noise(example):
        # Might be more realistic as annotators are more likely to mislabel an entire entity,
        # but now does not account for cases where only part of an entity is mislabeled.
        labels = example[label_column]
        str_labels = [class_labels[token_label] for token_label in labels]
        entity_spans = bio_tags_to_spans(str_labels)
        for entity_span in entity_spans:
            if np.random.random() <= corruption_prob:
                if corruption_type == "uniform":
                    entity_label = np.random.choice(io_labels)
                else:   # flip
                    choices = [io_label for io_label in io_labels if label != entity_span[0]]
                    entity_label = np.random.choice(choices)
                entity_start = entity_span[1][0]
                entity_end = entity_span[1][1]
                if entity_label == "O":
                    for i in range(entity_start, entity_end + 1):
                        labels[i] = str2int[entity_label]
                else:
                    labels[entity_start] = str2int["B-" + entity_label]
                    for i in range(entity_start + 1, entity_end + 1):
                        labels[i] = str2int["I-" + entity_label]
        return example

    # batched=True caused some problems with arrow tables (mix of non-lists and lists)
    if noise_level == "token":
        return dataset.map(add_token_level_noise, batched=False)
    else:
        return dataset.map(add_entity_level_noise, batched=False)

def prepare_data(gold_fraction, corruption_prob, args):
    conll2003 = Conll2003DataModule(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path, max_seq_length=args.max_seq_length, train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size, num_workers=args.prefetch, pin_memory=True, corruption_prob=corruption_prob,
        gold_fraction=gold_fraction
    )

    train_gold_loader = DataIterator(conll2003.train_gold_dataloader())
    train_silver_loader = conll2003.train_silver_dataloader()
    val_loader = conll2003.val_dataloader()
    test_loader = conll2003.test_dataloader()

    return train_gold_loader, train_silver_loader, val_loader, test_loader, conll2003.num_classes

class Conll2003DataModule:
    """DataModule for CONLL2003 dataset."""

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
            "O": 0,
            "B-PER": 1,
            "I-PER": 2,
            "B-ORG": 3,
            "I-ORG": 4,
            "B-LOC": 5,
            "I-LOC": 6,
            "B-MISC": 7,
            "I-MISC": 8
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.dataset = datasets.load_dataset("conll2003")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=True
        )

        if gold_fraction > 0.0:
            gold_silver = self.dataset.pop("train").train_test_split(test_size=gold_fraction)
            self.dataset["train_gold"] = gold_silver["test"]
            self.dataset["train_silver"] = gold_silver["train"]
        else:
            self.dataset["train_silver"] = self.dataset.pop("train")


        for split in self.dataset.keys():
            if split == "train_silver":
                self.dataset[split] = add_label_noise(self.dataset[split], self.corruption_prob)
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["id", "ner_tags", "pos_tags", "chunk_tags", "tokens"],
            )
            self.columns = [c for c in self.dataset[split].column_names]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    @property
    def num_classes(self):
        return 9

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
                drop_last=True  #TODO check if this is the best approach (if the last gold batch only contains one example and is split into a clean and a gold portion, the gold portion will be empty)
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
