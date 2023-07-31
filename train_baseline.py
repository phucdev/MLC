import torch
import argparse
import json
import transformers
import numpy as np
from datasets import concatenate_datasets
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from conll_2003.data_helper_conll2003 import Conll2003DataModule
from yelp_reviews.data_helper_yelp_reviews import YelpReviewsDataModule
from noisyner.data_helper_noisyner import NoisyNERDataModule


def main(arguments):
    model_save_path = args.model_save_path
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    if arguments.dataset == "conll2003":
        dataset = Conll2003DataModule(pretrained_model_name_or_path, 128,
                                      train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                      num_workers=arguments.num_workers, pin_memory=True,
                                      corruption_prob=args.corruption_prob, gold_fraction=args.gold_fraction)
    elif arguments.dataset == "noisyner":
        dataset = NoisyNERDataModule(pretrained_model_name_or_path, 256,
                                      train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                      num_workers=arguments.num_workers, pin_memory=True,
                                      configuration=args.configuration, gold_fraction=args.gold_fraction)
    elif arguments.dataset == "yelp_reviews":
        dataset = YelpReviewsDataModule(pretrained_model_name_or_path, 256,
                                        train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                        num_workers=arguments.num_workers, pin_memory=True,
                                        corruption_prob=args.corruption_prob, gold_fraction=args.gold_fraction)
    else:
        raise ValueError("Invalid dataset name")

    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    if "train_gold" in dataset.dataset:
        train_data = concatenate_datasets([dataset.dataset["train_gold"], dataset.dataset["train_silver"]])
    else:
        train_data = dataset.dataset["train_silver"]
    validation_data = dataset.dataset["validation"]
    test_data_loader = dataset.test_dataloader()

    id2label = dataset.id2label
    label2id = {label: _id for _id, label in id2label.items()}
    labels = [label for _id, label in id2label.items()]


    if arguments.dataset == "conll2003":
        data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = transformers.AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path,
                                                                             num_labels=len(labels),
                                                                             id2label=id2label, label2id=label2id)
    elif arguments.dataset == "noisyner":
        data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = transformers.AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path,
                                                                             num_labels=len(labels),
                                                                             id2label=id2label, label2id=label2id)
    elif arguments.dataset == "yelp_reviews":
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                                num_labels=len(labels),
                                                                                id2label=id2label,
                                                                                label2id=label2id)
    else:
        raise ValueError("Invalid dataset name")
    # model = torch.compile(model)  # somehow causes IndexError: Invalid key: 13078 is out of bounds for size 0 in the data loader
    training_args = transformers.TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        num_train_epochs=args.num_epochs,
        save_strategy="no"
    )

    # Metric helper method
    def compute_metrics(eval_pred):
        predictions, true_labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)

        flattened_predictions = predictions.flatten()
        flattened_labels = true_labels.flatten()
        flattened_predictions = flattened_predictions[flattened_labels != -100]
        flattened_labels = flattened_labels[flattened_labels != -100]
        # sklearn
        precision, recall, f1_score, support = precision_recall_fscore_support(
            flattened_labels, flattened_predictions, average=None, labels=[i for i in range(1, len(id2label))])
        class_metrics = {}
        for i, label in enumerate(id2label):
            if i == 0:
                continue
            class_metrics[label] = {
                "precision": precision[i - 1],
                "recall": recall[i - 1],
                "f1": f1_score[i - 1],
                "support": support[i - 1],
            }
        print(class_metrics)
        overall_precision, overall_recall, overall_f1_score, _ = precision_recall_fscore_support(
            flattened_labels, flattened_predictions, average="micro", labels=[i for i in range(1, len(id2label))])
        overall_accuracy = accuracy_score(flattened_labels, flattened_predictions)
        overall_metrics = {
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1_score,
            "overall_accuracy": overall_accuracy,
        }
        return overall_metrics

    model_trainer = transformers.Trainer
    trainer = model_trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(model_save_path)
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_save_path).cuda()
    y_pred = []
    y_true = []
    model.eval()
    for batch in test_data_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # flatten over batch dimension
        flattened_labels = batch["labels"].flatten()
        flattened_predictions = predictions.flatten()
        flattened_predictions = flattened_predictions[flattened_labels != -100]
        flattened_labels = flattened_labels[flattened_labels != -100]
        y_pred += flattened_predictions.tolist()
        y_true += flattened_labels.tolist()

    # sklearn
    cls_report = classification_report(
        y_true, y_pred, labels=np.unique(y_true)[1:], target_names=labels[1:], output_dict=True)
    print(classification_report(y_true, y_pred, labels=np.unique(y_true)[1:], target_names=labels[1:]))

    with open(model_save_path + "/test_results.json", "w") as f:
        json.dump(cls_report, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline training')
    parser.add_argument('--dataset', type=str, default='conll2003', help='dataset name')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--corruption_prob', default=0.0, type=float,
                        help='Corruption probability for label noise')
    parser.add_argument('--configuration', default="NoisyNER_labelset1", type=str,
                        help='Configuration name for NoisyNER')
    parser.add_argument('--gold_fraction', default=0.02, type=float,
                        help='Fraction of gold training data')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='bert-base-cased', help='model name')
    parser.add_argument('--model_save_path', type=str, default='models/conll2003_baseline', help='model path')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    args = parser.parse_args()

    main(args)
