import torch
import argparse
import json

from models import TransformerTokenClassifier, DistilBertSequenceClassifier, BertSequenceClassifier
from main import test
from conll_2003.data_helper_conll2003 import Conll2003DataModule
from yelp_reviews.data_helper_yelp_reviews import YelpReviewsDataModule
from noisyner.data_helper_noisyner import NoisyNERDataModule


def main(arguments):
    model_save_path = args.model_save_path
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    if arguments.dataset == "conll2003":
        dataset = Conll2003DataModule(pretrained_model_name_or_path, 128,
                                      train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                      num_workers=arguments.num_workers, pin_memory=True)
    elif arguments.dataset == "noisyner":
        dataset = NoisyNERDataModule(pretrained_model_name_or_path, 256,
                                     train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                     num_workers=arguments.num_workers, pin_memory=True)
    elif arguments.dataset == "yelp_reviews":
        dataset = YelpReviewsDataModule(pretrained_model_name_or_path, 256,
                                        train_batch_size=arguments.batch_size, eval_batch_size=arguments.batch_size,
                                        num_workers=arguments.num_workers, pin_memory=True)
    else:
        raise ValueError("Invalid dataset name")
    test_data_loader = dataset.test_dataloader()

    id2label = dataset.id2label
    labels = [label for _id, label in id2label.items()]

    if arguments.dataset == "conll2003" or arguments.dataset == "noisyner":
        model = TransformerTokenClassifier(pretrained_model_name_or_path, labels)
        target_names = list(dataset.id2label.values())[1:]
        ignore_majority_negative_class = True
    elif arguments.dataset == "yelp_reviews":
        if pretrained_model_name_or_path.startswith("bert"):
            model = BertSequenceClassifier(pretrained_model_name_or_path, labels)
        elif pretrained_model_name_or_path.startswith("distilbert"):
            model = DistilBertSequenceClassifier(pretrained_model_name_or_path, labels)
        target_names = list(dataset.id2label.values())
        ignore_majority_negative_class = False
    else:
        raise ValueError("Invalid dataset name")

    torch_dict = torch.load(model_save_path)
    model.load_state_dict(torch_dict["main_net"])
    model = model.cuda()

    metrics, cls_report = test(model, test_data_loader, ignore_majority_negative_class=ignore_majority_negative_class,
                               target_names=target_names)
    print(json.dumps(cls_report, indent=4))

    with open(args.metrics_save_path, "w") as f:
        json.dump(cls_report, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLC Evaluation')
    parser.add_argument('--dataset', type=str, default='conll2003')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='bert-base-cased',
                        help='model name')
    parser.add_argument('--model_save_path', type=str,
                        default='models/conll2003_hmlc_K_mix_unif_conll2003_run_5_1_1_0.02_0.2_best.pth',
                        help='model path')
    parser.add_argument('--metrics_save_path', type=str,
                        default='results/conll2003_hmlc_K_mix_unif_conll2003_run_5_1_1_0.02_0.2_bert-base-cased_best.json',
                        help='metrics save path')
    args = parser.parse_args()

    main(args)
