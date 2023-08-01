# Meta Label Correction for Noisy Label Learning

This repository contains the source code for the AAAI paper "[Meta Label Correction for Noisy Label Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17319/17126)".

![mlc_model](mlc.png)

## Data

The code will download automatically the CIFAR data set; for Clothing1M, please contact the [original creator](https://github.com/Cysu/noisy_label) for access. Put the obtained Clothing1M data set under directory ```data/clothing1M```. Then execute ```cd CLOTHING1M; python3 load_cloth1m_data.py``` to generate necessary folders for training.

## Example runs

On CIFAR-10 run MLC with UNIF noise and a noise level of 0.4 by executing
```bash
python main.py \
  --dataset cifar10 --optimizer sgd --batch_size 100 --corruption_type unif --corruption_prob 0.4 --gold_fraction 0.02 \
  --epochs 120 --main_lr 0.1 --meta_lr 3e-4 --runid cifar10_run  --cls_dim 128
```
On CIFAR-100, run MLC with FLIP noise and a noise level of 0.6 by executing
```bash
python main.py \
  --dataset cifar100 --optimizer sgd --batch_size 100 --corruption_type flip --corruption_prob 0.6 --gold_fraction 0.02 \
  --epochs 120 --main_lr 0.1 --meta_lr 3e-4 --runid cifar100_run  --cls_dim 128
```

On Clothing1M, run MLC as 
```bash
python main.py --dataset clothing1m --optimizer sgd --batch_size 32 --corruption_type unif --corruption_prob 0.1 --gold_fraction 0.1 --epochs 15 --main_lr 0.1 --meta_lr 0.003 --runid clothing1m_run --cls_dim 128 --skip --gradient_steps 5
```
(Note that for clothing1m, ```corruption_type```, ```corruption_level```, and ```gold_fraction``` have no effect as the original dataset comes with actual noisy labels and clean/noisy data splits.)

Refer to ```python3 main.py --help``` for a detailed explanations of all applicable arguments.

## [NEW] NLP: token classification and sequence classification

While the authors describe MLC in the context of image classification, the method is applicable to other tasks as well. 
They did experiments on text classification, but did not include the code or the hyperparameters in the original repository.
I added support for text classification and token classification.

### To run the NLP experiments

#### MLC
For text classification on the Yelp dataset run MLC as
```bash
python main.py \
    --dataset yelp_reviews --pretrained_model_name_or_path distilbert-base-cased --optimizer sgd --batch_size 16 \
    --corruption_type unif --corruption_prob 0.6 --gold_fraction 0.02 --epochs 15 --main_lr 0.1 --meta_lr 0.003 \
    --runid yelp_run --cls_dim 128 --model_save_dir models/MLC_yelp_reviews
```

For token classification on the CONLL2003 dataset run MLC as
```bash
python main.py \
  --dataset conll2003 --pretrained_model_name_or_path distilbert-base-cased --optimizer sgd --batch_size 16 \
  --corruption_type unif --corruption_prob 0.6 --gold_fraction 0.02 --epochs 30 --main_lr 0.1 --meta_lr 0.003 \
  --runid conll_run --cls_dim 128 --model_save_dir models/MLC_conll2003
```

For token classification on the NoisyNER dataset run MLC as
```bash
python main.py \
  --dataset noisyner --pretrained_model_name_or_path tartuNLP/EstBERT --optimizer sgd --batch_size 16 \
  --corruption_type unif --corruption_prob 0.6 --gold_fraction 0.02 --epochs 30 --main_lr 0.1 --meta_lr 0.003 \
  --configuration NoisyNER_labelset7 --runid noisyner_run --cls_dim 128 --model_save_dir models/MLC_noisyner
```
(Note that for noisyner, ```corruption_type```, ```corruption_level``` have no effect as the original dataset comes with actual noisy labels.
Instead, the use `configuration` to set a particular label set.)

#### Baselines
I also added support for training baseline models that do not use MLC. The `train_baseline.py` script is based on
huggingface's transformers library.

For example, you can train a transformer based token classification model on the CONLL2003 dataset as
```bash
python train_baseline.py \
  --dataset conll2003 --pretrained_model_name_or_path distilbert-base-cased --batch_size 16 \
  --corruption_type unif --corruption_prob 0.6 --gold_fraction 0.02 --epochs 5 \
  --model_save_path models/baseline_conll2003
```

### Challenges and observations so far
#### Challenges
During data processing for token classification we use the label id -100 for non-words (padding or special tokens).
This caused multiple challenges:
- The embedding layer of the meta model that creates label embeddings is set up using the number of labels in the dataset,
    but the label id -100 is not included in the label set. Since we don't need extra complexity for the -100 label, I
    chose to set all -100 labels to 0 in the `forward` method. This is not ideal, but it works. However, this may have
    confused the meta model.
- For the loss calculation the torch implementation of cross entropy filters out -100 labels per default. But when 
    calculating the loss of the main model on the corrected labels of the meta model we have soft labels (probabilities
    for each class). I adapted the loss function to use the information from the original hard labels to filter out the
    -100 labels.
- I also had to filter out all the padding tokens/labels for the evaluation.

Another minor thing is that in the implementation the gold training data batch is split into two for meta-evaluation.
If the batch only contains one example, the split will result in an empty tensor. This is why I dropped the last batch.

#### Observations
After overcoming these problems, I was able to train a model on the CONLL2003 dataset. Training the model with MLC
roughly takes 10x longer than training the baseline model using the huggingface transformers library. This is due to
multiple `backwards()` calls in the MLC algorithm. It also takes a lot of epochs for the meta and main model to learn.

In order to reproduce the text classification results from the paper, I would need their hyperparameters. 
It also does not help that the authors averaged the results over all the noise levels and types. 
They also say in the paper that for each dataset all the different models are trained for the same number of
epochs and refer to the repository, but it does not include the relevant information about the hyperparameters or
number of epochs.

Currently, the results on CONLL2003 are not as good as expected. The transformer based baseline model performs better 
than the MLC model on most noise levels, when it is trained for just 2 epochs. In fact, the baseline model performs 
surprisingly well on the corrupted training data. I suspect that this is due to the fact that the transformer model
is already very powerful and has a good grasp of PERSON, ORG and LOC entities even before training on CONLL2003.

When training for more epochs the baseline model quickly overfits on the corrupted training data, while the MLC model is 
still able to improve. Keep in mind though that I have not tried to optimize the hyperparameters for the MLC model, and 
it seems that the MLC model still has room to improve with more training epochs.
On CONLL2003 with 0.8 corruption probability the MLC model based on DistilBERT had a micro F1 of 0.549 on the test set 
after training for 15 epochs, compared to a micro F1 of 0.657 after training for 30 epochs.

| Corruption probability | Baseline (2 epochs) | Baseline (5 epochs) | MLC (5 epochs) | MLC (15 epochs) | MLC (30 epochs) |
| ---------------------- | ------------------- | ------------------- | -------------- | --------------- | --------------- |
| 0.0                    |                     | 0.904               | 0.322          | 0.600           |                 |
| 0.1                    |                     | 0.887               |                | 0.620           |                 |
| 0.2                    |                     | 0.882               | 0.439          | 0.657           |                 |
| 0.3                    |                     | 0.865               |                | 0.653           |                 |
| 0.4                    |                     | 0.850               |                | 0.609           |                 |
| 0.5                    |                     | 0.814               |                | 0.592           |                 |
| 0.6                    |                     | 0.787               | 0.087          | 0.558           |                 |
| 0.7                    |                     | 0.720               |                | 0.560           |                 |
| 0.8                    | 0.806               | 0.551               |                | 0.549           | 0.657           |
| 0.9                    | 0.647               | 0.311               |                | 0.556           |                 |



The results on the real word noise dataset (NoisyNER with label set 7) seem to be promising though. The MLC model based 
on EstBERT had a micro F1 of 0.71 on the test set after training for 30 epochs, while the baseline model (also based on EstBERT) only had 
a micro F1 of 0.63 after training for 5 epochs.

#### Potential next steps
- Do hyperparameter search for MLC models to find better hyperparameters for the main model and the meta model
- Train the MLC model for more epochs to see if it can improve further
- Add more datasets to the experiments
- Make label noise deterministic, so that the results are reproducible
- Repeat experiments using FLIP noise. Until now, I have used uniform noise, where the corrupted label might also happen 
to be the original label

## Citation

If you find MLC useful, please cite the following paper

```
@inproceedings{zheng2021mlc,
  title={Meta Label Correction for Noisy Label Learning},
  author={Zheng, Guoqing and Awadallah, Ahmed Hassan and Dumais, Susan},  
  journal={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  volume={35},
  year={2021},
}
```
For any questions, please submit an issue or contact [zheng@microsoft.com](zheng@microsoft.com). 

This repository is released under MIT License. (See [LICENSE](LICENSE))
