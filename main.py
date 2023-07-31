import numpy as np
import copy
import sys
import argparse
import os
from logger import get_logger
from tqdm import tqdm
from collections import deque
import json

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from mlc import step_hmlc_K
from mlc_utils import tocuda, DummyScheduler, cross_entropy

from models import *
from meta_models import *


# //////////////////////// defining model ////////////////////////

def get_data(dataset, gold_fraction, corruption_prob, get_C):
    if dataset == 'cifar10' or dataset == 'cifar100':
        sys.path.append('CIFAR')

        from data_helper_cifar import prepare_data
        args.use_mwnet_loader = True  # use exactly the same loader as in the mwnet paper
        logger.info('================= Use the same dataloader as in MW-Net =========================')
        return prepare_data(gold_fraction, corruption_prob, get_C, args)
    elif dataset == 'clothing1m':
        sys.path.append('CLOTHING1M')

        from data_helper_clothing1m import prepare_data
        return prepare_data(args)
    elif dataset == 'conll2003':
        sys.path.append('conll_2003')
        from data_helper_conll2003 import prepare_data
        return prepare_data(gold_fraction, corruption_prob, args)
    elif dataset == 'yelp_reviews':
        sys.path.append('yelp_reviews')
        from data_helper_yelp_reviews import prepare_data
        return prepare_data(gold_fraction, corruption_prob, args)
    elif dataset == 'noisyner':
        sys.path.append('noisyner')
        from data_helper_noisyner import prepare_data
        return prepare_data(gold_fraction, args)


def build_models(dataset, num_classes):
    cls_dim = args.cls_dim  # input label embedding dimension

    if dataset in ['cifar10', 'cifar100']:
        from CIFAR.resnet import resnet32

        # main net
        model = resnet32(num_classes)
        main_net = model

        # meta net
        hx_dim = 64  # 0 if isinstance(model, WideResNet) else 64 # 64 for resnet-32
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)

    elif dataset == 'clothing1m':  # use pretrained ResNet-50 model
        model = ResNet50(num_classes)
        main_net = model

        hx_dim = 2048  # from resnet50
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)

    elif dataset == 'conll2003' or dataset == 'noisyner':
        model = TransformerTokenClassifier(args.pretrained_model_name_or_path, num_classes, args.dropout)
        main_net = model
        hx_dim = 768  # from transformer
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)

    elif dataset == 'yelp_reviews':
        if args.model_name_or_path.startswith("distilbert"):
            model = DistilBertSequenceClassifier(args.pretrained_model_name_or_path, num_classes, args.dropout)
        elif args.model_name_or_path.startswith("bert"):
            model = BertSequenceClassifier(args.pretrained_model_name_or_path, num_classes, args.dropout)
        else:
            raise NotImplementedError("There is only support for distilbert or bert")
        main_net = model
        hx_dim = 768  # from transformer
        meta_net = MetaNet(hx_dim, cls_dim, 128, num_classes, args)

    main_net = main_net.cuda()
    meta_net = meta_net.cuda()

    logger.info('========== Main model ==========')
    logger.info(model)
    logger.info('========== Meta model ==========')
    logger.info(meta_net)

    return main_net, meta_net


def setup_training(main_net, meta_net, exp_id=None):
    # ============== setting up from scratch ===================
    # set up optimizers and schedulers
    # meta net optimizer
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=args.meta_lr,
                                      weight_decay=0,  # args.wdecay, # meta should have wdecay or not??
                                      amsgrad=True, eps=args.opt_eps)
    meta_scheduler = DummyScheduler(meta_optimizer)

    # main net optimizer
    main_params = main_net.parameters()

    if args.optimizer == 'adam':
        main_opt = torch.optim.Adam(main_params, lr=args.main_lr, weight_decay=args.wdecay, amsgrad=True,
                                    eps=args.opt_eps)
    elif args.optimizer == 'sgd':
        main_opt = torch.optim.SGD(main_params, lr=args.main_lr, weight_decay=args.wdecay, momentum=args.momentum)

    if args.dataset in ['cifar10', 'cifar100']:
        # follow MW-Net setting
        main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[80, 100], gamma=0.1)
    elif args.dataset in ['clothing1m']:
        main_schdlr = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=[5], gamma=0.1)
    else:
        main_schdlr = DummyScheduler(main_opt)

    last_epoch = -1
    # what is the purpose of returning the main_net and meta_net if they are not modified?
    return main_net, meta_net, main_opt, meta_optimizer, main_schdlr, meta_scheduler, last_epoch


def uniform_mix_C(num_classes, mixing_ratio):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(num_classes, corruption_prob):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(args.seed)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


# //////////////////////// run experiments ////////////////////////
def run():
    corruption_fnctn = uniform_mix_C if args.corruption_type == 'unif' else flip_labels_C
    results_filename = '_'.join([args.dataset, args.method, args.corruption_type, args.runid, str(args.epochs), str(args.seed),
                                 str(args.data_seed)])

    results = {}

    gold_fractions = [0.02]

    if args.gold_fraction != -1:
        assert 0 <= args.gold_fraction <= 1, 'Wrong gold fraction!'
        gold_fractions = [args.gold_fraction]

    corruption_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if args.corruption_prob != -1:  # specified one corruption_prob
        assert 0 <= args.corruption_prob <= 1, 'Wrong noise level!'
        corruption_probs = [args.corruption_prob]

    for gold_fraction in gold_fractions:
        results[gold_fraction] = {}
        for corruption_prob in corruption_probs:
            # //////////////////////// load data //////////////////////////////
            # use data_seed her
            gold_loader, silver_loader, valid_loader, test_loader, num_classes = get_data(args.dataset, gold_fraction,
                                                                                          corruption_prob,
                                                                                          corruption_fnctn)

            # //////////////////////// build main_net and meta_net/////////////
            main_net, meta_net = build_models(args.dataset, num_classes)

            # //////////////////////// train and eval model ///////////////////
            exp_id = '_'.join([results_filename, str(gold_fraction), str(corruption_prob)])
            test_metrics, baseline_metrics = train_and_test(
                main_net, meta_net, gold_loader, silver_loader, valid_loader, test_loader,
                args.model_save_dir, exp_id, perf_metric=args.metric,
                ignore_majority_negative_class=args.ignore_majority_negative_class, target_names=args.target_names)

            results[gold_fraction][corruption_prob] = {}
            results[gold_fraction][corruption_prob]['method'] = test_metrics
            results[gold_fraction][corruption_prob]['baseline'] = baseline_metrics  # will always be 0
            logger.info(' '.join(['Gold fraction:', str(gold_fraction), '| Corruption level:', str(corruption_prob),
                                  '| Method acc:', str(results[gold_fraction][corruption_prob]['method']),
                                  '| Baseline acc:', str(results[gold_fraction][corruption_prob]['baseline'])]))
            logger.info('')

    if not os.path.isdir('results'):
        os.mkdir('results')
    with open('results/' + results_filename  + '.json', 'w', encoding='utf8') as file:
        json.dump(results, file)
    logger.info("Wrote results in file: " + results_filename  + ".json")


def test(main_net, test_loader, ignore_majority_negative_class=False, target_names=None):  # this could be eval or test
    # //////////////////////// evaluate method ////////////////////////

    # forward
    main_net.eval()

    labels = []
    predictions = []

    for idx, batch in enumerate(test_loader):
        if isinstance(batch, list):  # Computer vision datasets return images and labels as lists
            *data, target = batch
            data, target = tocuda(data), tocuda(target)
        else:  # NLP datasets return dictionaries with keys 'input_ids', 'token_type_ids', 'attention_mask', 'labels'
            batch = {k: v.cuda() for k, v in batch.items()}
            data = batch
            target = batch['labels']

        # forward
        with torch.no_grad():
            output = main_net(data)

        labels.append(target.flatten())
        predictions.append(output.argmax(-1).flatten())

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    # sklearn
    y_true = labels.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    (
        overall_precision,
        overall_recall,
        overall_f1_score,
        _,
    ) = precision_recall_fscore_support(y_true, y_pred, average="micro",
                                        labels=np.unique(y_true)[1:] if ignore_majority_negative_class else None)
    overall_accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1_score,
        "overall_accuracy": overall_accuracy,
    }
    cls_report = classification_report(y_true, y_pred, digits=4,
                                       labels=np.unique(y_true)[1:] if ignore_majority_negative_class else None,
                                        target_names=target_names, output_dict=True)

    # set back to train
    main_net.train()

    return metrics, cls_report


####################################################################################################
###  training code 
####################################################################################################
def train_and_test(main_net, meta_net, gold_loader, silver_loader, valid_loader, test_loader, model_save_dir, exp_id=None,
                   perf_metric="overall_accuracy", ignore_majority_negative_class=False,target_names=None):
    writer = SummaryWriter(args.logdir + '/' + exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main_net, meta_net, main_opt, meta_opt, main_schdlr, meta_schdlr, last_epoch = setup_training(main_net, meta_net,
                                                                                                  exp_id)

    # //////////////////////// switching on training mode ////////////////////////
    meta_net.train()
    main_net.train()

    # set up statistics 
    best_params = None
    best_main_opt_sd = None
    best_main_schdlr_sd = None

    best_meta_params = None
    best_meta_opt_sd = None
    best_meta_schdlr_sd = None
    best_val_metric = float('inf')

    val_metric_queue = deque()
    # set done

    args.dw_prev = [0 for _ in meta_net.parameters()]  # 0 for previous iteration
    args.steps = 0

    for epoch in tqdm(range(last_epoch + 1, args.epochs)):  # change to epoch iteration
        logger.info('Epoch %d:' % epoch)

        for i, batch in enumerate(silver_loader):
            if isinstance(batch, list):     # Computer vision datasets return images and labels as lists
                *data_s, target_s = batch
                data_s, target_s_ = tocuda(data_s), tocuda(target_s)
            else:  # NLP datasets return dictionaries with keys 'input_ids', 'token_type_ids', 'attention_mask', 'labels'
                batch = {k: v.cuda() for k, v in batch.items()}
                data_s = batch
                target_s_ = batch['labels']
            gold_batch = next(gold_loader)
            if isinstance(batch, list):
                *data_g, target_g = gold_batch
                data_g, target_g = tocuda(data_g), tocuda(target_g)
            else:
                gold_batch = {k: v.cuda() for k, v in gold_batch.items()}
                data_g = gold_batch
                target_g = gold_batch['labels']

            # bi-level optimization stage
            eta = main_schdlr.get_lr()[0]
            if args.method == 'hmlc_K':
                loss_g, loss_s = step_hmlc_K(main_net, main_opt, meta_net, meta_opt, cross_entropy,
                                             data_s, target_s_, data_g, target_g,
                                             None, None,
                                             eta, args)
            elif args.method == 'hmlc_K_mix':   # default option
                # split the clean set to two, one for training and the other for meta-evaluation
                gbs = int(target_g.size(0) / 2)
                if type(data_g) is list:
                    data_c = [x[gbs:] for x in data_g]
                    data_g = [x[:gbs] for x in data_g]
                elif type(data_g) is dict:  # NLP datasets
                    data_c = {k: v[gbs:] for k, v in data_g.items()}
                    data_g = {k: v[:gbs] for k, v in data_g.items()}
                else:
                    data_c = data_g[gbs:]
                    data_g = data_g[:gbs]

                target_c = target_g[gbs:]
                target_g = target_g[:gbs]
                assert target_c.numel() > 0 and target_g.numel() > 0, 'Empty clean or gold data!'
                loss_g, loss_s = step_hmlc_K(main_net, main_opt, meta_net, meta_opt, cross_entropy,
                                             data_s, target_s_, data_g, target_g,
                                             data_c, target_c,
                                             eta, args)

            args.steps += 1
            if i % args.every == 0:
                writer.add_scalar('train/loss_g', loss_g.item(), args.steps)
                writer.add_scalar('train/loss_s', loss_s.item(), args.steps)

                ''' get entropy of predictions from meta-net '''
                logit_s, x_s_h = main_net(data_s, return_h=True)
                pseudo_target_s = meta_net(x_s_h.detach(), target_s_).detach()
                entropy = -(pseudo_target_s * torch.log(pseudo_target_s + 1e-10)).sum(-1).mean()

                writer.add_scalar('train/meta_entropy', entropy.item(), args.steps)

                main_lr = main_schdlr.get_lr()[0]
                meta_lr = meta_schdlr.get_lr()[0]
                writer.add_scalar('train/main_lr', main_lr, args.steps)
                writer.add_scalar('train/meta_lr', meta_lr, args.steps)
                writer.add_scalar('train/gradient_steps', args.gradient_steps, args.steps)

                logger.info(
                    'Iteration %d loss_s: %.4f\tloss_g: %.4f\tMeta entropy: %.3f\tMain LR: %.8f\tMeta LR: %.8f' % (
                        i, loss_s.item(), loss_g.item(), entropy.item(), main_lr, meta_lr))

        # PER EPOCH PROCESSING

        # lr scheduler
        main_schdlr.step()
        # scheduler.step()

        # evaluation on validation set
        val_metrics, val_cls_report = test(main_net, valid_loader, ignore_majority_negative_class, target_names)

        if perf_metric == "overall_accuracy":
            val_metric = val_metrics["overall_accuracy"]
            logger.info('Val acc: %.4f' % val_metric)
            if args.local_rank <= 0:  # single GPU or GPU 0
                writer.add_scalar('train/val_acc', val_metric, epoch)
        else:   # we will assume precision, recall, f1
            val_metric = val_metrics["overall_f1"]
            logger.info('Val p: %.4f/r: %.4f/f1: %.4f' %
                        (val_metrics["overall_precision"], val_metrics["overall_recall"], val_metrics["overall_f1"]))
            if args.local_rank <= 0:  # single GPU or GPU 0
                for metric in val_metrics:
                    writer.add_scalar('train/val_' + metric, val_metrics[metric], epoch)

        if len(val_metric_queue) == args.queue_size:  # keep at most this number of records
            # remove the oldest record
            val_metric_queue.popleft()

        val_metric_queue.append(-val_metric)

        avg_val_metric = sum(list(val_metric_queue)) / len(val_metric_queue)
        if avg_val_metric < best_val_metric:
            best_val_metric = avg_val_metric

            best_params = copy.deepcopy(main_net.state_dict())

            best_main_opt_sd = copy.deepcopy(main_opt.state_dict())
            best_main_schdlr_sd = copy.deepcopy(main_schdlr.state_dict())

            best_meta_params = copy.deepcopy(meta_net.state_dict())
            best_meta_opt_sd = copy.deepcopy(meta_opt.state_dict())
            best_meta_schdlr_sd = copy.deepcopy(meta_schdlr.state_dict())

            # dump best to file also
            ####################### save best models so far ###################

            logger.info('Saving best models...')

            torch.save({
                'epoch': epoch,
                'val_metric': best_val_metric,
                'main_net': best_params,
                'main_opt': best_main_opt_sd,
                'main_schdlr': best_main_schdlr_sd,
                'meta_net': best_meta_params,
                'meta_opt': best_meta_opt_sd,
                'meta_schdlr': best_meta_schdlr_sd
            }, f'{model_save_dir}/{exp_id}_best.pth')
        if perf_metric == "overall_accuracy":
            writer.add_scalar('train/val_acc_best', -best_val_metric, epoch)  # write current best val_acc to tensorboard
        else:
            writer.add_scalar('train/val_f1_best', -best_val_metric, epoch)  # write current best val_f1 to tensorboard

    # //////////////////////// evaluating method ////////////////////////
    main_net.load_state_dict(best_params)
    test_metrics, test_cls_report = test(main_net, test_loader, ignore_majority_negative_class, target_names)  # evaluate best params picked from validation
    print(json.dumps(test_cls_report, indent=4))
    if perf_metric == "overall_accuracy":
        test_metric = test_metrics["overall_accuracy"]
        logger.info('Test acc: %.4f' % test_metric)
        writer.add_scalar('test/acc', test_metric,
                      args.steps)  # this test_acc should be roughly the best as it's taken from the best iteration
    else:
        logger.info('Test p: %.4f/r: %4.f/f1: %.4f' %
                    (test_metrics["overall_precision"], test_metrics["overall_recall"], test_metrics["overall_f1"]))
        for metric in test_metrics:
            writer.add_scalar('test/' + metric, test_metrics[metric], args.steps)

    return test_metrics, 0       # returns 0 for baseline accuracy/metric?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLC Training Framework')
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'cifar100', 'clothing1m', 'conll2003', 'yelp_reviews', 'noisyner'], default='cifar10')
    parser.add_argument('--method', default='hmlc_K_mix', type=str, choices=['hmlc_K_mix', 'hmlc_K'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_seed', type=int, default=1)
    parser.add_argument('--epochs', '-e', type=int, default=75, help='Number of epochs to train.')
    parser.add_argument('--num_iterations', default=100000, type=int)
    parser.add_argument('--every', default=100, type=int, help='Eval interval (default: 100 iters)')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_bs', default=100, type=int, help='batch size')
    parser.add_argument('--gold_bs', type=int, default=32)
    parser.add_argument('--cls_dim', type=int, default=64, help='Label embedding dim (Default: 64)')
    parser.add_argument('--grad_clip', default=0.0, type=float, help='max grad norm (default: 0, no clip)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--main_lr', default=3e-4, type=float, help='lr for main net')
    parser.add_argument('--meta_lr', default=3e-5, type=float, help='lr for meta net')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='eps for optimizers')
    # parser.add_argument('--tau', default=1, type=float, help='tau')
    parser.add_argument('--wdecay', default=5e-4, type=float, help='weight decay (default: 5e-4)')

    # noise parameters
    parser.add_argument('--corruption_type', default='unif', type=str, choices=['unif', 'flip'])
    parser.add_argument('--corruption_prob', default='-1', type=float, help='Corruption probability')
    parser.add_argument('--gold_fraction', default='-1', type=float, help='Gold fraction')

    parser.add_argument('--skip', default=False, action='store_true', help='Skip link for LCN (default: False)')
    parser.add_argument('--sparsemax', default=False, action='store_true',
                        help='Use softmax instead of softmax for meta model (default: False)')
    parser.add_argument('--tie', default=False, action='store_true',
                        help='Tie label embedding to the output classifier output embedding of metanet (default: False)')

    parser.add_argument('--runid', default='exp', type=str)
    parser.add_argument('--queue_size', default=1, type=int, help='Number of iterations before to compute mean loss_g')

    ############## LOOK-AHEAD GRADIENT STEPS FOR MLC ##################
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help='Number of look-ahead gradient steps for meta-gradient (default: 1)')

    # CIFAR
    # Positional arguments
    parser.add_argument('--data_path', default='data', type=str, help='Root for the datasets.')
    # Optimization options
    parser.add_argument('--nosgdr', default=False, action='store_true', help='Turn off SGDR.')

    # NLP
    parser.add_argument('--pretrained_model_name_or_path', default='distilbert-base-cased', type=str,
                        help='Path to pre-trained model or shortcut name (default: bert-base-cased)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout for transformer (default: 0.1)')
    parser.add_argument('--max_seq_length', default=128, type=int, help='Max sequence length (default: 128)')
    parser.add_argument('--metric', default='overall_accuracy', type=str,
                        choices=['overall_accuracy', 'overall_f1'],
                        help='Performance metric for training and evaluation.')
    parser.add_argument('--model_save_dir', type=str, required=True, help='Where to save model')
    parser.add_argument('--configuration', type=str, help='Dataset configuration for noisyner')

    # Acceleration
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--logdir', type=str, default='runs', help='Log folder.')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank (-1 for local training)')

    args = parser.parse_args()

    if args.dataset == "conll2003":
        args.metric = "overall_f1"
        args.ignore_majority_negative_class = True
        args.target_names = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    elif args.dataset == "noisyner":
        args.metric = "overall_f1"
        args.ignore_majority_negative_class = True
        args.target_names = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    elif args.dataset == "yelp_reviews":
        args.metric = "overall_accuracy"
        args.ignore_majority_negative_class = False
        args.target_names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    else:
        args.metric = "overall_accuracy"
        args.ignore_majority_negative_class = False
        args.target_names = None

    # //////////////// set logging and model outputs /////////////////
    filename = '_'.join([args.dataset, args.method, args.corruption_type, args.runid, str(args.epochs), str(args.seed),
                         str(args.data_seed)])
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logfile = 'logs/' + filename + '.log'
    logger = get_logger(logfile, args.local_rank)

    if not os.path.isdir(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    # ////////////////////////////////////////////////////////////////

    logger.info(args)
    logger.info('CUDA available:' + str(torch.cuda.is_available()))

    # cuda set up
    torch.cuda.set_device(0)  # local GPU

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    run()
