import argparse
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataloader import CAT_ConceptLinkingDataset
from model import ConceptLinkingClassifier
from utils import special_token_list

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str,
                             required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--pretrain_model_from_path", required=False, default="",
                             help="pretrained model from a checkpoint")
    group_model.add_argument("--pretrain_tokenizer_from_path", required=False, default="",
                             help="pretrained tokenizer from a checkpoint")
    group_model.add_argument("--prompt", required=False, default=1, type=int,
                             help="The number of prompt to use in dataloader", choices=[1, 2])
    group_model.add_argument("--candidate_num", default=9, required=False, type=int,
                             help="Number of candidates to use for in-context learning")

    # training-related args
    group_trainer = parser.add_argument_group("training configs")
    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                               help="device")
    group_trainer.add_argument("--gpu", default=0, type=int, required=False,
                               help="gpu number")
    group_trainer.add_argument("--optimizer", default='ADAMW', type=str, required=False,
                               help="optimizer")
    group_trainer.add_argument("--lr", default=5e-7, type=float, required=False,
                               help="learning rate")
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                               help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                               help="test batch size")
    group_trainer.add_argument("--epochs", default=50, type=int, required=False,
                               help="number of epochs")
    group_trainer.add_argument("--max_length", default=25, type=int, required=False,
                               help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_every", default=100, type=int, required=False,
                               help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                               help="whether to use special token to represent relation.")

    # IO-related
    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="CAT_Experiments", type=str, required=False,
                            help="where to output.")
    group_data.add_argument("--log_dir", default='logs', type=str, required=False,
                            help="Where to save logs.")
    group_data.add_argument("--experiment_name", default='', type=str, required=False,
                            help="A special name that will be prepended to the dir name of the output.")

    group_data.add_argument("--seed", default=621, type=int, required=False, help="random seed")

    args = parser.parse_args()

    return args


def main():
    # get all arguments
    args = parse_args()

    # Check device status
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CUDA available:', torch.cuda.is_available())
    print(torch.cuda.get_device_name())
    print('Device number:', torch.cuda.device_count())
    print(torch.cuda.get_device_properties(device))
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    experiment_name = args.experiment_name

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm),
                                                       f"bs{args.batch_size}", f"lr{args.lr}",
                                                       f"evalstep{args.eval_every}",
                                                       f"prompt{args.prompt}",
                                                       f"candNum{args.candidate_num}"]) + experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("concept-linking-{}".format(args.ptlm))
    handler = logging.FileHandler(os.path.join(save_dir, f"log_seed_{args.seed}.txt"))
    logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model

    model = ConceptLinkingClassifier(args.ptlm)

    if args.pretrain_tokenizer_from_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_from_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm)
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })

    model.model.resize_token_embeddings(len(tokenizer))
    if args.pretrain_model_from_path:
        model.load_state_dict(torch.load(args.pretrain_model_from_path))
    model = model.to(args.device)

    # load data
    full_dataset = pd.read_csv('../../data/head_annotated.csv', index_col=None)
    pseudo_dataset = pd.read_csv('./initial_pseudo_label/TOTAL_prediction_without_annotated.csv', index_col=None)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    dev_dataset = CAT_ConceptLinkingDataset(full_dataset, pseudo_dataset, tokenizer, max_length=args.max_length,
                                            split='dev', prompt_num=args.prompt, model=args.ptlm, predict=True)
    tst_dataset = CAT_ConceptLinkingDataset(full_dataset, pseudo_dataset, tokenizer, max_length=args.max_length,
                                            split='tst', prompt_num=args.prompt, model=args.ptlm, predict=True)
    training_set = CAT_ConceptLinkingDataset(full_dataset, pseudo_dataset, tokenizer, max_length=args.max_length,
                                             split='trn', prompt_num=args.prompt, model=args.ptlm, predict=True)

    training_loader = DataLoader(training_set, **train_params, drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, **val_params, drop_last=False)
    tst_dataloader = DataLoader(tst_dataset, **val_params, drop_last=False)

    # model training

    if args.optimizer == "ADAM":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAMW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError

    criterion = torch.nn.CrossEntropyLoss()
    best_val_auc, best_val_acc, best_val_tuned_acc, best_tst_auc, best_tst_acc, best_tst_tuned_acc = 0, 0, 0, 0, 0, 0
    best_val_threshold, best_tst_threshold = 0, 0

    model.eval()
    progress_bar = tqdm(range(len(training_loader) + len(dev_dataloader) + len(tst_dataloader)))

    iteration = 0

    split_name = ['dev', 'tst', 'trn']
    split_dataset = [dev_dataset, tst_dataset, training_set]
    prediction_csv_list = []
    for ind, loader in enumerate([dev_dataloader, tst_dataloader, training_loader]):
        predict_score = torch.tensor([], dtype=torch.float)
        for iteration, data in enumerate(loader, iteration + 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            tokens = {"input_ids": ids, "attention_mask": mask}

            outputs_logits = model(tokens)
            logits = torch.softmax(outputs_logits, dim=1)
            pred_prob = logits[:, 1]
            predict_score = torch.cat((predict_score, pred_prob.detach().clone().cpu()))

            progress_bar.update(1)

        assert len(predict_score.tolist()) == len(split_dataset[ind])
        prediction_csv = split_dataset[ind].merge_predicted_score(predict_score.tolist())
        prediction_csv.to_csv(os.path.join(save_dir, '{}_prediction.csv'.format(split_name[ind])), index=False)
        prediction_csv_list.append(prediction_csv)
        print("Successfully predicted {} split".format(split_name[ind]))

    total_csv = pd.concat(prediction_csv_list, ignore_index=True)
    total_csv.to_csv(os.path.join(save_dir, 'TOTAL_prediction.csv'), index=False)


if __name__ == "__main__":
    main()
