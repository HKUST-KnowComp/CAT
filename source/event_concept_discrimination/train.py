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

from dataloader import ConceptLinkingAnnotatedDataset
from evaluate import evaluate
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
                             help="pretrain this model from a checkpoint")
    group_model.add_argument("--pretrain_tokenizer_from_path", required=False, default="",
                             help="pretrain tokenizer from a checkpoint")
    group_model.add_argument("--prompt", required=False, default=1, type=int,
                             help="The number of prompt to use in dataloader", choices=[1, 2])

    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                               help="device")
    group_trainer.add_argument("--gpu", default=0, type=int, required=False,
                               help="gpu number")
    group_trainer.add_argument("--optimizer", default='ADAMW', type=str, required=False,
                               help="optimizer")
    group_trainer.add_argument("--lr", default=5e-6, type=float, required=False,
                               help="learning rate")
    group_trainer.add_argument("--lrdecay", default=1, type=float, required=False,
                               help="learning rate decay every x steps")
    group_trainer.add_argument("--decay_every", default=500, type=int, required=False,
                               help="show test result every x steps")
    group_trainer.add_argument("--batch_size", default=64, type=int, required=False,
                               help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                               help="test batch size")
    group_trainer.add_argument("--epochs", default=50, type=int, required=False,
                               help="number of epochs")
    group_trainer.add_argument("--steps", default=-1, type=int, required=False,
                               help="the number of iterations to train model on labeled data. used for the case training model less than 1 epoch")
    group_trainer.add_argument("--max_length", default=25, type=int, required=False,
                               help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_every", default=50, type=int, required=False,
                               help="eval on test set every x steps.")
    group_trainer.add_argument("--relation_as_special_token", action="store_true",
                               help="whether to use special token to represent relation.")

    # IO-related
    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="CAT_results", type=str, required=False,
                            help="where to output.")
    group_data.add_argument("--train_csv_path", default='', type=str, required=False)
    group_data.add_argument("--evaluation_file_path", default="data/evaluation_set.csv",
                            type=str, required=False)
    group_data.add_argument("--save_best_model", action="store_true",
                            help="whether to save the best model.")
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

    experiment_name = args.experiment_name

    save_dir = os.path.join(args.output_dir, "_".join([os.path.basename(args.ptlm),
                                                       f"bs{args.batch_size}", f"lr{args.lr}",
                                                       f"evalstep{args.eval_every}",
                                                       f"prompt{args.prompt}"]) + experiment_name)
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

    model = ConceptLinkingClassifier(args.ptlm).to(args.device)

    if args.pretrain_tokenizer_from_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_from_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm)

        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
    model.model.resize_token_embeddings(len(tokenizer))
    if args.pretrain_model_from_path:
        ckpt = torch.load(args.pretrain_model_from_path)
        model.load_state_dict(ckpt)

    # load data
    full_dataset = pd.read_csv('../../data/head_annotated.csv', index_col=None)

    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 5
    }

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }

    training_set = ConceptLinkingAnnotatedDataset(full_dataset, tokenizer, args.max_length, split='trn',
                                                  prompt_num=args.prompt, model=args.ptlm)
    dev_dataset = ConceptLinkingAnnotatedDataset(full_dataset, tokenizer, args.max_length, split='dev',
                                                 prompt_num=args.prompt, model=args.ptlm)
    tst_dataset = ConceptLinkingAnnotatedDataset(full_dataset, tokenizer, args.max_length, split='tst',
                                                 prompt_num=args.prompt, model=args.ptlm)

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
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

    model.train()
    progress_bar = tqdm(range(len(training_loader) * args.epochs), desc="Training")

    iteration = 0
    for e in range(args.epochs):

        for iteration, data in enumerate(training_loader, iteration + 1):
            # the iteration starts from 1.

            y = data['label'].to(args.device, dtype=torch.long)

            ids = data['ids'].to(args.device, dtype=torch.long)
            mask = data['mask'].to(args.device, dtype=torch.long)
            tokens = {"input_ids": ids, "attention_mask": mask}

            logits = model(tokens)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

            if args.eval_every > 0 and iteration % args.eval_every == 0:
                model.eval()

                eval_auc, eval_acc, eval_tuned_acc, eval_tuned_threshold, _ = evaluate(tokenizer, model, args.device,
                                                                                       dev_dataloader, logger, 0.5,
                                                                                       finetune_threshold=False,
                                                                                       model_type=args.ptlm)
                assert _ == len(dev_dataset)
                print('Current dev eval: AUC ', eval_auc, 'Best AUC: ', best_val_auc)
                print('Current dev eval: ACC ', eval_acc, 'Best ACC: ', best_val_acc)
                print('Current dev eval: Tuned_ACC ', eval_tuned_acc, 'Best Tuned_ACC: ', best_val_tuned_acc)
                updated = []
                if eval_auc > best_val_auc:
                    updated.append("AUC")
                    best_val_auc = eval_auc
                    torch.save(model.state_dict(), save_dir + f"/best_val_auc_model_seed_{args.seed}.pth")
                    tokenizer.save_pretrained(save_dir + "/best_val_auc_tokenizer")
                if eval_acc > best_val_acc:
                    updated.append("ACC")
                    best_val_acc = eval_acc
                    # torch.save(model.state_dict(), save_dir + f"/best_val_acc_model_seed_{args.seed}.pth")
                    # tokenizer.save_pretrained(save_dir + "/best_val_acc_tokenizer")
                if eval_tuned_acc > best_val_tuned_acc:
                    updated.append("Tuned ACC")
                    best_val_tuned_acc = eval_tuned_acc
                    best_val_threshold = eval_tuned_threshold
                    # torch.save(model.state_dict(), save_dir + f"/best_val_tunedAcc_model_seed_{args.seed}.pth")
                    # tokenizer.save_pretrained(save_dir + "/best_tuned_tunedAcc_tokenizer")

                if updated:
                    tst_auc, tst_acc, tst_tuned_acc, tst_tuned_threshold, _ = evaluate(tokenizer, model, args.device,
                                                                                       tst_dataloader, logger,
                                                                                       eval_tuned_threshold,
                                                                                       finetune_threshold=False,
                                                                                       model_type=args.ptlm)
                    print('Current tst eval: AUC ', tst_auc, 'Best AUC: ', best_tst_auc)
                    print('Current tst eval: ACC ', tst_acc, 'Best ACC: ', best_tst_acc)
                    print('Current tst eval: Tuned_ACC ', tst_tuned_acc, 'Best Tuned_ACC: ', best_tst_tuned_acc)

                    best_tst_acc = tst_acc
                    # torch.save(model.state_dict(), save_dir + f"/best_tst_acc_model_seed_{args.seed}.pth")
                    # tokenizer.save_pretrained(save_dir + "/best_tst_acc_tokenizer")

                    best_tst_auc = tst_auc
                    torch.save(model.state_dict(), save_dir + f"/best_tst_auc_model_seed_{args.seed}.pth")
                    tokenizer.save_pretrained(save_dir + "/best_tst_auc_tokenizer")

                    best_tst_tuned_acc = tst_tuned_acc
                    best_tst_threshold = tst_tuned_threshold
                    # torch.save(model.state_dict(), save_dir + f"/best_tst_tunedAcc_model_seed_{args.seed}.pth")
                    # tokenizer.save_pretrained(save_dir + "/best_tst_tunedAcc_tokenizer")

                    logger.info(
                        f"Validation {updated} reached best at epoch {e} step {iteration}, evaluating on test set")
                    logger.info(
                        "Best Test Scores: AUC: {}\tACC: {}\t Tuned ACC: {}\t Threshold: {}\n"
                        "Best Dev Scores: AUC: {}\tACC: {}\t Tuned ACC: {}\t Threshold: {}".format(best_tst_auc,
                                                                                                   best_tst_acc,
                                                                                                   best_tst_tuned_acc,
                                                                                                   best_tst_threshold,
                                                                                                   best_val_auc,
                                                                                                   best_val_acc,
                                                                                                   best_val_tuned_acc,
                                                                                                   best_val_threshold))
                model.train()


if __name__ == "__main__":
    main()
