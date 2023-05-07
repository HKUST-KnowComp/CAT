import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from dataloader import TripleVerificationDataset
from evaluate import evaluate
from model import TripleVerificationClassifier
from utils import special_token_list

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str,
                             required=False, help="choose from huggingface PTLM")
    group_model.add_argument("--automatic_find_semi_supervise", default=0, choices=[0, 1], type=int,
                             help="Search pretrain model from existing semi-supervised checkpoint")
    group_model.add_argument("--pretrain_model_from_path", required=False, default="",
                             help="pretrain this model from a checkpoint")
    group_model.add_argument("--pretrain_tokenizer_from_path", required=False, default="",
                             help="pretrain tokenizer from a checkpoint")
    group_model.add_argument("--use_atomic", default=0, type=int, choices=[0, 1], help="Whether to use atomic")
    group_model.add_argument("--candidate_num", default=2, type=int, help="Number of candidates to use")
    group_model.add_argument("--mode", default='semi-supervised', choices=['supervised', 'semi-supervised'])

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
    group_trainer.add_argument("--batch_size", default=32, type=int, required=False,
                               help="batch size")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                               help="test batch size")
    group_trainer.add_argument("--epochs", default=30, type=int, required=False,
                               help="number of epochs")
    group_trainer.add_argument("--max_length", default=50, type=int, required=False,
                               help="max_seq_length of h+r+t")
    group_trainer.add_argument("--eval_every", default=25, type=int, required=False,
                               help="eval on test set every x steps.")

    # IO-related
    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="CAT_results", type=str, required=False,
                            help="where to output.")
    group_data.add_argument("--experiment_name", default='', type=str, required=False,
                            help="A special name that will be prepended to the dir name of the output.")

    group_data.add_argument("--seed", default=621, type=int, required=False, help="random seed")

    args = parser.parse_args()
    return args


def main():
    # get all arguments
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
                                                       f"atomic{args.use_atomic}",
                                                       f"candNum{args.candidate_num}", f"mode{args.mode}",
                                                       f"seed{args.seed}"]) + experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("triple_verifier-{}".format(args.ptlm))
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
    model = TripleVerificationClassifier(args.ptlm).to(args.device)

    if args.automatic_find_semi_supervise:
        tokenizer = AutoTokenizer.from_pretrained(
            "./CIICL_SemiSupervise_try/{}_bs32_lr5e-06_evalstep50_atomic0_candNum2_modesemi-supervised_seed{}/best_val_auc_tokenizer/".format(
                os.path.basename(args.ptlm), args.seed))
    elif args.pretrain_tokenizer_from_path:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_tokenizer_from_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ptlm)

        tokenizer.add_special_tokens({
            'additional_special_tokens': special_token_list,
        })
    model.model.resize_token_embeddings(len(tokenizer))
    if args.automatic_find_semi_supervise:
        ckpt = torch.load(
            "./CIICL_SemiSupervise_try/{}_bs32_lr5e-06_evalstep50_atomic0_candNum2_modesemi-supervised_seed{}/best_val_auc_model_seed_{}.pth".format(
                os.path.basename(args.ptlm), args.seed, args.seed))
        model.load_state_dict(ckpt)
    if args.pretrain_model_from_path:
        ckpt = torch.load(args.pretrain_model_from_path)
        model.load_state_dict(ckpt)

    # load data
    if args.mode == 'supervised':
        full_dataset = pd.read_csv('../../data/triple_annotated.csv', index_col=None)

        if args.automatic_find_semi_supervise:
            head_dataset = pd.read_csv('../../data/head_annotated.csv', index_col=None)
            head_annotated_positive = head_dataset[head_dataset.score >= 3].reset_index(drop=True)
            head_unlabeled_dataset = pd.read_csv('../../data/head_unlabeled_prediction.csv', index_col=None)
            head_unlabeled_positive = head_unlabeled_dataset[head_unlabeled_dataset.score >= 0.8].reset_index(drop=True)
            head_positive = pd.concat([head_annotated_positive, head_unlabeled_positive],
                                      ignore_index=True).drop_duplicates(subset=['head', 'concept_text']).reset_index(
                drop=True)
        else:
            head_dataset = pd.read_csv('../../data/head_annotated.csv', index_col=None)
            head_positive = head_dataset[head_dataset.score >= 3].reset_index(drop=True)

        training_set = TripleVerificationDataset(full_dataset, head_positive, tokenizer, args.max_length,
                                                 split='trn', model=args.ptlm, use_atomic=args.use_atomic,
                                                 candidate_num=args.candidate_num)
        dev_dataset = TripleVerificationDataset(full_dataset, head_positive, tokenizer, args.max_length,
                                                split='dev', model=args.ptlm, use_atomic=args.use_atomic,
                                                candidate_num=args.candidate_num)
        tst_dataset = TripleVerificationDataset(full_dataset, head_positive, tokenizer, args.max_length,
                                                split='tst', model=args.ptlm, use_atomic=args.use_atomic,
                                                candidate_num=args.candidate_num)

    else:
        head_dataset = pd.read_csv('../../data/head_annotated.csv', index_col=None)
        head_annotated_positive = head_dataset[head_dataset.score >= 3].reset_index(drop=True)
        head_unlabeled_dataset = pd.read_csv('../../data/head_unlabeled_prediction.csv', index_col=None)
        head_unlabeled_positive = head_unlabeled_dataset[head_unlabeled_dataset.score >= 0.8].reset_index(drop=True)
        head_positive = pd.concat([head_annotated_positive, head_unlabeled_positive],
                                  ignore_index=True).drop_duplicates(subset=['head', 'concept_text']).reset_index(
            drop=True)

        full_annotated_dataset = pd.read_csv('../../data/triple_annotated.csv', index_col=None)
        full_unlabeled_prediction = pd.read_csv('../../data/triple_unlabeled_prediction.csv', index_col=None)
        full_unlabeled_prediction = full_unlabeled_prediction[
            (full_unlabeled_prediction.score >= 0.99) | (full_unlabeled_prediction.score <= 0.10)].reset_index(
            drop=True)
        full_unlabeled_prediction['label'] = full_unlabeled_prediction['score'].apply(
            lambda x: 1 if x >= 0.99 else 0)

        trn_dataset = pd.concat(
            [full_annotated_dataset, full_unlabeled_prediction.sample(n=int(len(full_annotated_dataset)),
                                                                      random_state=args.seed)], ignore_index=True)

        training_set = TripleVerificationDataset(trn_dataset, head_positive, tokenizer,
                                                 args.max_length, split='trn', model=args.ptlm,
                                                 use_atomic=args.use_atomic, candidate_num=args.candidate_num)
        dev_dataset = TripleVerificationDataset(full_annotated_dataset, head_positive, tokenizer,
                                                args.max_length, split='dev', model=args.ptlm,
                                                use_atomic=args.use_atomic, candidate_num=args.candidate_num)
        tst_dataset = TripleVerificationDataset(full_annotated_dataset, head_positive, tokenizer,
                                                args.max_length, split='tst', model=args.ptlm,
                                                use_atomic=args.use_atomic, candidate_num=args.candidate_num)

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

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(training_loader) * args.epochs)
    criterion = CrossEntropyLoss()
    best_val_auc, best_val_acc, best_val_tuned_acc, best_tst_auc, best_tst_acc, best_tst_tuned_acc = 0, 0, 0, 0, 0, 0
    best_val_threshold, best_tst_threshold = 0, 0

    if args.automatic_find_semi_supervise:
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
            if tst_acc > best_tst_acc:
                best_tst_acc = tst_acc
                # torch.save(model.state_dict(), save_dir + f"/best_tst_acc_model_seed_{args.seed}.pth")
                # tokenizer.save_pretrained(save_dir + "/best_tst_acc_tokenizer")
            if tst_auc > best_tst_auc:
                best_tst_auc = tst_auc
                torch.save(model.state_dict(), save_dir + f"/best_tst_auc_model_seed_{args.seed}.pth")
                tokenizer.save_pretrained(save_dir + "/best_tst_auc_tokenizer")
            if tst_tuned_acc > best_tst_tuned_acc:
                best_tst_tuned_acc = tst_tuned_acc
                best_tst_threshold = tst_tuned_threshold
                # torch.save(model.state_dict(), save_dir + f"/best_tst_tunedAcc_model_seed_{args.seed}.pth")
                # tokenizer.save_pretrained(save_dir + "/best_tst_tunedAcc_tokenizer")

            logger.info(
                f"Validation {updated} reached best at before training, evaluating on test set")
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
        np.save(save_dir + "/best_record.npy", {'val_auc': best_val_auc, 'tst_auc': best_tst_auc})

    model.train()
    progress_bar = tqdm(range(len(training_loader) * args.epochs), desc="Training")

    iteration = 0
    for e in range(args.epochs):
        epoch_time = []

        for iteration, data in enumerate(training_loader, iteration + 1):
            # the iteration starts from 1.
            batch_start = time.time()
            y = data['label'].to(args.device, dtype=torch.long)

            ids = data['ids'].to(args.device, dtype=torch.long)
            mask = data['mask'].to(args.device, dtype=torch.long)
            tokens = {"input_ids": ids, "attention_mask": mask}

            logits = model(tokens)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)
            batch_end = time.time()
            epoch_time.append(batch_end - batch_start)

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
                np.save(save_dir + "/best_record.npy", {'val_auc': best_val_auc, 'tst_auc': best_tst_auc})
        print("Epoch {} time is".format(e), sum(epoch_time))
        logger.info("Epoch {} time is {}".format(e, sum(epoch_time)))


if __name__ == "__main__":
    main()
