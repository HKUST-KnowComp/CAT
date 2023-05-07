import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

cr_loss = torch.nn.functional.cross_entropy


def evaluate(tokenizer, model, device, loader, logger, test_threshold=0.5, finetune_threshold=False,
             model_type="roberta"):
    predicted_scores = torch.tensor([]).to(device)
    predicted_class = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)

    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, desc="Evaluating"), 0):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids": ids, "attention_mask": mask}

            if "roberta" in model_type or 'bert-' in model_type or 'deberta' in model_type or 'bart' in model_type or 'electra' in model_type:
                outputs_logits = model(tokens)
                logits = torch.softmax(outputs_logits, dim=1)
                pred_prob = logits[:, 1]
                class_pred = torch.argmax(logits, dim=-1)

            elif 'gpt2' in model_type:
                outputs = model(input_ids=ids, attention_mask=mask, labels=ids)

                shift_logits = outputs[1][..., :-1, :].contiguous().view(-1, outputs[1].size(-1))
                shift_labels = ids[..., 1:].contiguous().view(-1)

                losses = cr_loss(shift_logits, shift_labels,
                                 ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)

                losses = torch.div(torch.sum(losses, dim=1), torch.sum(mask[:, 1:], dim=1))
                # (batch_size, ) get the loss after removing PAD_TOKEN

                pred_prob = -losses
                class_pred = pred_prob.detach().clone()
                class_pred[class_pred >= 0.5] = 1
                class_pred[class_pred < 0.5] = 0
            predicted_class = torch.cat((predicted_class, class_pred))
            predicted_scores = torch.cat((predicted_scores, pred_prob))
            labels = torch.cat((labels, y))

    threshold = test_threshold
    predicted_scores_copy = predicted_scores.detach().clone()
    predicted_scores_copy[predicted_scores_copy >= threshold] = 1
    predicted_scores_copy[predicted_scores_copy < threshold] = 0
    max_acc = accuracy_score(labels.tolist(), (predicted_scores_copy).tolist())

    if finetune_threshold:
        for s in predicted_scores.tolist():
            score_copy = predicted_scores.detach().clone()
            score_copy[score_copy >= s] = 1
            score_copy[score_copy < s] = 0
            acc = accuracy_score(labels.tolist(), score_copy.tolist())
            if acc > max_acc:
                max_acc = acc
                threshold = float(s)
                # logger.info("Updated threshold to {}".format(threshold))
    return roc_auc_score(labels.tolist(), (predicted_scores).tolist()), \
           accuracy_score(labels.tolist(), (predicted_class).tolist()), max_acc, threshold, len(labels)


def score_triples(tokenizer, model, device, loader, model_type="kgbert"):
    """
        return: predicted_scores (list) The scores predicted by the model.
                for KG-BERT, the returned score is the softmax score for the triple_conceptualization being true.
                    GPT2, the returned score is the negative GPT2 loss.
    """
    model.eval()

    predicted_scores = torch.tensor([]).to(device)

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids": ids, "attention_mask": mask}

            if "roberta" in model_type:
                outputs_logits = model(tokens)

                logits = torch.softmax(outputs_logits, dim=1)
                values = logits[:, 1]
            elif model_type == "gpt2":
                outputs = model(input_ids=ids, attention_mask=mask, labels=ids)

                shift_logits = outputs[1][..., :-1, :].contiguous().view(-1, outputs[1].size(-1))
                shift_labels = ids[..., 1:].contiguous().view(-1)

                losses = cr_loss(shift_logits, shift_labels,
                                 ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)

                losses = torch.div(torch.sum(losses, dim=1),
                                   torch.sum(mask[:, 1:],
                                             dim=1))  # (batch_size, ) get the loss after removing PAD_TOKEN

                values = -losses

            predicted_scores = torch.cat((predicted_scores, values))

    return predicted_scores.tolist()
