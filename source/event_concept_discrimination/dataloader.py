import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())


class ConceptLinkingAnnotatedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=30, model="roberta-base", split='trn', prompt_num=2,
                 predict=False):
        self.tokenizer = tokenizer
        self.model = model
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.split = split
        self.sep_token = " "
        self.predict = predict
        assert self.split in list(self.data['split'].unique()), "{} not found in {}".format(self.split,
                                                                                            self.data['split'].unique())
        if 'gpt2' in model and 'split' == 'trn':
            self.data = self.data[self.data.score >= 3]
        self.data = self.data[self.data.split == self.split].reset_index(drop=True)

        self.head = self.data['head']
        self.concept = self.data['concept_text']
        self.score = self.data['score']

        self.prompt_num = prompt_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Preprocessing
        event_sent = self.head[index]  # PersonX drinks [CocaCola]
        event_sent_token_bounded = event_sent.replace('[', '<c> ').replace(']',
                                                                           ' </c>')  # PersonX drinks <c> CocaCola </c>
        instance_to_be_conceptualized = self.head[index].split('[')[-1].split(']')[0]  # CocaCola
        token_bounded_instance = "<c> {} </c>".format(instance_to_be_conceptualized)  # <c> CocaCola </c>
        concept_text = self.concept[index]  # beverage
        token_bounded_concept = "<c> {} </c>".format(concept_text)  # <c> beverage </c>

        if 'bert-' in self.model or 'deberta' in self.model or 'electra' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} [SEP] {}".format(event_sent_token_bounded, token_bounded_concept)
            elif self.prompt_num == 2:
                prompted_sent = "{} [SEP] {} is an instance of {}".format(event_sent_token_bounded,
                                                                          token_bounded_instance, token_bounded_concept)
            else:
                raise Exception("Unimplemented prompts")
        elif 'gpt2' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} [SEP] {} [EOS]".format(event_sent_token_bounded, token_bounded_concept)
            elif self.prompt_num == 2:
                prompted_sent = "{} [SEP] {} is an instance of {} [EOS]".format(event_sent_token_bounded,
                                                                                token_bounded_instance,
                                                                                token_bounded_concept)
            else:
                raise Exception("Unimplemented prompts")
        elif 'roberta' in self.model or 'bart' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} </s> {}".format(event_sent_token_bounded, token_bounded_concept)
            elif self.prompt_num == 2:
                prompted_sent = "{} </s> {} is an instance of {}".format(event_sent_token_bounded,
                                                                         token_bounded_instance, token_bounded_concept)
            else:
                raise Exception("Unimplemented prompts")

        source = self.tokenizer.batch_encode_plus([prompted_sent],
                                                  padding='max_length', max_length=self.max_length,
                                                  return_tensors='pt', truncation=True)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long),
            'label': torch.tensor(self.score[index] >= 3).to(dtype=torch.long),
        }


class CAT_ConceptLinkingDataset(Dataset):
    def __init__(self, dataframe, pseudo_dataframe, tokenizer, candidate_number=3, max_length=30,
                 model="roberta-base", split='trn', prompt_num=1, predict=False, drop_same=True,
                 mode='semi-supervised'):
        self.tokenizer = tokenizer
        self.model = model
        self.data = dataframe
        self.pseudo_data = pseudo_dataframe
        self.mode = mode
        self.max_length = max_length
        self.split = split
        self.sep_token = " "
        self.predict = predict
        self.prompt_num = prompt_num
        self.candidate_number = candidate_number
        self.drop_same = drop_same

        assert self.split in list(self.data['split'].unique()), "{} not found in {}".format(self.split,
                                                                                            self.data['split'].unique())
        if 'gpt2' in model and 'split' == 'trn':
            self.data = self.data[self.data.score >= 3]
        self.data = self.data[self.data.split == self.split].reset_index(drop=True)
        self.data['score'] = self.data['score'].apply(lambda x: x / 5)

        if self.mode == 'semi-supervised' and self.split == 'trn':
            pos_pseudo_data = self.pseudo_data[
                (self.pseudo_data.score > 0.9) | (self.pseudo_data.score < 0.1)].reset_index(drop=True)
            self.data = pd.concat([pos_pseudo_data, self.data], ignore_index=True)

        unique_head_event = list(self.data['head'].unique())
        self.unlabeled_head2concept_dict = {h: None for h in unique_head_event}
        for h in tqdm(unique_head_event, desc="Building head unlabeled data"):
            head_labeled = self.data[self.data['head'] == h].reset_index(drop=True).sort_values(
                by='score', ascending=False)
            head_labeled = head_labeled[head_labeled.score >= 0.5].reset_index(drop=True)
            if self.mode == 'semi-supervised':
                head_unlabeled = self.pseudo_data[self.pseudo_data['head'] == h].reset_index(drop=True).sort_values(
                    by='score', ascending=False)
                # total = pd.concat([head_labeled, head_unlabeled]).reset_index(drop=True).sort_values(by='score',
                #                                                                                      ascending=False) \
                #     .drop_duplicates(subset=['head', 'concept_text'], keep='first')
                # total = total[total.score >= 0.5]
                head_unlabeled = head_unlabeled[head_unlabeled.score >= 0.5]
                self.unlabeled_head2concept_dict[h] = head_unlabeled['concept_text'].tolist()
            else:
                head_unlabeled = self.pseudo_data[self.pseudo_data['head'] == h].reset_index(drop=True).sort_values(
                    by='score', ascending=False)
                head_unlabeled_pos = head_unlabeled[head_unlabeled.score > 0.5]
                self.unlabeled_head2concept_dict[h] = head_unlabeled_pos['concept_text'].tolist()

        self.head = self.data['head']
        self.concept = self.data['concept_text']
        self.score = self.data['score']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Preprocessing
        event_sent = self.head[index]  # PersonX drinks [CocaCola]
        event_sent_remove_bracket = event_sent.replace('[', '"').replace(']', '"')  # PersonX drinks CocaCola
        event_sent_bounded = event_sent.replace('[', '<c> ').replace(']', ' </c>')  # PersonX drinks <c> CocaCola </c>
        instance_to_be_conceptualized = self.head[index].split('[')[-1].split(']')[0]  # CocaCola
        token_bounded_instance = "<c> {} </c>".format(instance_to_be_conceptualized)  # <c> CocaCola </c>
        concept_text = self.concept[index]  # beverage
        token_bounded_concept = "<c> {} </c>".format(concept_text)  # <c> beverage </c>
        candidate_filter_original = [i for i in self.unlabeled_head2concept_dict[event_sent] if i != concept_text]
        candidate = ', '.join(candidate_filter_original[
                              :self.candidate_number]) if candidate_filter_original else ""  # liquid, fluid, drink

        if 'bert-' in self.model or 'deberta' in self.model or 'electra' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} [SEP] {} [SEP] {}".format(event_sent_bounded, concept_text, candidate)
            elif self.prompt_num == 2:
                prompted_sent = "{} [SEP] {} is an instance of {} [SEP] {} is also instance of {}".format(
                    event_sent_remove_bracket,
                    token_bounded_instance, token_bounded_concept, token_bounded_instance, candidate)
            else:
                raise Exception("Unimplemented prompts")
        elif 'gpt2' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} [SEP] {} [SEP] {} [EOS]".format(event_sent_bounded, concept_text, candidate)
            elif self.prompt_num == 2:
                prompted_sent = "{} [SEP] {} is an instance of {} [SEP] {} is also instance of {} [EOS]".format(
                    event_sent_remove_bracket,
                    token_bounded_instance, token_bounded_concept, token_bounded_instance, candidate)
            else:
                raise Exception("Unimplemented prompts")
        elif 'roberta' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{} </s> {} </s> {}".format(event_sent_bounded, concept_text, candidate)
            elif self.prompt_num == 2:
                prompted_sent = "{} </s> {} is an instance of {} </s> {} is also instance of {}".format(
                    event_sent_remove_bracket,
                    token_bounded_instance, token_bounded_concept, token_bounded_instance, candidate)
            else:
                raise Exception("Unimplemented prompts")
        elif 'bart' in self.model:
            if self.prompt_num == 1:
                prompted_sent = "{}. {}. {}".format(event_sent_bounded, concept_text, candidate)
            elif self.prompt_num == 2:
                prompted_sent = "{}. {} is an instance of {}. {} is also instance of {} </s>".format(
                    event_sent_remove_bracket,
                    token_bounded_instance, token_bounded_concept, token_bounded_instance, candidate)
            else:
                raise Exception("Unimplemented prompts")
        else:
            raise NotImplementedError("model {} not found in embedding layer".format(self.model))

        # print(prompted_sent)
        source = self.tokenizer.batch_encode_plus([prompted_sent],
                                                  padding='max_length', max_length=self.max_length,
                                                  return_tensors='pt', truncation=True)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long),
            'label': torch.tensor(self.score[index] >= 0.5).to(dtype=torch.long),
        }

    def merge_predicted_score(self, prediction):
        if not self.predict:
            print("Not in prediction model.")
        else:
            self.data['predict_score'] = list(prediction)
            return self.data

    def merge_predicted_label(self, prediction_label):
        if not self.predict:
            print("Not in prediction model.")
        else:
            self.data['predict_label'] = list(prediction_label)
            return self.data
