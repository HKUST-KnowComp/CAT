import json
import os
import sys
from random import sample

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm import trange

from utils import relation2text_dict

sys.path.append(os.getcwd())


def replace_bias(x):
    return str(x).replace('person x', 'PersonX').replace('person y', 'PersonY').replace('Person X', 'PersonX').replace(
        'Person Y', 'PersonY').replace('person X', 'PersonX').replace('person Y', 'PersonY')


class TripleVerificationDataset(Dataset):
    def __init__(self, dataframe, head_concept_dataframe, tokenizer, max_length=30, model="roberta-base", split='trn',
                 predict=False, use_atomic=False, candidate_num=9):
        self.tokenizer = tokenizer
        self.model = model
        self.data = dataframe.reset_index(drop=True)
        self.head_data = head_concept_dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.split = split
        self.sep_token = " "
        self.predict = predict
        self.use_atomic = use_atomic
        self.candidate_num = candidate_num

        assert self.split in list(self.data['split'].unique()), "{} not found in {}".format(self.split,
                                                                                            self.data['split'].unique())
        if 'gpt2' in model and 'split' == 'trn':
            self.data = self.data[self.data.label == 1]
        self.data = self.data[self.data.split == self.split].reset_index(drop=True)

        self.data['head'] = self.data['head'].apply(replace_bias)
        self.data['tail'] = self.data['tail'].apply(replace_bias)

        self.head = self.data['head'].tolist()
        self.relation = self.data['relation'].tolist()
        self.tail = self.data['tail'].tolist()
        self.label = self.data['label'].tolist()
        if 'info' in list(self.data):
            self.info = self.data['info'].tolist()

        self.head_instance_dict = {}
        for i in tqdm(self.data['head'].unique(), desc="Processing instances"):
            concept = i.split('[')[-1].split(']')[0]
            concept_df = self.head_data[self.head_data.concept_text == concept].sort_values(by=['score'],
                                                                                            ascending=True).drop_duplicates(
                subset=['head']).reset_index(drop=True)
            if len(concept_df) == 0:
                self.head_instance_dict[concept] = [concept]
            else:
                concept_df['instance'] = concept_df['head'].apply(lambda x: x.split('[')[-1].split(']')[0])
                self.head_instance_dict[concept] = concept_df['instance'].tolist()

        if self.use_atomic and split == 'trn':
            atomic = pd.read_csv('../../data/atomic/v4_atomic_all_agg.csv', index_col=None)
            involved_head_index = list(set([json.loads(i)['d_i'] for i in self.info]))
            involved_atomic = atomic.loc[involved_head_index].reset_index(drop=True)
            for j in trange(len(involved_atomic), desc="Adding ATOMIC"):
                for r in ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']:
                    tails = json.loads(involved_atomic.loc[j, r])
                    for t in tails:
                        if t == "none":
                            continue
                        else:
                            self.head.append(replace_bias(involved_atomic.loc[j, 'event_conceptualization']))
                            self.relation.append(r)
                            self.tail.append(replace_bias(t))
                            self.label.append(1)
                            self.head_instance_dict[replace_bias(involved_atomic.loc[j, 'event_conceptualization'])] = []

    def __len__(self):
        return len(self.head)

    def __getitem__(self, index):
        # Preprocessing
        event_sent = self.head[index]  # PersonX drinks [CocaCola]
        event_sent_token_bounded = event_sent.replace('[', ' <c> ').replace(']',
                                                                            ' </c> ')  # PersonX drinks <c> CocaCola </c>
        instance_candidates = list(set(self.head_instance_dict[event_sent.split('[')[-1].split(']')[0]]))
        if len(instance_candidates) > self.candidate_num:
            instance_candidates = sample(instance_candidates, self.candidate_num)
        candidate_str = ", ".join(instance_candidates)
        relation_text = relation2text_dict[self.relation[index]]
        tail = self.tail[index]

        if 'bert-' in self.model or 'deberta' in self.model or 'electra' in self.model:
            prompted_sent = "{}, {}, {} [SEP] {}".format(event_sent_token_bounded, relation_text, tail, candidate_str)
        elif 'gpt2' in self.model:
            prompted_sent = "{}, {}, {}. {} [EOS]".format(event_sent_token_bounded, relation_text, tail, candidate_str)
        elif 'roberta' in self.model:
            prompted_sent = "{}, {}, {} </s> {}".format(event_sent_token_bounded, relation_text, tail, candidate_str)
        elif 'bart' in self.model:
            prompted_sent = "{}, {}, {}. {}".format(event_sent_token_bounded, relation_text, tail, candidate_str)
        else:
            raise NotImplementedError("Model type {} not implemented in dataloader".format(self.model))
        source = self.tokenizer.batch_encode_plus([prompted_sent],
                                                  padding='max_length', max_length=self.max_length,
                                                  return_tensors='pt', truncation=True)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        # print(prompted_sent)
        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long),
            'label': torch.tensor(self.label[index]).to(dtype=torch.long),
        }

    def merge_predicted_score(self, prediction):
        if not self.predict:
            print("Not in prediction model.")
        else:
            self.data['score'] = list(prediction)
            return self.data

    def merge_predicted_label(self, prediction_label):
        if not self.predict:
            print("Not in prediction model.")
        else:
            self.data['label'] = list(prediction_label)
            return self.data
