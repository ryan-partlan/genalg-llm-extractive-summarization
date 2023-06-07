import os
import torch
import json
from fuzzywuzzy import fuzz
from utils import *
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, BertModel, OpenAIGPTModel

class ExtractiveSummaryDataset(Dataset):
    def __init__(self, json_file, num_examples, tok_fn, emb_fn):
        self.num_examples = num_examples
        self.tok_fn = tok_fn
        self.emb_fn = emb_fn
        self.data = self.extract(json_file)

    def extract(self, file):
        data = []
        with open(file) as f:
            num_ex = 0
            for i, ln in enumerate(f):
                obj = json.loads(ln)
                if obj["density_bin"] == "extractive":
                    num_ex += 1
                    # split_text = torch.tensor([self.BERTify(sent) for sent in obj["text"].split("\n") if sent])
                    # split_sum = torch.tensor([self.BERTify(sent.lstrip() + ".") for sent in obj["summary"].split(".") if sent])
                    # split_text = [sent for sent in obj["text"].split("\n") if sent]
                    split_text = sent_tokenize(obj["text"])

                    # split_summ = split_into_sentences(obj["summary"])
                    # split_summ = sent_tokenize(obj["summary"])
                    split_summ = [sent.lstrip() + "." for sent in obj["summary"].split(".") if sent]
                    summ_indices = torch.tensor([any([fuzz.ratio(item1, item2) > 80 for item2 in split_summ]) for item1 in split_text])
                    # print(summ_indices)
                    # I use fuzz here because of sentence tokenization errors.
                    # Without fuzz, won't detect the correct sentence.
                    # print(summ_indices)
                    # print("sims found", sum(summ_indices))
                    # print("sum length", len(split_summ))
                    tens_text = torch.vstack([self.BERTify(sent) for sent in split_text])
                    tens_summ = torch.vstack([self.BERTify(sent) for sent in split_summ])
                    title = self.BERTify(obj['title'])
                    # data.append({"title":title,"" tens_text, tens_summ, split_text, split_summ, summ_indices))
                    data.append({"title":title.squeeze(0),
                                 "tens_text":tens_text,
                                 "tens_summ":tens_summ,
                                 "split_text":split_text,
                                 "split_summ":split_summ,
                                 "summ_indices":summ_indices})

                if num_ex >= self.num_examples:
                    break
        return data

    def BERTify(self, sent):
        toked = self.tok_fn(
            sent,
            return_tensors="pt",
            padding="max_length",
            max_length=10,  # Parametrize please
            truncation=True,
        )
        embedded = self.emb_fn(**toked).last_hidden_state
        return embedded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dataloaders(train_file, test_file, dev_file, tok_fn, emb_fn):
    train_set = ExtractiveSummaryDataset(train_file, 10, tok_fn, emb_fn)
    dev_set = ExtractiveSummaryDataset(dev_file, 1, tok_fn, emb_fn)
    test_set = ExtractiveSummaryDataset(test_file, 1, tok_fn, emb_fn)
    return train_set, dev_set, test_set
    # print(len(train_set[0][0]))
    # print(len(dev_set[0][1]))
    # print(len(test_set[0][2]))

    # print(dev_set[0])
    # print(test_set[0])

    # train_loader = DataLoader(train_set, shuffle=True)
    # dev_loader = DataLoader(dev_set)
    # test_loader = DataLoader(test_set)
    #
    # return train_loader, dev_loader, test_loader

llm_catalog = {
    "distilbert-base-uncased": (
        DistilBertTokenizer.from_pretrained,
        DistilBertModel.from_pretrained,
    ),
    "bert-base-uncased": (AutoTokenizer.from_pretrained, BertModel.from_pretrained),
    "openai-gpt": (AutoTokenizer.from_pretrained, OpenAIGPTModel.from_pretrained)
    ## add more model options here if desired
}

def get_llm(model_name):
    tok_fn, model_fn = llm_catalog[model_name]
    tok_fn = tok_fn(model_name)
    model_fn = model_fn(model_name)
    if tok_fn.pad_token is None:
        tok_fn.add_special_tokens({'pad_token': '[PAD]'})
        model_fn.resize_token_embeddings(len(tok_fn))
    return tok_fn, model_fn


if __name__ == "__main__":
    tok_fn, emb_fn = get_llm("openai-gpt")
    # x = tok_fn("hello my name is", return_tensors="pt")
    # y = model_fn(**x).last_hidden_state.squeeze()
    # print(y)
    # train_loader, dev_loader, test_loader = get_dataloaders("release/train-stats.jsonl",
    #                                                         "release/test-stats.jsonl",
    #                                                         "release/dev-stats.jsonl",
    #                                                         tok_fn,
    #                                                         emb_fn)
    train_set, dev_set, test_set = get_dataloaders("release/train-stats.jsonl",
                                                            "release/test-stats.jsonl",
                                                            "release/dev-stats.jsonl",
                                                            tok_fn,
                                                            emb_fn)

    # title, text, summary
    # Each has size
        # 1, num_sentence, max_sentence_length_cutoff, embedding_dim
        # All titles are dimension (1, 1, 10, 768), since each title is one sentence.

    for j, batch in enumerate(train_set):
        # print([ten.size() for ten in batch])
        print(batch["title"].size())
        print(batch["tens_text"].size())
        print(batch["tens_summ"].size())
        print(batch["split_text"])
        print(batch["split_summ"])
        print(batch["summ_indices"])

    # data.append({"title": title,
    #              "tens_text": tens_text,
    #              "tens_summ": tens_summ,
    #              "split_text": split_text,
    #              "split_summ": split_summ,
    #              "summ_indices": summ_indices})
    #
    # for i, batch in enumerate(dev_loader):
    #     print([ten.size() for ten in batch])
    # for i, batch in enumerate(test_loader):
    #     print([ten.size() for ten in batch])
