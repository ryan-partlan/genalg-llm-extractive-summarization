from ExtractiveSummaryDataset import *
from utils import *
from FunctifiedDataset import *
from Population import *
from torch.utils.data import Dataset, DataLoader

def get_llm(model_name):
    tok_fn, model_fn = llm_catalog[model_name]
    tok_fn = tok_fn(model_name)
    model_fn = model_fn(model_name)
    if tok_fn.pad_token is None:
        tok_fn.add_special_tokens({'pad_token': '[PAD]'})
        model_fn.resize_token_embeddings(len(tok_fn))
    return tok_fn, model_fn

if __name__=="__main__":
    # data.append({"title": title,
    #              "tens_text": tens_text,
    #              "tens_summ": tens_summ,
    #              "split_text": split_text,
    #              "split_summ": split_summ,
    #              "summ_indices": summ_indices})
    tok_fn, emb_fn = get_llm("openai-gpt")
    funcs = [homogeneity, num_letters, num_words, sim_to_title]
    train_set = ExtractiveSummaryDataset("release/train-stats.jsonl", 10, tok_fn, emb_fn)
    gp = GenePool(100, train_set, *funcs)
    # dev_set = ExtractiveSummaryDataset("release/train-stats.jsonl",10)
    # train_set = ExtractiveSummaryDataset("release/train-stats.jsonl")
    # test_ex = train_set[0]
    # tit = test_ex["title"]
    # print(tit.size())
    # av = av_sentence_vec(tit, 0)
    # print(av)
    # print(av.size())
    # print(test_ex)
    # print(test_ex["tens_text"].size())
    # sentence_num = 2
    # print(test_ex["split_text"])

    # print(num_letters(test_ex))
    # print(num_words(test_ex))
    # print(sim_to_title(test_ex))
    # print(homogeneity(test_ex))
    #print(homogeneity(test_ex))

    # test_ex = pull_sentence(test_ex["tens_text"], sentence_num)
    # print(sim_to_title(test_ex, 1))
    # print(sim_to_title(test_ex))

    # print(test_ex.size())


