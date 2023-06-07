import numpy as np
import re
import torch
import torch.nn as nn
# This file is just a bunch of similarity metrics to throw into a vector.
# Similarity should be sim(sentence, other_thing)
# Example: similarity between sentence and title
# Each function should take a datum (dataset entry) and output a score.


cos = nn.CosineSimilarity(dim=1)

def get_estimates_from_example(population, example):
    """
    Population is a list of "ranking" vectors.
    Goal is to take n x f tensor population, hit the entire corpus
    :param population: dim (n x f)
        n is the number of candidates
        f is the number of functions per candidate
    :param function_list:
    :return: estimates:
    """

def av_sentence_vec(paragraph):
    """
    Takes list of sentences of contextualized embeddings, outputs average of those vectors,
    this gets a contextualized sentence vector
    :param embeddings: the chunk of text for processing, size (len, 768)
    :param sentence_idx: which sentence to look at
    :return:
    """
    # embeddings: (1, num sentences, num words per sentence, size embedding)
    return torch.mean(paragraph, axis=0)  # maybe .unsqueeze(0)

def sentence_vec_from_paragraph(paragraph, sentence_idx):
    vec = pull_sentence(paragraph, sentence_idx)
    return sentence_vec(vec)

def sentence_vec(sentence):
    return torch.mean(sentence, axis=0).unsqueeze(0)

def pull_sentence(embedded_paragraph, sentence_idx):
    return embedded_paragraph[sentence_idx]

def convert_par_to_sent(paragraph):
    return torch.mean(paragraph, axis=1).squeeze()

def sim_to_title(datum):
    paragraph = datum["tens_text"]
    title = sentence_vec(datum["title"])
    paragraph = convert_par_to_sent(paragraph)
    return cos(paragraph, title)

def num_letters(datum):
    """
    :param datum:
    :return: number of letters in each sentence (a list)
    """
    text = datum["split_text"]
    num_let = [len(sent.replace(" ", "")) for sent in text]
    return torch.tensor(num_let)

def num_words(datum):
    """
    :param datum:
    :return: number of words in each sentence (a list)
    """
    text = datum["split_text"]
    num_w = [len(sent.split(" ")) for sent in text]
    return torch.tensor(num_w)

def homogeneity(datum):
    """
    :param datum:
    :return: average cosine similarity of one sentence to the rest
    """
    paragraph = datum["tens_text"]
    sents = convert_par_to_sent(paragraph)
    cos_s = nn.CosineSimilarity(dim=1)
    homo_ranking = torch.vstack([cos_s(sents[i], sents) for i in range(sents.size(0))])
    avg_ranking = torch.mean(homo_ranking, dim=1)
    return avg_ranking

# def sim_to_title(datum, sentence_idx):
#     """
#     Takes a title, outputs a vector of the cos similarity to each
#     title: (k x 1)
#     embeddings: (n x k)
#         n is the number of candidates in the genepool
#         k is the length of each candidate (padded into a tensor)
#     output: (n x 1)
#     :param embeddings:
#     :return: vector of cos similarities between title and candidate slns
#     """
#     paragraph = datum["tens_text"]
#     title = datum["title"]
#     sent = sentence_vec_from_paragraph(paragraph, sentence_idx)
#     tit = sentence_vec(title)
#     sim = cos(sent, tit)
#     return sim