import torch
import torch.nn.functional as F
import random
import nltk
from rouge_metric import PyRouge

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

def get_avg_rouge(generated_summary, summary):
    scores = rouge.evaluate(generated_summary, summary)
    total = 0
    num_rouges = 0
    for key in scores.keys():
        total += sum(scores[key].values()) / len(scores[key].values())
        num_rouges += 1
    return total / num_rouges

class GenePool:
    def __init__(self, pop_size, dataset, *args):
        """
        :param pop_size: number of rankings
        :param dataset: dataset
        :param args: functions
        """
        self.pop_size = pop_size
        self.poputensor = torch.rand(pop_size, len(args))
        self.funcy_dataset = [torch.vstack([arg(datum) for arg in args]) for datum in dataset]
        self.dataset = dataset
        res = self.get_rankings()
        summaries = self.generate_summaries(res)
        print(self.fitness(summaries, dataset))
        # print(res)
        # print(self.generate_summaries(res))


    def get_rankings(self, top_k=3):
        res = [torch.matmul(self.poputensor, funcy_datum) for funcy_datum in self.funcy_dataset]
        rankings = [torch.topk(a, top_k, dim=1).indices.tolist() for a in res]
        return rankings

    def generate_summaries(self, rankings):
        """
        :param rankings:
        :return: list of list of summaries.
            Each list of summaries is ordered wrt the ordering of the population
        """
        summaries = []
        for i in range(len(self.dataset)):
            datum = self.dataset[i]
            text = datum["split_text"]
            ranking = rankings[i]
            summaries_datum = [" ".join([text[r] for r in sorted(rank)]) for rank in ranking]
            summaries.append(summaries_datum)
        return summaries

# incomplete due to PyRouge complications
#########################################################################################################
    def fitness(self, summaries, dataset):
        """
        takes a list of summaries, produces averaged rouge score for each.
        :return: fitness vector 1 x num_pop
        """
        fitmat = torch.zeros(len(dataset), self.pop_size)
        for i in range(len(dataset)):
            correct_summary = " ".join(dataset[i]["split_summ"])
            for j in range(self.pop_size):
                # print(summaries[i][j])
                # print(correct_summary)
                fitmat[i][j] += nltk.translate.bleu_score.sentence_bleu(summaries[i][j], correct_summary)
                # This is all incomplete
        return fitmat
#########################################################################################################