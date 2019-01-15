import random

import numpy as np
import torch

from const import *


class DataLoader:
    def __init__(self, story,
                 question,
                 answer,
                 max_q_len,
                 max_s_len,
                 word2idx,
                 answer2idx,
                 cuda=True,
                 batch_size=64,
                 shuffle=True):

        self.sents_size = len(story)
        self.max_q_len = max_q_len
        self.max_s_len = max_s_len
        self.cuda = cuda
        self.bsz = batch_size
        self.step = 0

        self.word2idx = word2idx
        self.answer2idx = answer2idx

        self.story = np.asarray(story)
        self.question = np.asarray(question)
        self.answer = np.asarray(answer, dtype=np.int64)

        self.stop_step = self.sents_size // batch_size

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.arange(self.story.shape[0])
        np.random.shuffle(index)
        self.story = self.story[index]
        self.question = self.question[index]
        self.answer = self.answer[index]

    def __iter__(self):
        return self

    def __next__(self):
        def story2tensor(story):
            story = np.array([np.concatenate((np.array([sent + [PAD] * (self.max_s_len-len(sent))
                                                        for sent in s]), np.zeros((STORYLEN - len(s), self.max_s_len)))) for s in story], dtype=np.int64)
            story = torch.from_numpy(story)
            if self.cuda:
                story = story.cuda()
            return story

        def qustion2tensor(question):
            question = np.array([q + [PAD] * (self.max_q_len - len(q))
                                 for q in question], dtype=np.int64)
            question = torch.from_numpy(question)
            if self.cuda:
                question = question.cuda()
            return question

        def answer2tensor(answer):
            answer = torch.from_numpy(answer)
            if self.cuda:
                answer = answer.cuda()
            return answer

        if self.step == self.stop_step:
            self.step = 0
            raise StopIteration()

        start = self.step * self.bsz
        self.step += 1

        story = story2tensor(self.story[start:start + self.bsz])
        question = qustion2tensor(self.question[start:start + self.bsz])
        answer = answer2tensor(self.answer[start:start + self.bsz])

        return story, question, answer


if __name__ == "__main__":
    data = torch.load("data/corpus.pt")
    dl = DataLoader(data["story"],
                    data["question"],
                    data["answer"],
                    data["max_q_len"],
                    data["max_s_len"],
                    data["word2idx"],
                    data["answer2idx"], batch_size=2)

    s, q, a = next(dl)
    print(s)
    print(q)
    print(a.shape)

    dl = DataLoader(data["test_story"],
                    data["test_question"],
                    data["test_answer"],
                    data["max_q_len"],
                    data["max_s_len"],
                    data["word2idx"],
                    data["answer2idx"], batch_size=2)

    s, q, a = next(dl)
    print(s)
    print(q)
    print(a.shape)
    print(s[:, 1, :])
