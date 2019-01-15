import torch
import re

from const import *


def story2idx(stories, word2idx):
    return [[[word2idx[w] if w in word2idx else UNK for w in sent]
             for sent in story] for story in stories]


def question2idx(questions, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in question] for question in questions]


def answer2idx(answers, word2idx):
    return [word2idx[w] for w in answers]


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def parse_answer(answers):
    answer2idx = {}
    for answer in answers:
        if answer2idx.get(answer) is None:
            answer2idx[answer] = len(answer2idx)

    return answer2idx


class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
        }
        self.idx = len(self.word2idx)

    def add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def parse_s(self, stories):
        words = [word for story in stories for sent in story for word in sent]
        for word in words:
            self.add(word)

    def parse_q(self, questions):
        words = [word for question in questions for word in question]
        for word in words:
            self.add(word)

    def __len__(self):
        return self.idx


class Corpus(object):
    def __init__(self, save_data="data/corpus.pt", max_s_len=10, max_q_len=16):
        self.save_data = save_data
        self.word = Dictionary()
        self.max_s_len = max_s_len
        self.max_q_len = max_q_len

        self.parse_data()
        self.save()

    def parse_data(self):
        stories, questions, answers = [], [], []
        test_stories, test_questions, test_answers = [], [], []

        ignore_q = ignore_s = 0
        for i in range(1, 21):
            for line in open(f'data/qa{i}_train.txt'):
                index, line = line.strip().split(" ", 1)
                if int(index) == 1:
                    story = []

                contents = line.split("\t")
                if len(contents) == 3:
                    q, a, _ = contents
                    qs = normalizeString(q).split()
                    if len(qs) > self.max_q_len:
                        qs = qs[:self.max_q_len]
                        ignore_q += 1

                    stories.append(story.copy()[-20:])
                    questions.append(qs)
                    answers.append(a)
                else:
                    s = normalizeString(contents[0]).split()
                    if len(s) > self.max_s_len:
                        s = s[:self.max_s_len]
                        ignore_s += 1
                    story.append(s)

            for line in open(f'data/qa{i}_test.txt'):
                index, line = line.strip().split(" ", 1)
                if int(index) == 1:
                    story = []

                contents = line.split("\t")
                if len(contents) == 3:
                    q, a, _ = contents
                    qs = normalizeString(q).split()
                    if len(qs) > self.max_q_len:
                        qs = qs[:self.max_q_len]
                        ignore_q += 1

                    test_stories.append(story.copy()[-20:])
                    test_questions.append(qs)
                    test_answers.append(a)
                else:
                    s = normalizeString(contents[0]).split()
                    if len(s) > self.max_s_len:
                        s = s[:self.max_s_len]
                        ignore_s += 1
                    story.append(s)

        self.word.parse_q(questions)
        self.word.parse_s(stories)

        self.answer2idx = parse_answer(answers)

        self.stories = stories
        self.questions = questions
        self.answers = answers

        self.test_stories = test_stories
        self.test_questions = test_questions
        self.test_answers = test_answers

        print(f"ignore story lenght - {ignore_s}")
        print(f"ignore question lenght - {ignore_q}")
        print(f"answer length - {len(self.answer2idx)}")
        print(f"word length - {len(self.word)}")

    def save(self):
        data = {
            'max_q_len': self.max_q_len,
            'max_s_len': self.max_s_len,
            'word2idx': self.word.word2idx,
            'answer2idx': self.answer2idx,
            'story': story2idx(self.stories, self.word.word2idx),
            'question': question2idx(self.questions, self.word.word2idx),
            'answer': answer2idx(self.answers, self.answer2idx),
            'test_story': story2idx(self.test_stories, self.word.word2idx),
            'test_question': question2idx(self.test_questions, self.word.word2idx),
            'test_answer': answer2idx(self.test_answers, self.answer2idx),
        }

        torch.save(data, self.save_data)
        print(f'Finish dumping the data to file - {self.save_data}')


if __name__ == "__main__":
    Corpus()
