from operator import itemgetter
import argparse
from random import choices
import pickle
from concurrent.futures import Future, ProcessPoolExecutor
from time import perf_counter
import json

def _calculate_next_token_probabilities(curr_gram: tuple, token_counts: dict[tuple, int], vocab, text_len, data, next_model = {}, next_counts = {}) -> tuple[dict[tuple, dict[str, float]], dict[tuple, int]]:
    print("go token")
    # construct the string to search for
    curr_string = ""
    for token in curr_gram:
        curr_string += token + " "

    # calculate all the probabilities with this gram to predict the next word
    for token in vocab:
        index = 0
        count = 0

        # count all the occurrences of the token to calculate probability
        while True:
            index = data.find(curr_string + token, index)
            if index == -1: 
                break
            count += 1
            index += 1

        if count == 0 :
            continue

        if curr_gram not in next_model:
            next_model[curr_gram] = {}
        dict_at_gram = next_model[curr_gram]

        next_counts[curr_gram + (token,)] = count
        # calculating the unigram? p = count of token / total amount of tokens in text
        # n-gram n > 1? p(a|b) = p(ab) / p(b). 
        # In other words, count of all occurrences of curr_gram & token in order / all occurrences of curr_gram
        if len(curr_gram) == 0:
            dict_at_gram[token] = count / text_len
        else:
            dict_at_gram[token] = count / token_counts[curr_gram]

    return next_model, next_counts

def bfs(curr_gram: tuple, token_counts: dict[tuple, int], ngram_order, vocab, text_len, data, next_model = {}, next_counts = {}):
    if len(curr_gram) > ngram_order or curr_gram not in token_counts or token_counts[curr_gram] == 0:
        return ({}, {}) 

    if len(curr_gram) > 1:
        print(curr_gram)

    _calculate_next_token_probabilities(curr_gram, token_counts, vocab, text_len, data, next_model, next_counts)
    # recursively call bfs, appending all vocab words to this gram
    for token in vocab:
        bfs(curr_gram + (token,), token_counts, ngram_order, vocab, text_len, data, next_model, next_counts)

    return next_model, next_counts

"""
A Machine Learning model generating words based on provided training text.
The model looks at the previous N tokens and predicts which word will come next based
on the probability that word appears given the previous N tokens.
"""
class NgramModel:
    model: dict[tuple, dict[str, float]]
    ngram_order: int

    """
    Positional arguments:
    - ngram_order: how many tokens the model can guess on at a time. Increasing this will increase 
                   compute requirements and/or time

    - model: the model to restore from if predicting.
    """
    def __init__(self, ngram_order: int, model: dict[tuple, dict[str, float]] = {}):
        self.ngram_order = ngram_order
        self.model = model

    """
    Trains the model on the given data string. Saves the model for subsequent use,
    as well as returning it. Pass the result into the NgramModel constructor to
    restore the trained model.
    """
    def train(self, data: str) -> dict[tuple, dict[str, float]]:
        print("Traning ngram model...")
        start = perf_counter()

        # tokenize data - split by whitespace and punctuation
        print("tokenizing...")
        splits: list[str] = [] 
        tokens = []
        last = 0
        for i in range(len(data)):
            if not data[i].isalnum():
                splits.append(data[last:i])
                tokens.append(data[i])
                last = i+1
        for split in splits:
            tokens.extend(split.split())
        text_len = len(tokens)
        vocab = set(tokens)

        print("training...")
        self.model, tokens_count = _calculate_next_token_probabilities((), {}, vocab, text_len, data)

        with ProcessPoolExecutor() as executor:
            print("Starting worker processes...")
            futures: list[Future] = []
            for word in vocab:
                future = executor.submit(bfs, (word,), tokens_count, self.ngram_order, vocab, text_len, data)
                futures.append(future)

            for i, future in enumerate(futures):
                new_model, _ = future.result()
                self.model |= new_model
                print("process finished " + str(i))
        elapsed_time = perf_counter() - start

        print("Training completed successfully!")
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
        return self.model

    """
    Predicts the next word based on the loaded model.

    Positional arguments: 
    - words: A tuple containing the words to predict off of. Tuple len cannot exceed ngram order - 1.
    - deterministic: A bool determining which method of selection to use. True will pick the maximum probability,
                     False will choose a word randomly based on the probabilies the model was trained on.
    Returns: A string of the word the model chose.
    """
    def predict_next_word(self, words: tuple, deterministic: bool) -> str:
        while words not in self.model:
            if len(words) == 0:
                raise ValueError("Word(s) not in the model's trained dictionary.")
            words = words[1:]

        curr_predicts = self.model[words]
        values = sorted(curr_predicts.items(), key=itemgetter(1))

        if deterministic:
            # index into the model at len of words, choose the max value in the dict
            return values[0][0]

        [token] = choices(list(curr_predicts.keys()), list(curr_predicts.values()))
        return token


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_or_predict', type=str, help="enter predict_ngram to predict the next word, train_ngram to train the ngram model based on data input.")
    parser.add_argument('--data', '-d', type=str, help="The path (including extension) to the data file used for training.")
    parser.add_argument('--save', '-s', type=str, help="The path (including extension) where the trained model will be stored.")
    parser.add_argument('--load', '-l', type=str, help="The path (including extension) to load an ngram model from.")
    parser.add_argument('--word', '-w', type=str, help="Predicts the next word that comes after the word provided")
    parser.add_argument('--nwords', type=int, help="The number of words to predict and output")
    parser.add_argument('--d', action='store_true', help="whether the output is deterministic or not")
    parser.add_argument('-n', type=int, help="The order of the ngram to train or predict")
    args = vars(parser.parse_args())

    if 'n' not in args:
        print("ngram order argument missing")
        exit(1)

    if args['train_or_predict'] == "train_ngram":
        if 'save' not in args:
            print("save path argument missing")
            exit(1)

        if 'data' not in args:
            print("data path argument missing")
            exit(1)

        ngram = NgramModel(args['n'])
        data = "testing my new testing model!"
        with open(args['data']) as f:
            data = f.read()

        model = ngram.train(data)
        with open(args['save'], "wb+") as f:
            pickle.dump(model, f)

        with open('./model.json', "w+") as f:
            f.write(str(model))

        exit(0)

    if 'load' not in args:
        print("model load path argument missing")
        exit(1)

    if 'nwords' not in args:
        print("max word output argument missing")
        exit(1)
    if 'word' not in args:
        print("word argument missing")
        exit(1)

    ngram = None

    with open(args['load'], 'rb') as f:
        model = pickle.load(f)
        ngram = NgramModel(args['n'], model)
    output = [args['word']] 

    while len(output) < args['nwords']:
        predict_tuple = None
        if len(output) < (args['n'] - 1):
            predict_tuple = tuple(output)
        else:
            # get order-1 tokens from the end of output
            predict_tuple = tuple(output[-(args['n']-1):])
        token = ngram.predict_next_word(predict_tuple, args['d']) 
        output.append(token)

    print(" ".join(output))