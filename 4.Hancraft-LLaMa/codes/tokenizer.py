import torch.nn as nn
import torch
from collections import defaultdict

class BPE_Tokenizer:

    def __init__(self, pre_tokenizer) -> None:
        self.pre_tokenizer = pre_tokenizer
        self.vocab = None
        self.merge_rule = None
        pass

    def compute_word_freq(self, corpus):
        """ 
        split sentence into words and compute the occurance freq of each word
        return dict(word, freq)
        """
        word2freq = defaultdict(int)

        for text in corpus:
            words_with_offsets = self.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word2freq[word] += 1
        
        return word2freq
    
    def split_words(self, words):
        """ 
        split word into its splits
        return dict(word, splits)
        """
        word2split = {}
        for word in words:
            tmp = []
            for i in range(len(word)):
                """
                because GPT preprocess distinguish the first letter and other letter of a word by the 'Ġ', 
                which is the blank symbol, so we don't have to add ##
                alter 'merge_pair' function BTW
                """
                # if i==0:
                #     tmp.append(word[i])
                # else:
                #     tmp.append(f'##{word[i]}')
                tmp.append(word[i])
            word2split[word] = tmp
        return word2split
    
    def get_alphabet_from_words(self, words):
        """ 
        get initial alphabet from directly sperating words
        return list[str]
        """
        alphabet = set()
        for word in words:
            alphabet = alphabet.union([c for c in word])

        alphabet = sorted(alphabet) 
        return alphabet
    
    def get_alphabet_from_splits(self, splits):
        """ 
        get initial alphabet from using words' splits
        return list[str]
        """
        alphabet = set()
        for split in splits:
            alphabet = alphabet.union(split)

        alphabet = sorted(alphabet) 
        return alphabet
    

    def compute_pair_freq(self, word2split, word2freq):
        """ 
        compute the freq of each pair in these words
        return dict[pair, int]
        """
        pair2freq = defaultdict(int)
        for word, split in word2split.items():
            if len(word) == 1:
                pass
            else:
                freq = word2freq[word]
                for i in range(len(split)-1):
                    pair = (split[i], split[i+1])
                    pair2freq[pair] += freq
        return pair2freq
    

    def find_most_frequent_pair(self, pair2freq):
        """ 
        find the pair with the biggest frequence
        return pair, int
        """
        assert len(pair2freq) >= 1
        max_freq = -1
        max_freq_pair = None
        for pair, freq in pair2freq.items():
            if freq > max_freq:
                max_freq = freq
                max_freq_pair = pair
        return max_freq_pair, max_freq
    

    def merge_pair(self, pair):
        """ 
        merge two tokens, '##' is considered
        return str
        """
        # return pair[0] + pair[1][2:]
        return pair[0] + pair[1]
    
    def update_splits(self, pair, word2split, new_byte=None):
        """ 
        update the splits of words according to one rule of a specific pair of tokens
        return dict(word, splits)
        """
        if new_byte is None:
            new_byte = self.merge_pair(pair)
        for word, split in word2split.items():
            if len(word) == 1:
                pass
            else:
                i = 0
                while i < len(split)-1:
                    if (split[i], split[i+1]) == pair:
                        split = split[:i] + [new_byte] + split[i+2:]
                    else:
                        i += 1
                word2split[word] = split
        return word2split

    def train(self, corpus, vocab_size, special_tokens=['<eos>']):
        corpus = [sentence.lower() for sentence in corpus]
        word2freq = self.compute_word_freq(corpus)
        alphabet = self.get_alphabet_from_words(word2freq.keys())
        word2split = self.split_words(word2freq.keys())
        alphabet = self.get_alphabet_from_splits(word2split.values())
        vocab = special_tokens + list(alphabet)
        print(vocab)
        merge_rule = {}
        while len(vocab) < vocab_size:
            # get the pair freq and the biggest
            pair2freq = self.compute_pair_freq(word2split, word2freq)
            pair, freq = self.find_most_frequent_pair(pair2freq)
            # merge rule is kept for faster tokenization
            merge_rule[pair] = self.merge_pair(pair)
            vocab.append(self.merge_pair(pair))
            # update splits according to the new pair rule
            word2split = self.update_splits(pair, word2split)
        print(vocab)
        self.vocab = vocab
        self.merge_rule = merge_rule


    def tokenize(self, text):
        """ 
        just like the loop above
        """
        if self.merge_rule is None:
            raise AttributeError(name="You haven't train it.")
        text = text.lower()
        words_with_offsets = self.pre_tokenizer.pre_tokenize_str(text)
        words = [word for word, offset in words_with_offsets]
        word2split = self.split_words(words)
        for pair, new_byte in self.merge_rule.items():
            word2split = self.update_splits(pair, word2split, new_byte)
        tokenized_words = sum([word2split[word] for word in words], [])
        return tokenized_words
    

if __name__ == '__main__':

    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./DataCollection/officials/gpt2")
    pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    bpt = BPE_Tokenizer(pre_tokenizer)
    bpt.train(corpus, 50)
    print(bpt.tokenize("This is not a token."))
