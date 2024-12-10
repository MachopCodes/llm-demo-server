import random
from collections import defaultdict

tokens = ["I", "try", 'to', 'learn', 'something', 'new', 'every', 'day']
graph = defaultdict(list)

class MarkovChain:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def _tokenize(self, text):
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )
        
    def train(self, text):
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            if (len(tokens) - 1) == i:
                break
            self.graph[token].append(tokens[i + 1])
            
    def generate(self, prompt, length = 10):
        # lastd token from prompt
        current = self._tokenize(prompt)[-1]
        # initialize output
        output = prompt
        for i in range(length):
            # look up options in the graph dictionary
            options = self.graph.get(current, [])
            if not options:
                continue
            # use random choice method to pick option
            current = random.choice(options)
            # add random choice to output string
            output += f" {current}"
            return output




