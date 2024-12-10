# Naive Bayes
from collections import Counter
from collections import defaultdict
from string import punctuation

# negative | positive
#  I           I
#  dislike     like
#  this        this
post_comments_with_labels = [
    ("I love this post.", "pos"),
    ("This post is your best work.", "pos"),
    ("I really liked this post.", "pos"),
    ('I agree 100 percent. This is true', 'pos'),
    ("This post is spot on!", "pos"),
    ("So smart!", "pos"),
    ("What a good point!", "pos"),
    ("Bad stuff.", "neg"),
    ("I hate this.", "neg"),
    ("This post is horrible.", "neg"),
    ("I really disliked this post.", "neg"),
    ("What a waste of time.", "neg"),
    ("I do not agree with this post.", "neg"),
    ("I can't believe you would post this.", "neg"),
]

class NaiveBayesClassifier:
    def __init__(self, samples):
        self.mapping = { "pos": [], "neg": []}
        self.sample_count = len(samples)
        for text, label in samples:
            self.mapping[label] += self._tokenize(text)
        self.pos_counter = Counter(self.mapping["pos"])
        self.neg_counter = Counter(self.mapping["neg"])
        
    @staticmethod
    def _tokenize(text):
        return (
            text.translate(str.maketrans("", "", punctuation + "1234567890"))
            .replace("\n", " ")
            .split(" ")
        )    
        
    def classify(self, text):
        tokens = self._tokenize(text)
        pos = []
        neg = []
        
        for token in tokens:
            pos.append(self.pos_counter[token] / self.sample_count)
            neg.append(self.neg_counter[token] / self.sample_count)
            if (sum(pos) > sum(neg)): return "pos" 
            elif (sum(pos) < sum(neg)): return "neg"
            else: return "neutral"
            
cl = NaiveBayesClassifier(post_comments_with_labels)

def get_sentiment(text):
    cl = NaiveBayesClassifier(post_comments_with_labels)
    return cl.classify(text)
    