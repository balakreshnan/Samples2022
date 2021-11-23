# Deploy Custom NLTK model to Azure ML inferencing in AKS

## Custom NLTK model deployment

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Create a compute instance for notebook

## Note book Code

```
!pip install nltk
```

```
import nltk
```

```
nltk.download('punkt')
```

```
nltk.download('movie_reviews')
```

```
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]
```

```
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
```

```
featuresets = [(find_features(rev), category) for (rev, category) in documents]
```

```
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]
```

```
classifier = nltk.NaiveBayesClassifier.train(training_set)
```

```
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
```

```
classifier.show_most_informative_features(15)
```

```
import pickle
```

```
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
```

```
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
#classifier_f.close()
```

```
sorted(classifier.labels())
```

```
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

```
review_santa = '''

It would be impossible to sum up all the stuff that sucks about this film, so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume (funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear; an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly and caaarreee-fulll-yyy prrooo-noun-ceeed; a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.

The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
'''
print(review_santa )
```

```
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))

#print(pos_reviews[0])
print(len(pos_reviews))
```

```
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

#print(pos_reviews[0])
print(len(neg_reviews))
```

```
# This is how the Naive Bayes classifier expects the input
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict
```

```
words = word_tokenize(review_santa)
words = create_word_features(words)
classifier.classify(words)
```

```
classifier_f.close()
```

