from Tukey_utils import Tukey_utils
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato y el perro juegan en el jardín",
    "la casa es grande y bonita",
    "los perros y gatos son amigos"
]


tukey = Tukey_utils()
tokens = [tukey.preprocess(sentence, lang='spanish') for sentence in corpus]

vocabulary = tukey.get_vocabulary(tokens)

tfidf = tukey.tf_idf(tokens, vocabulary)

print("########################\n\n")
print(vocabulary)
print(tokens)
print("\n")
print(tfidf)


word_weights = np.sum(tfidf, axis=0)

word_freq = {word: weight for word, weight in zip(vocabulary, word_weights)}

wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()