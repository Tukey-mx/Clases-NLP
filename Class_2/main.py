from Tukey_utils import Tukey_utils
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Class 2/spotify_app_reviews.csv')

tukey = Tukey_utils()

df['cleaned_reviews'] = df['Review'].apply(lambda x: tukey.preprocess(x, lang='english'))

vocabulary = tukey.get_vocabulary(df['cleaned_reviews'])

print(f'Vocabulario\n{vocabulary}')
bow_vectors = [tukey.bag_of_words(vocabulary, sentence) for sentence in df['cleaned_reviews']]
#print("Bag of Words Vectors:")

word_freq = np.sum(bow_vectors, axis=0)
print(f'word rfeq')
print(word_freq)

sorted_idx = np.argsort(word_freq)[::-1]
top_words = np.array(vocabulary)[sorted_idx][:10]
top_freqs = word_freq[sorted_idx][:10]

plt.figure(figsize=(10, 6))
plt.bar(top_words, top_freqs)
plt.title("Top 10 Palabras Más Frecuentes")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45)
plt.show()
