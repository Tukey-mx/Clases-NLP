import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

class Tukey_utils:

    def __init__(self):
        pass

    def letter_to_lower(self, letter: str) -> str:
        ascii_code = ord(letter)
        if 65 <= ascii_code <= 90:
            return chr(ascii_code + 32)
        else:
            return letter

    def word_to_lower(self, word: str) -> str:
        result = ''
        for letter in word:
            result += self.letter_to_lower(letter)
        return result
    
    def remove_special(self,  text: str, remove_numbers: bool = True) -> str:
        result = ''
        for char in text:
            if char.isalpha() or char.isspace():
                result += char
            elif char.isdigit() and not remove_numbers:
                result += char
        return result
    
    def tokenize(self, text: str) -> list:
        tokens = list()
        j = 0
        for i in range(len(text)):
            char = text[i]
            if char == ' ':
                tokens.append(text[j:i])
                j = i + 1
        tokens.append(text[j:])

        return tokens

    def remove_stopwords(self, tokens: list, lang: str = 'spanish') -> list:
        stop_words = set(stopwords.words(f'{lang}'))
        
        return [token for token in tokens if token not in stop_words]
    
    def get_vocabulary(self, tokens:list) -> list:
        vocabulary = set()
        for sentence in tokens:
            vocabulary.update(sentence)
        
        return list(vocabulary)
    
    def preprocess(self, text: str, lang: str) -> list:
        # Minusculas
        text = self.word_to_lower(text)
        # Eliminar puntuación, números
        text = self.remove_special(text, remove_numbers=False)
        text = text.strip() # Eliminar posibles espacios al inicio y al final
        # Tokeniazcion
        tokens = self.tokenize(text)
        
        tokens = self.remove_stopwords(tokens, lang)
        return tokens
    
    def bag_of_words(self, vocab: list, sentence_tokens: list):
        vector = [0] * len(vocab)
        for word in sentence_tokens:
            if word in vocab:
                idx = vocab.index(word)
                vector[idx] += 1
        return vector
