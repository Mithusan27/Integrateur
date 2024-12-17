import pandas as pd
import re
import tensorflow as tf

# Charger le fichier CSV nettoyé
data = pd.read_csv('dataset.csv')

# 1. Prétraitement des textes
def preprocess_text(text):
    text = text.lower()  # Conversion en minuscules
    text = re.sub(r'[^a-z\s]', '', text)  # Suppression des caractères spéciaux et chiffres
    return text

data['Cleaned_Text'] = data['MainText'].apply(preprocess_text)

# 2. Tokenization avec TensorFlow
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data['Cleaned_Text'])

data['Tokenized_Text'] = tokenizer.texts_to_sequences(data['Cleaned_Text'])

# 3. Sauvegarder les tokens nettoyés dans un fichier CSV
data[['MainText', 'Tokenized_Text']].to_csv('tokens_cleaned.csv', index=False)
print("Les tokens ont été sauvegardés dans 'tokens_cleaned.csv'.")
