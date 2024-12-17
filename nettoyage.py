import pandas as pd

# Charger le fichier CSV
data = pd.read_csv('dataset.csv')

# 1. Supprimer la colonne d'index inutile 'Unnamed: 0'
data_cleaned = data.drop(columns=['Unnamed: 0'])

# 2. Vérifier s'il y a des valeurs manquantes
missing_values = data_cleaned.isnull().sum()
print("Valeurs manquantes :\n", missing_values)

# 3. Supprimer les doublons
data_cleaned = data_cleaned.drop_duplicates()

# 4. Renommer les colonnes pour plus de clarté
data_cleaned.columns = ['Label', 'Text']

# Afficher les informations finales
data_cleaned.info()
print("\nAperçu des premières lignes :")
print(data_cleaned.head())

# Sauvegarder le dataset nettoyé si besoin
data_cleaned.to_csv('dataset_cleaned.csv', index=False)
print("\nLe dataset nettoyé a été sauvegardé sous 'dataset_cleaned.csv'.")
