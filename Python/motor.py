import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
import joblib

# Descargar stopwords si no están disponibles
try:
    nltk.corpus.stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# Cargar datos
hoteles = pd.read_csv('hoteles.csv')

# Cargar stopwords en español
stop_words_es = list(nltk.corpus.stopwords.words('spanish'))  # Convertir a lista

# Preparar datos para el modelo
hoteles['Municipio_Precio'] = hoteles['Municipio'] + ' ' + hoteles['Precio'].astype(str)

vectorizer = TfidfVectorizer(stop_words=stop_words_es)
X = vectorizer.fit_transform(hoteles['Municipio_Precio'])

# Entrenar modelo de factorización de matrices no negativas (NMF)
nmf = NMF(n_components=10, random_state=42)
X_nmf = nmf.fit_transform(X)

# Calcular similitud entre hoteles
from sklearn.metrics.pairwise import cosine_similarity
similitud = cosine_similarity(X_nmf)

# Guardar el vectorizador, modelo NMF y matriz de similitud
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(nmf, 'nmf_model.pkl')
joblib.dump(similitud, 'similitud.pkl')
joblib.dump(hoteles, 'hoteles.pkl')

print("Modelos y objetos guardados exitosamente.")