import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# Cargar datos
hoteles = pd.read_csv('hoteles.csv')

# Preparar datos para el modelo
hoteles['Ubicacion_Precio'] = hoteles['Municipio'] + ' ' + hoteles['Precio'].astype(str)

vectorizer = TfidfVectorizer(stop_words='spanish')
X = vectorizer.fit_transform(hoteles['Ubicacion_Precio'])

# Entrenar modelo de factorización de matrices no negativas (NMF)
nmf = NMF(n_components=10, random_state=42)
X_nmf = nmf.fit_transform(X)

# Calcular similitud entre hoteles
similitud = cosine_similarity(X_nmf)

# Definir función para obtener recomendaciones
def obtener_recomendaciones(municipio, precio):
    # Encontrar el índice del hotel que más se parece a la ubicación y precio deseada
    indice_semejante = None
    similitud_max = 0
    for i in range(len(hoteles)):
        if hoteles.iloc[i]['Municipio'] == municipio and abs(hoteles.iloc[i]['Precio'] - precio) < 10000:
            similitud_actual = cosine_similarity(X_nmf[i:i+1], X_nmf).flatten()
            similitud_max_actual = max(similitud_actual)
            if similitud_max_actual > similitud_max:
                similitud_max = similitud_max_actual
                indice_semejante = i
    
    # Obtener recomendaciones
    similitud_hoteles = similitud[indice_semejante]
    indices_recomendaciones = similitud_hoteles.argsort()[::-1][1:]
    recomendaciones = hoteles.iloc[indices_recomendaciones]
    return recomendaciones

# Pedir al usuario que proporcione los parámetros municipio y precio
municipio = input("Ingrese el municipio que busca: ")
precio = int(input("Ingrese el precio máximo que está dispuesto a pagar: "))

# Obtener recomendaciones
recomendaciones = obtener_recomendaciones(municipio, precio)
print("Recomendaciones de hoteles en", municipio, "con un precio máximo de", precio, ":")
print(recomendaciones)

