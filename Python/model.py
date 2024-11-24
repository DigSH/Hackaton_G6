# Script 2: Cargar el modelo y realizar recomendaciones
import pandas as pd
import joblib

# Definir función para obtener recomendaciones
def obtener_recomendaciones(municipio, precio, personas, top_n=5):
    """
    Obtener recomendaciones de hoteles en un municipio con un precio máximo.

    Parámetros:
    municipio (str): Municipio que busca.
    precio (int): Precio máximo que está dispuesto a pagar.
    personas (int): Número de personas para la reserva.
    top_n (int): Número de recomendaciones a devolver.

    Retorna:
    recomendaciones (pd.DataFrame): Recomendaciones de hoteles.
    """
    # Cargar el vectorizador, modelo NMF, matriz de similitud y los datos de los hoteles
    vectorizer = joblib.load('vectorizer.pkl')
    nmf = joblib.load('nmf_model.pkl')
    similitud = joblib.load('similitud.pkl')
    hoteles = joblib.load('hoteles.pkl')

    # Filtrar los hoteles en el municipio deseado, con precio cercano y suficientes camas
    hoteles_filtrados = hoteles[
        (hoteles['Municipio'].str.lower() == municipio.lower()) & 
        (hoteles['Precio'] <= precio) &
        (hoteles['Camas'] >= personas)
    ]
    
    if hoteles_filtrados.empty:
        return pd.DataFrame()

    # Encontrar el índice del primer hotel filtrado como referencia para similitud
    indice_referencia = hoteles_filtrados.index[0]
    
    # Calcular las similitudes de los hoteles filtrados con respecto al hotel de referencia
    similitud_hoteles = similitud[indice_referencia]
    indices_recomendaciones = similitud_hoteles.argsort()[::-1][1:top_n + 1]  # Obtener las `top_n` recomendaciones más similares
    
    recomendaciones = hoteles.iloc[indices_recomendaciones]
    
    # Filtrar las recomendaciones para asegurarse de que tengan suficientes camas y habitaciones
    recomendaciones = recomendaciones[(recomendaciones['Camas'] >= personas) & (recomendaciones['Habitaciones'] >= (personas // 2))]
    
    return recomendaciones[['Nombre.Comercial', 'Direccion.Comercial', 'Correo.Electronico', 'Habitaciones', 'Camas', 'Empleados']]

# Pedir al usuario que proporcione los parámetros municipio, precio y número de personas
municipio = input("Ingrese el municipio que busca: ")
precio = int(input("Ingrese el precio máximo que está dispuesto a pagar: "))
personas = int(input("Ingrese el número de personas para la reserva: "))

# Obtener recomendaciones
recomendaciones = obtener_recomendaciones(municipio, precio, personas)
if not recomendaciones.empty:
    print(f"Recomendaciones de hoteles en {municipio} con un precio máximo de {precio} para {personas} personas:")
    print(recomendaciones)
else:
    print(f"No se encontraron recomendaciones de hoteles en {municipio} con un precio máximo de {precio} para {personas} personas")
