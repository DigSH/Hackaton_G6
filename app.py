from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el vectorizador, modelo NMF, matriz de similitud y los datos de los hoteles
vectorizer = joblib.load('Python/vectorizer.pkl')
nmf = joblib.load('Python/nmf_model.pkl')
similitud = joblib.load('Python/similitud.pkl')
hoteles = joblib.load('Python/hoteles.pkl')

# Definir función para obtener recomendaciones
def obtener_recomendaciones(municipio, precio, personas, top_n=5):
    hoteles_filtrados = hoteles[
        (hoteles['Municipio'].str.lower() == municipio.lower()) & 
        (hoteles['Precio'] <= precio) &
        (hoteles['Camas'] >= personas)
    ]
    if hoteles_filtrados.empty:
        return pd.DataFrame()

    indice_referencia = hoteles_filtrados.index[0]
    similitud_hoteles = similitud[indice_referencia]
    indices_recomendaciones = similitud_hoteles.argsort()[::-1][1:top_n + 1]
    recomendaciones = hoteles.iloc[indices_recomendaciones]

    recomendaciones = recomendaciones[
        (recomendaciones['Camas'] >= personas) & (recomendaciones['Habitaciones'] >= (personas // 2))
    ]
    return recomendaciones[['Nombre.Comercial', 'Direccion.Comercial', 'Correo.Electronico', 'Habitaciones', 'Camas', 'Empleados']]

# Ruta para obtener recomendaciones
@app.route('/api/recomendaciones', methods=['POST'])
def recomendaciones():
    data = request.json
    municipio = data.get('municipio')
    precio = data.get('precio')
    personas = data.get('personas')

    recomendaciones = obtener_recomendaciones(municipio, precio, personas)
    if recomendaciones.empty:
        return jsonify({"mensaje": "No se encontraron recomendaciones."})
    else:
        return recomendaciones.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
