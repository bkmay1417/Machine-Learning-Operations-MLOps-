import os
import uvicorn
from fastapi import FastAPI, Request, Query, Path, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app= FastAPI(title='Proyecto Integrador I Hecho por Michael Martinez')
templates = Jinja2Templates(directory="templates")

df_recom = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/2596484f682598ca027fdb025193a4a04d317166/Dataset/recomendacion3_v1.parquet?raw=True')
# Preprocesamiento de los géneros para generar la matriz TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_recom.groupby('item_id')['genres'].apply(lambda x: ' '.join(x)))
# Cálculo de la similitud del coseno

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


@app.get("/", tags=['Página Principal'])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/consulta1")
async def developer(developer:str = Query(default='Monster Games')):
    """
    ( desarrollador : str ): Cantidad de items y porcentaje de contenido 
    Free por año según empresa desarrolladora. Ejemplo de retorno:

    Año	    Cantidad de Items	Contenido Free
    2023	        50                 27%
    2022	        45	               25%
    xxxx	        xx	               xx%
    
    Monster Games
    """
    df_games = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/82702a42172b2b0f23c1e24c6f9fdb294c52d78e/Dataset/developer.parquet?raw=True')

    df_filtrado = df_games[df_games['developer'] == developer]

    # Contar los registros por año
    conteo_por_año = df_filtrado.groupby('release_date').size().reset_index(name='Cantidad de Items')

    # Contar los registros 'Free to Play' por año
    conteo_free_to_play_por_año = df_filtrado[df_filtrado['price'] == 0.00].groupby('release_date').size().reset_index(name='free_to_play_games')

    # Combinar los DataFrames
    df_resultado = pd.merge(conteo_por_año, conteo_free_to_play_por_año, on='release_date', how='left')

    # Calcular el porcentaje de juegos 'Free to Play' por año
    df_resultado['Contenido Free'] = ((df_resultado['free_to_play_games'] / df_resultado['Cantidad de Items']) * 100).map('{:.2f}%'.format)

    df_resultado=df_resultado.drop(columns=['free_to_play_games'])

    # Reemplazar los valores NaN en la columna 'Contenido Free' con '0%'
    df_resultado['Contenido Free'] = df_resultado['Contenido Free'].replace('nan%', '0%')
    # Convertir el DataFrame a un diccionario
    resultado_dict = df_resultado.to_dict(orient='records')
    # Imprimir el DataFrame resultante
    return(resultado_dict)

@app.get("/consulta2")
async def userdata(user_id:str = Query(default='mathzar')):
    """
    ( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario,
     el porcentaje de recomendación en base areviews.recommend y cantidad de items.

    Ejemplo de retorno:
     {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    
    """
    #userdata
    user_data = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/14062c66a031d8e736219f75536dfb552372ac48/Dataset/userdata.parquet?raw=True')
    # Filtrar el DataFrame para el usuario específico
    user_df = user_data[user_data['user_id'] == user_id]
    
    if user_df.empty:
        return {"Usuario": user_id, "Dinero gastado": "0 USD", "% de recomendación": "0%", "cantidad de items": 0}

    # Calcular la cantidad de dinero gastado
    dinero_gastado = user_df['price'].sum()
    
    # Calcular el porcentaje de recomendaciones
    total_reviews = len(user_df)
    recomendaciones_positivas = user_df['recommend'].sum()
    porcentaje_recomendacion = (recomendaciones_positivas / total_reviews) * 100 if total_reviews > 0 else 0
    
    # Calcular la cantidad de ítems
    cantidad_items = user_df['item_id'].nunique()
    
    return {
        "Usuario": user_id,
        "Dinero gastado": f"{dinero_gastado:.2f} USD",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "cantidad de items": cantidad_items
    }




@app.get("/consulta3")
async def UserForGenre(genero:str = Query(default='Action')):
    """
    ( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario,
     el porcentaje de recomendación en base areviews.recommend y cantidad de items.

    Ejemplo de retorno:
     {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    
    """
    df =pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/24007966593d2680fcec0321bcf520ef0e8a2b1f/Dataset/UserForGenre.parquet?raw=True')
   # Filtrar el DataFrame por el género dado
    df_genero = df[df['genres'].apply(lambda x: genero in x)]
    
    # Agrupar por usuario y sumar las horas jugadas para siempre
    user_playtime = df_genero.groupby('user_id')['playtime_forever'].sum()
    
    # Encontrar el usuario con más horas jugadas para el género dado
    max_user = user_playtime.idxmax()
    
    # Utilizar directamente la columna release_year para obtener el año de lanzamiento
    df_genero['year'] = df_genero['release_date'].astype(float)
    
    # Filtrar filas que tienen un año válido
    df_genero = df_genero.dropna(subset=['year'])
    
    # Agrupar por año y sumar las horas jugadas
    hours_by_year = df_genero.groupby('year')['playtime_forever'].sum().reset_index()
    hours_by_year.columns = ['Año', 'Horas']
    
    # Convertir a la estructura de lista de diccionarios requerida
    horas_jugadas = hours_by_year.to_dict(orient='records')
    
    # Construir el resultado final
    resultado = {
        f"Usuario con más horas jugadas para Género {genero}": max_user,
        "Horas jugadas": horas_jugadas
    }
    
    return resultado


    


@app.get("/consulta4")
async def best_developer_year(year: int = Query(default=2005)):
    """
    ( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por 
    usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
    
    Ejemplo de retorno: 
    [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    
    """
    merged_df = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/8f87ccc010ef4ab3025d5e95d5f0cc1ee11fd276/Dataset/best_developer_year.parquet?raw=True')
    merged_df = merged_df[(merged_df['release_date'] == year) ]
    developer_counts = merged_df['developer'].value_counts()
    top_developers = developer_counts.head(3).index
    result = []
    for i, developer in enumerate(top_developers, 1):
        result.append({f"Puesto {i}": developer})


    return(result)

@app.get("/consulta5")
async def developer_reviews_analysis(desarrolladora= Query(default='Valve')):
    """
    ( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el 
    nombre del desarrollador como llave y una lista con la cantidad total de registros de 
    reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento 
    como valor positivo o negativo.
    
    Ejemplo de retorno:
    {'Valve' : [Negative = 182, Positive = 278]}
    
    """
    reviews = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/8f87ccc010ef4ab3025d5e95d5f0cc1ee11fd276/Dataset/reviews_analysis.parquet?raw=True')

    reviews = reviews[(reviews['developer'] == desarrolladora) ]
    counts = reviews['sentiment_analysis'].value_counts()
    # Crear el diccionario de salida
    resultado = {
        desarrolladora: [
            f"Negative = {counts.get(0, 0)}", 
            f"Positive = {counts.get(2, 0)}"
        ]
    }

    
    return(resultado)

@app.get("/Sistema_de_recomendacion")
async def recomendacion_juego(item_id : float = Query(default=6010.0)):
    try:
        # Obtener el índice del juego en el DataFrame
        idx = df_recom[df_recom['item_id'] == item_id].index[0]
    except IndexError:
        # Manejar el caso donde no se encuentra el ID del juego
        raise HTTPException(status_code=404, detail="Item ID no encontrado.")
    
    # Calcular la similitud de coseno entre el juego dado y todos los juegos
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Ordenar los juegos por su similitud de coseno
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    # Obtener los índices de los juegos recomendados
    game_indices = [i[0] for i in sim_scores]
    # Obtener los títulos de los juegos recomendados
    juegos_recomendados = df_recom['title'].iloc[game_indices].tolist() 
    return juegos_recomendados


