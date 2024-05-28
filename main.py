"""
Desarrollador : Michael Martinez.
Github:https://github.com/bkmay1417
email : yam8991@gmail.com
fecha: 28/05/2024
"""
# LIbrerias Utilizadas
from fastapi import FastAPI, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Declaracion de la clase FastAPI
app= FastAPI(title = 'Machine Learning Operations (MLOps)',
    description='API para realizar consultas',
    version='1.01')

#cargar Dataset para ser utilizados por los endpoints
df_recom = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/d5610d0cdf2412a4fdcc6572f39c0268fe51dae9/Dataset/recomendacion.parquet?raw=True')
df_games = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/82702a42172b2b0f23c1e24c6f9fdb294c52d78e/Dataset/developer.parquet?raw=True')
user_data = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/14062c66a031d8e736219f75536dfb552372ac48/Dataset/userdata.parquet?raw=True')
df = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/d20e2c29b6ce650305e0a57f85e9c074d4be8b3f/Dataset/UserForGenre.parquet?raw=True')



# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Se muestra una porta hecha con html
@app.get("/", tags=['Página Principal'])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/developer",tags=["Funciones"])
async def developer(developer:str = Query(default='Monster Games')):
    """
    ( desarrollador : str ): Cantidad de items y porcentaje de contenido 
    Free por año según empresa desarrolladora. Ejemplo de retorno:
    """
    #Año	    Cantidad de Items	Contenido Free
    #2023	        50                 27%
    #2022	        45	               25%
    #xxxx	        xx	               xx% 

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

@app.get("/userdata",tags=["Funciones"])
async def userdata(user_id:str = Query(default='mathzar')):
    """
    ( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario,
     el porcentaje de recomendación en base areviews.recommend y cantidad de items.

    Ejemplo de retorno:
     {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    
    """
    #userdata
    
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




@app.get("/UserForGenre", tags=["Funciones"])
async def UserForGenre(genero:str = Query(default='action')):
    """
    def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el 
    género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
    Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, 
    "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
    """
   
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


@app.get("/best_developer_year",tags=["Funciones"])
async def best_developer_year(year: int = Query(default=2005)):
    """
    ( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por 
    usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
    
    Ejemplo de retorno: 
    [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    
    """
    best_developer = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/8f87ccc010ef4ab3025d5e95d5f0cc1ee11fd276/Dataset/best_developer_year.parquet?raw=True')
    best_developer = best_developer[(best_developer['release_date'] == year) ]
    developer_counts = best_developer['developer'].value_counts()
    top_developers = developer_counts.head(3).index
    result = []
    for i, developer in enumerate(top_developers, 1):
        result.append({f"Puesto {i}": developer})


    return(result)


@app.get("/developer_reviews_analysis", tags=["Funciones"])
async def developer_reviews_analysis(desarrolladora= Query(default='Valve')):
    """
    ( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el 
    nombre del desarrollador como llave y una lista con la cantidad total de registros de 
    reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento 
    como valor positivo o negativo.
    
    Ejemplo de retorno:
    {'Valve' : [Negative = 182, Positive = 278]}
    
    """
    dev_reviews = pd.read_parquet(r'https://github.com/bkmay1417/Machine-Learning-Operations-MLOps-/blob/8f87ccc010ef4ab3025d5e95d5f0cc1ee11fd276/Dataset/reviews_analysis.parquet?raw=True')

    dev_reviews = dev_reviews[(dev_reviews['developer'] == desarrolladora) ]
    counts = dev_reviews['sentiment_analysis'].value_counts()
    # Crear el diccionario de salida
    resultado = {
        desarrolladora: [
            f"Negative = {counts.get(0, 0)}", 
            f"Positive = {counts.get(2, 0)}"
        ]
    }

    
    return(resultado)


@app.get("/Sistema_de_recomendacion por Genero", tags=["Sistema de Recomendacion"])
async def recomendacion_juego(item_id: float = Query(default=10.0)):
    """
    10.0 = Counter-Strike
    """
    # Verificar que el item_id exista en el DataFrame
    if item_id not in df_recom['item_id'].values:
        return "El juego con el item_id proporcionado no existe."

    # Obtener el índice del juego dado su item_id
    idx = df_recom[df_recom['item_id'] == item_id].index[0]

    # Vectorizar los géneros
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_recom['genres_str'])

    # Obtenemos el vector tf-idf del item_id ingresado
    item_tfidf_vector = tfidf_matrix[idx]

    # Calcular la similitud del coseno
    cosine_sim = cosine_similarity(item_tfidf_vector, tfidf_matrix)

    # Obtener las puntuaciones de similitud del juego con todos los demás juegos
    sim_scores = list(enumerate(cosine_sim.flatten()))

    # Ordenar los juegos por puntuación de similitud (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los 5 juegos más similares (excluyendo el propio juego)
    sim_scores = sim_scores[1:6]

    # Obtener los item_id de los 5 juegos más similares
    game_indices = [i[0] for i in sim_scores]

    return df_recom['title'].iloc[game_indices].tolist()


