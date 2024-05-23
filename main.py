from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=['Página Principal'])
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/consulta1")
async def developer(developer:str):
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
async def userdata():
    """
    ( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario,
     el porcentaje de recomendación en base areviews.recommend y cantidad de items.

    Ejemplo de retorno:
     {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    
    """
    return()

@app.get("/consulta3")
async def UserForGenre():
    """
    ( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario,
     el porcentaje de recomendación en base areviews.recommend y cantidad de items.

    Ejemplo de retorno:
     {"Usuario X" : us213ndjss09sdf, "Dinero gastado": 200 USD, "% de recomendación": 20%, "cantidad de items": 5}
    
    """
    return()

@app.get("/consulta4/{year}")
async def best_developer_year(year:int):
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
async def developer_reviews_analysis(desarrolladora):
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