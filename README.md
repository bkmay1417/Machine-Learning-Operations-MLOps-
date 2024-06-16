![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)
![Numpy](https://img.shields.io/badge/-Numpy-333333?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-333333?style=flat&logo=matplotlib)
![Scikitlearn](https://img.shields.io/badge/-Scikitlearn-333333?style=flat&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/-FastAPI-333333?style=flat&logo=fastapi)
![Render](https://img.shields.io/badge/-Render-333333?style=flat&logo=render)

# Machine Learning Operations MLOps

![Descripción de la imagen](https://images.squarespace-cdn.com/content/v1/5feb53185d3dab691b47361b/1609930650139-9NRI63XUJ29Y7E9LEA9G/12eca-machine-learning.gif)


<p id="readme-top"></p>

<div style="display: flex; align-items: center;">
    <img src="https://img.shields.io/badge/STATUS-EN%20DESAROLLO-green" />
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/bkmay1417/Machine-Learning-Operations-MLOps-" />
</div>

## Tabla de Contenidos

[Introducion](#Introducion) | [Exploración, Transformación y Carga (ETL)](#Exploración,-Transformación-y-Carga-(ETL)) | [Funcionalidades de la API con FastAPI](#Funcionalidades-de-la-API-con-FastAPI) | [Fastapi](#Fastapi)  | [Render](#Render) 

## Introducion
Este proyecto es una API web desarrollada con FastAPI deployada en render que permite acceder a varios servicios relacionados con análisis de datos de videojuegos de la platatafoma de steam para poder crear un Producto Minimo Viable (MVP), que contiene una la implementaciónde una API. La API proporciona endpoints para obtener información sobre desarrolladores, usuarios, géneros de videojuegos, recomendaciones de juegos, análisis de reseñas y un sistema de recomendacion hecho en base a la  similitud del coseno. Los datos se obtienen de archivos json los cual fueron limpiados y almacenados en formato parquet .

## Exploración, Transformación y Carga (ETL)

![Descripción de la imagen](https://assets-global.website-files.com/634fa785d369cb60d80b6dd1/6393298e18f50e62a1657530_ETL%20process%20DataChannel.webp)

A partir de los 3 datasets proporcionados (steam_games, user_reviews y user_items) referentes a la plataforma de Steam, en el cual dos de ellos, "review" y "item", requirieron un tratamiento especial y la creación de una función para poder ser leídos, ya que ambos archivos estaban corruptos. Estas bases de datos fueron guardadas en nuevos archivos en formato Parquet y comprimidos en formato shappy para mejorar su lectura para el ETL en primera instancia se realizó el proceso de limpieza de los datos.

### Steam games
- Se eliminaron filas nulas, valores nulos y duplicados.
- Se cambió el tipo de dato a Int.
- Se identificaron valores únicos.
- Se procesó cada formato de fecha para extraer años y los no válidos se eliminaron.
### User reviews
- Se eliminaron filas y columnas con valores nulos porque los consideré irrelevantes.
- Se separaron todas las reviews generando una fila por cada review dentro de la columna "reviews".
- Se creó una nueva columna llamada 'sentiment_analysis' usando análisis de sentimiento y se eliminó la columna de review.
- Se exportó para tener el dataset limpio.
### User Items
- Se procesaron los datos para el conjunto de los items que posee cada usuario.
- Se eliminaron columnas innecesarias para el desarrollo de la API.
- Se separaron todas los items generando una fila por cada item dentro de la columna "item"

## Análisis Exploratorio de Datos (EDA)

![Descripción de la imagen](https://i.ibb.co/PFJKrQg/stock-photo-eda-concept-text-sunlight-d-illustration-2267252419-transformed.jpg)


Se realizó un análisis exploratorio de los datos de los 3 conjuntos de datos para así entender las estadísticas identificando patrones y tendencias de los juegos y géneros mas recomendados por los usuarios, encontrar valores atípicos y orientar un futuro análisis.

[EDA]("./EDA.ipynb") 

## Fastapi :

<div align="center">
  <img src="https://media.licdn.com/dms/image/D5612AQH0PUY2JLO2mg/article-cover_image-shrink_600_2000/0/1697375557641?e=2147483647&v=beta&t=0FpqTAdO-BiTXihJP1HymWJfPf87WymWLJj75mYWC4U" alt="Ejemplo de Imagen">
</div>

 Se construyeron 5 funciones que permiten realizar consultas mas un sistema de recomenacion


Estas funciones son:

+ **`developer:`** Esta función recibe como parámetro el desarrollador en formato str y devuelve un diccionario con la cantidad de items por año y el porcentaje de juegos free que hay en cada año.

+ **`UserForGenre:`** Esta función recibe como parámetro el género en str de un juego y devuelve el usuario que ha jugado más horas en ese género, junto con las horas jugadas por ese usuario en ese género a lo largo de los años.

+ **`userdata:`** Esta función devuelve el gasto total, el porcentaje de recomendaciones positivas y la cantidad de ítems de un usuario.

+ **`developer_reviews_analysis:`** Analiza y muestra la cantidad de reseñas positivas y negativas de un desarrollador específico.

+ **`best_developer_year:`** Esta función recibe como parametro el str año y devuelve el top 3 de desarrolladoras con juegos mas recomendados por usuarios para el año dado.

+ **`Sistema_de_recomendacion_generos:`** Proporciona una lista de juegos recomendados basados en la similitud de géneros usando un sistema de recomendación basado en TF-IDF y similitud de coseno.

### Extra

+ **`Read_root:`** Esta función maneja la solicitud GET a la ruta raíz ("/"). Utiliza Jinja2Templates para renderizar una página HTML utilizada como portada del proyecto.

<p align="center"><img src="img/gifml.gif" alt="portada"  height="100%" width="100%" /></p>

Se creo el entorno vitual para fastapi y se construyo el main.py necesario para usar las funciones.

## RENDER: 

![Descripción de la imagen](https://docs.vendure.io/assets/images/deploy-to-render-b2b7fbd4d3153076c1e91c3d9969a719.webp)


+ [Fastapi main link](/main.py)

Finalmente se uso Render para deployar fastapi en una página web.

[link del Proyecto](https://machine-learning-operations-mlops-f1jx.onrender.com)

[link del video](https://drive.google.com/file/d/1es5iFOBZCpHyY4xwT2KMiooBb79Baj1I/view?usp=drive_link)



## Desarrolladores

| [<img src="https://avatars.githubusercontent.com/u/163685041?v=4" width=115><br><sub>Michael Martinez</sub>](https://github.com/bkmay1417) |
| :---: |

Copyright (c) 2024 [Michael Martinez] yam8991@gmail.com







