# Primer avance - Clasificación de reseñas de recetas

## Objetivo
El objetivo de este proyecto es construir un sistema de clasificación de texto. En esta primera etapa se realizó la selección del dataset, el preprocesamiento de los datos y la separación en conjuntos de entrenamiento y prueba, como base para el entrenamiento posterior de un modelo de aprendizaje.

## Dataset seleccionado
Se seleccionó el dataset **Recipe Reviews and User Feedback**, el cual contiene comentarios textuales de usuarios sobre recetas junto con información adicional como número de estrellas, votos y reputación del usuario.

Se eligió este porque:
- permite plantear un problema de clasificacion supervisada
- tiene el nunero adecuado de instancias
- requiere una etapa real de limpieza y preparación de datos

En un contexto real, este tipo de modelo podría utilizarse para analizar automáticamente grandes volúmenes de comentarios de usuarios, identificando opiniones positivas y negativas para apoyar la toma de decisiones, mejorar contenido y detectar problemas frecuentes.

## Variables utilizadas
Se utilizaron las siguientes columnas:
- "text": es el comentario textual del usuario. Esta columna se utilizara como la variable de entrada.
- "stars": calificacion otorgada por el usuario. Esta columna se utilizó como base para construir la variable objetivo.

## Problema planteado
A partir del contenido de la reseña, el objetivo sera clasificar si la opinion es positiva o negativa.

Por ello se construyó la etiqueta:
- 1 y 2 estrellas se consideran reseñas negativas (0)
- 4 y 5 estrellas se consideran reseñas positivas (1)
- No se tomaron en cuenta 0 y 3 estrellas por representar casos ambiguos o no informativos.
- Se manejo de esta manera para evitar casos ambiguos como comentarios de tipo "It was okey, no great" o "Pretty decent recipe"

## Preprocesamiento realizado
1. Selección de columnas relevantes 
   Se conservaron únicamente las columnas necesarias: `text` y `stars`.

2. Eliminación de valores nulos 
   Se descartaron registros con texto vacío o sin calificación válida.

3. Filtrado de clases
   Se eliminaron los registros con valor de `stars` igual a 0 y 3:
   - `0` se consideró no informativo
   - `3` se consideró una valoración neutral o ambigua

4. Construcción de la etiqueta objetivo
   Se generó una nueva columna `label`:
   - `0` = negativo
   - `1` = positivo

5. Limpieza del texto 
   Se aplicaron transformaciones básicas al texto:
   - conversión a minúsculas
   - corrección de entidades HTML
   - normalización de espacios en blanco

6. Eliminación de duplicados
   Se eliminaron registros repetidos para reducir ruido en el dataset.

## Separación de datos
Después del preprocesamiento, el dataset fue dividido en dos subconjuntos:

- **80% para entrenamiento**
- **20% para prueba**

## Archivos generados
Como resultado de esta etapa se generaron los siguientes archivos:

- `reviews_binary_clean.csv`: dataset completo y limpio
- `train.csv`: conjunto de entrenamiento
- `test.csv`: conjunto de prueba
- `preprocess.py`: código utilizado para el preprocesamiento

## NOTAS IMPORTANTES
Al analizar la distribución de la variable objetivo, se observó un fuerte desbalance entre clases. Aproximadamente el 96.72% de las instancias corresponden a reseñas positivas y solo el 3.28% a reseñas negativas. Esta característica deberá considerarse en etapas posteriores del modelado y evaluación.

## Siguiente paso
En la siguiente etapa del proyecto se realizará la transformación del texto a representaciones numéricas adecuadas para el modelo, así como la implementación y entrenamiento de un modelo de clasificación utilizando un framework de aprendizaje profundo.

# Segundo avance - Implementar el modelo usando un framework seleccionado.

## Objetivo
Desarrollar una primera implementación funcional de un modelo de clasificación de reseñas positivas y negativas, utilizando un framework de aprendizaje automático. En esta etapa se busca partir de los datos previamente preprocesados, convertir el texto a una representación numérica adecuada para el modelo y realizar un entrenamiento inicial que permita evaluar de manera preliminar su desempeño.

