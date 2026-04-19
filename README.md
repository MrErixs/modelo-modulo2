# Clasificación de reseñas de recetas

## Objetivo
El objetivo de este proyecto es construir un modelo de clasificación de texto capaz de identificar si una reseña de receta es positiva o negativa a partir de su contenido textual.


## Dataset seleccionado
Se seleccionó el dataset **Recipe Reviews and User Feedback**, el cual contiene comentarios textuales de usuarios sobre recetas junto con información adicional como número de estrellas, votos y reputación del usuario.

Se eligió este porque:
- permite plantear un problema de clasificacion supervisada
- tiene el nunero adecuado de instancias
- requiere una etapa real de limpieza y preparación de datos

En un contexto real, este tipo de modelo podría utilizarse para analizar automáticamente grandes volúmenes de comentarios de usuarios, identificando opiniones positivas y negativas para apoyar la toma de decisiones, mejorar contenido y detectar problemas frecuentes.

## Variables utilizadas
Se utilizaron las siguientes columnas:
- "text": es el comentario textual del usuario. Se utilizó como la variable de entrada.
- "stars": calificacion otorgada por el usuario. Se utilizó como base para construir la variable objetivo.

## Problema planteado
A partir del contenido de la reseña, el objetivo es clasificar si la opinión es positiva o negativa.

Para ello se construyó la etiqueta:
- **1 y 2 estrellas** se consideran reseñas negativas (`0`)
- **4 y 5 estrellas** se consideran reseñas positivas (`1`)

No se tomaron en cuenta las reseñas con **0 y 3 estrellas**:
- `0` se consideró no informativo
- `3` se consideró una valoración neutral o ambigua

Esto se hizo para evitar casos ambiguos, por ejemplo comentarios como *“It was okay, not great”* o *“Pretty decent recipe”*.

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

## Desvalance de clases
Al analizar la distribución de la variable objetivo, se observó un fuerte desbalance entre clases. Aproximadamente:
- **96.72%** de las instancias corresponden a reseñas positivas
- **3.28%** corresponden a reseñas negativas

Esta característica tuvo un impacto importante en el entrenamiento y la evaluación del modelo, ya que una métrica como accuracy por sí sola podía dar una impresión engañosa del desempeño real.

## Archivos generados
Como resultado de esta etapa se generaron los siguientes archivos:

- `reviews_binary_clean.csv`: dataset completo y limpio
- `train.csv`: conjunto de entrenamiento
- `test.csv`: conjunto de prueba
- `preprocess.py`: código utilizado para el preprocesamiento

## Representación numérica del texto
Para poder entrenar el modelo, el texto fue transformado a una representación numérica en varias etapas:

1. tokenización del texto
2. conversión a secuencias numéricas
3. padding para igualar la longitud de las secuencias

Posteriormente se utilizó una capa **Embedding**, la cual aprende representaciones densas de las palabras durante el entrenamiento. Esto permitió que el modelo trabajara con una representación más útil del lenguaje.

## Modelo implementado
Se implementó un modelo de clasificación binaria en **TensorFlow/Keras** con una arquitectura sencilla basada en:
- capa `Embedding`
- capa `GlobalAveragePooling1D`
- capa `Dense` con activación ReLU
- capa de salida `Dense(1)`

La función de pérdida utilizada fue `binary_crossentropy` y el optimizador fue `adam`.

## Manejo del desbalance
Para reducir el efecto del desbalance de clases, se utilizó `class_weight`, calculado con `compute_class_weight` de scikit-learn.

Esto permitió asignar mayor peso a la clase minoritaria durante el entrenamiento, de manera que los errores cometidos sobre esa clase tuvieran mayor penalización.

## Experimentos y refinamiento
Después de implementar el modelo base, se realizaron varias pruebas modificando:
- número de épocas
- dimensión del embedding
- tamaño de la capa densa

La configuración base fue:
- **epochs = 5**
- **embedding dim = 16**
- **dense = 24**

Sin embargo, durante las pruebas se observó que el entrenamiento presentaba variaciones entre ejecuciones, especialmente en el F1-score de la clase minoritaria.

Por esta razón, no se seleccionó la configuración final únicamente a partir de una corrida aislada con un valor alto, sino a partir de **múltiples pruebas repetidas**. Además del valor máximo obtenido, también se consideró la **consistencia** de cada configuración.

La configuración que mostró mejor comportamiento de forma más estable fue:
- **epochs = 15**
- **embedding dim = 24**
- **dense = 24**

Esta configuración fue la que con mayor frecuencia produjo valores de **F1-score para la clase 0 por encima de 0.50**, por lo que se consideró la mejor opción final.

## Resultados
En la versión base, el modelo logró una accuracy alta, pero todavía presentaba dificultades importantes para clasificar correctamente la clase minoritaria.

### Modelo base
- **epochs = 5**
- **embedding dim = 16**
- **dense = 24**

Resultados representativos:
- **Accuracy:** 0.96
- **F1-score clase 0:** 0.54
- **Macro average F1:** 0.76

### Modelo final seleccionado
- **epochs = 15**
- **embedding dim = 24**
- **dense = 24**

Resultados representativos:
- **Accuracy:** 0.9681
- **F1-score clase 0:** 0.58
- **Macro average F1:** 0.78

## Interpretación de resultados
Los experimentos mostraron que:
- el modelo clasifica muy bien la clase positiva
- el principal reto es detectar correctamente la clase negativa, ya que es la minoritaria
- la accuracy por sí sola no es suficiente para evaluar el desempeño
- métricas como **precision, recall, F1-score y macro average** describen mejor el comportamiento real del modelo
- el uso de embeddings y `class_weight` ayudó a mejorar el desempeño sobre la clase minoritaria
