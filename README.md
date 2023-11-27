# IA_B2_M2_E1
---
A01367585 | Fabian Gonzalez Vera

En este repositorio se encuentra, una implementacion de una red neuronal convolucional utilizando el framework de TensorFlow y el usando el modelo VGG16 como base. El fin de esta implementacion es determinar la especie a la que pertenece un ave dado una imagen.

Dentro del archivo *actividadM2.ipynb* se encuentra un EDA y el desarrollo de los modelos, la implementacion final del modelo se encuentra dentro del archivo [*modelTraining.py*](https://github.com/fabian994/IAB2_M2Actividad/blob/main/modelTraining.py), dentro del archivo [*modelPredictions.py*](https://github.com/fabian994/IAB2_M2Actividad/blob/main/modelPredictions.py) se encuentra un script para correr el modelo. Dentro del archivo [*setup.py*](https://github.com/fabian994/IAB2_M2Actividad/blob/main/setup.py) se encuentra un script para descargar los archivos necesarios para poder correr los scripts antes mencionados.



## Requisitos Minimos

 - numpy==1.24.3
 - pandas==2.0.3
 - matplotlib==3.7.2
 - tensorflow==2.13.0
 - scikit-learn==1.3.0



# Instrucciones de Uso

 1. Correr el archivo *setup.py*
 1. Para entrenar el modelo correr el archivo *modelTraining.py*
 1. Para entrenar el modelo correr el archivo *modelPredictions.py*



## Dataset

El dataset fue obtenido del repositorio de Instituto Tecnológico de California. Contiene imagenes de 200 distintas especies de aves.

Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2022). CUB-200-2011 (1.0) [Data set]. CaltechDATA. https://doi.org/10.22002/D1.20098