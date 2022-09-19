
# Selección de Modelo de Machine Learning para descripción de Nanopartículas Metálicas: 🌕🌟🌕

![imagen_intro](src/fig/Final-banner-2-1536x480.jpeg) 

---

## The Bridge | Digital Talent Accelerator Bootcamp Data Science

## Machine Learning Project (*Projecto Fin de Bootcamp*)

###  Josefina Cresta

#### Septiembre 2022

---

### Objetivo

---
---
Se entrenan 12 modelos de regresión, de los cuales se seleccionan los mejores 3, que serán capaces de predecir la energía potencial por átomo de nanopartículas metalicas.

---
---
### En [project_resume_1](https://github.com/JosefinaCresta/ML_NanoWorld_Models/blob/master/src/project_resume_1.ipynb)

* Carga base de datos de 185 columnas con información estructural, topologica y energetica de 4000 nanoparticulas de oro.
* Limpieza de datos
* Separación de datos de entrenamiento y evaluación

** Datos de entrenamiento ** 
* Normalización de las variables a utilizar con [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
* Reducción de dimensionalidad con analisis de componentes principales [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA)
* CLustering de los datos disponibles con [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans)

** Entrenamiento de modelos ** 
* Busqueda de los mejores hiperparametros con GridSearchCV y CrossValidation con RepeatedKFold, entre los sigueintes modelos de regresión:
    * LinearRegression, 
    * Ridge, 
    * Lasso, 
    * ElasticNet, 
    * KernelRidge, 
    * DecisionTreeRegressor, 
    * ExtraTreeRegressor, 
    * RandomForestRegressor, 
    * KNeighborsRegressor,
    * GradientBoostingRegressor, 
    * AdaBoostRegressor, 
    * BaggingRegressor, 
    * SVR


### En [project_resume_1](https://github.com/JosefinaCresta/ML_NanoWorld_Models/blob/master/src/project_resume_2.ipynb)
Se realizan predicciónes con la data de evaluación y se calculan las métricas de los mejores modelos de regresión para cada grupo de información. 

Las metricas utilizadas para la selección fueron la raiz cuadratica media del error (RSME) y el coeficiente de determinación [R2](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html?highlight=r2#sklearn.metrics.r2_score). 

Finalmene se escojen los 3 mejores modelos, se guardan junto con su información para su posterior productivización. 




