
######################################################################################################################################################

def missing_report(df):
        '''
        Pequeño reporte de los % de missings de cada columna del dataframe

                Parámetros:
                        df: (pd.DataFrame) dataset, que se analizará
                Returns:
                        (pd.DataFrame) con información calculada
        '''
        precent_missing = df.isnull().sum()*100/len(df)
        missing_value_df = pd.DataFrame({'column_name': df.columns,
                                'percent_missing': precent_missing}).sort_values('percent_missing', ascending=False)
        return missing_value_df


######################################################################################################################################################

def zeros_report(df):
    '''
        Pequeño reporte de los % de zeros de cada columna del dataframe

                Parámetros:
                        df: (pd.DataFrame) dataset, que se analizará
                Returns:
                        (pd.DataFrame) con información calculada
    '''
    precent_zeros = (df==0).sum()*100/len(df)
    zeros_value_df = pd.DataFrame({'column_name': df.columns,
                                'percent_zeros': precent_zeros}).sort_values('percent_zeros', ascending=False)
    
    with pd.option_context("display.max_rows", None):
        display(zeros_value_df) 
    cols_to_drop = zeros_value_df[zeros_value_df['percent_zeros'] > 60].index.values
    print("Columnas con más del 60 porciento de valores igual a cero:", cols_to_drop)

######################################################################################################################################################


def del_zeros(df):
    '''
    Eliminación de variables que contienen más de 60 porciento de valores igual a cero.

            Parámetros:
                    df: (pd.DataFrame) dataset que se desea limpiar 
            Returns:
                    nombre de las variables eliminadas.
                    cantidad de filas antes y despues de la limpieza
    '''
    precent_zeros = (df==0).sum()*100/len(df)
    zeros_value_df = pd.DataFrame({'column_name': df.columns,
                                'percent_zeros': precent_zeros}).sort_values('percent_zeros', ascending=False)
    cols_to_drop = zeros_value_df[zeros_value_df['percent_zeros'] > 60].index.values
    print("Cols:", cols_to_drop)

    print("Columnas pre drop:", len(df.columns))

    df.drop(columns=cols_to_drop, inplace=True)

    print("Columnas post drop:", len(df.columns))

######################################################################################################################################################

def modeling(model, parameters, df_train_scaled_pca_cluster, n_clusters, dict_models):
    '''
    Dividir dataframe entre features y target, 
    y estos mismos en subconjuntos aleatorios de entrenamiento y prueba.
    Configuración de validación cruzada repetida de K-Fold y evaluar 
    y GridSearchCV para seleccionar de forma sistemática los parámetros (parameters) del modelo (model) deseado.

                Parámetros:
                        model: (Scikit-Learn model) estimator modelo a usar
                        parameters: (dict) diccionario conteniendo como "claves" los hiperparámetros del modelo, 
                                    y como "valores" los valores de los hiperparámetros que queremos probar
                        data: (pd.DataFrame) base de datos
                Returns:
                        mejores hiperparametros
                        mejor score
                        tiempo de entrenamiento y de predicción 
                        metricas del mejor modelo
                        escritura en fichero de información
    '''

    for i in range(n_clusters):
        df=df_train_scaled_pca_cluster[df_train_scaled_pca_cluster["Cluster"]==i].drop(columns="Cluster")

        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123)

        modelo_grid = GridSearchCV(estimator= model, 
        param_grid = parameters,
        scoring = "r2", 
        cv = cv, n_jobs= -1, verbose=0)

        start = dt.datetime.now()

        model_fit = modelo_grid.fit(df.drop(columns="target"), df["target"])

        dict_models[f"{model.__class__.__name__}_cluster_"+str(i)] = model_fit


        print(f'{model.__class__.__name__}_cluster_{i} :')
        print('\t Best Hyperparameters: %s' % model_fit.best_params_)
        print('\t Best Score: %s' % model_fit.best_score_)
        
        train_time = (dt.datetime.now() - start).seconds
        print("\t Training time: %0.3fs" % train_time)

    with open(f"utils/pickle/models/{model.__class__.__name__}_entrenados.pkl", "wb") as file:
        pickle.dump(dict_models, file)

######################################################################################################################################################

def get_key(dicc, val):
    for key, value in dicc.items():
         if val == value:
             return key
 
    return "No está ese valor"


######################################################################################################################################################

def guardar_en_fichero(ruta_target, row):
    '''
    Guarda en fichero csv datos de las metricas de los modelos entrenados
    '''
    #comprobacion que exista fichero / directorio
    try:
        if os.path.isfile(ruta_target):
            print ("File exist")
            with open(f'{ruta_target}', 'a') as out:
                #out.write(f"{content[0]} , {content[1]} ,{content[2]} \n")
                writer = csv.writer(out)
                writer.writerow(row)
        else:
            print ("File not exist")
            with open(f'{ruta_target}', 'w') as out:
                out.write("model_name , cluster, hyper_parametros, best_score, R2,  MAE, MSE, RMSE, MAPE\n")
                writer = csv.writer(out)
                writer.writerow(row)
                print("Model Info saved")
    except Exception as e:
            print(e)

######################################################################################################################################################

def score(n_cluster, models, df_test_scaled_pca_cluster, path_model_info):
    y_test_clusters = {}
    y_preds={}
    clusters = {f'cluster_{i}':df_test_scaled_pca_cluster.loc[df_test_scaled_pca_cluster.Cluster==i,:] for i in range(n_cluster)}

    for i in range(n_cluster):
        
        y_test_clusters[f'y_test_cluster_{i}'] = clusters[f'cluster_{i}']['target']
        model = models[list(models)[i]]

        y_preds[f'predict_cluster_{i}']=model.predict(clusters[f'cluster_{i}'].drop(columns=['target','Cluster']))

        #df con indices correctos para cada cluster y def y_test e y_pred
        df_y = pd.DataFrame()
        df_y["y_preds"] = pd.DataFrame(y_preds[f'predict_cluster_{i}'])[0]
        df_y["y_test"] = pd.DataFrame(y_test_clusters[f'y_test_cluster_{i}']).reset_index()["target"]
        y_pred = abs(df_y["y_preds"])
        y_test = abs(df_y["y_test"])

        print("*"*30, f"\nScore de \n\t {model.get_params()['estimator']} \n")

        r2 = round(r2_score(y_test, y_pred), 4)
        print(f"\tR2 score cluster_{i}:" , r2 )

        mae = round(mean_absolute_error(y_test, y_pred), 4)
        print(f"\tMAE score cluster_{i}:" , mae )

        mse = round(mean_squared_error(y_test, y_pred), 4)
        print(f"\tMSE score cluster_{i}:" , mse)

        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        print(f"\tRMSE score cluster_{i}:" , rmse)

        mape = round(mean_absolute_percentage_error(y_test, y_pred), 4)
        print(f"\tMAPE score cluster_{i}:" , mape)
        print("."*30)

        row=[model.get_params()['estimator'], f'cluster_{i}', model.best_params_, round(model.best_score_ , 4), r2, mae, mse, rmse, mape]
        guardar_en_fichero(path_model_info, row)


######################################################################################################################################################

def modeling2(model, parameters, df_train):
    '''
    Configuración de validación cruzada repetida de K-Fold y evaluar 
    y GridSearchCV para seleccionar de forma sistemática los parámetros (parameters) del modelo (model) deseado.

                Parámetros:
                        model: (Scikit-Learn model) estimator modelo a usar
                        parameters: (dict) diccionario conteniendo como "claves" los hiperparámetros del modelo, 
                                    y como "valores" los valores de los hiperparámetros que queremos probar
                        data: (pd.DataFrame) base de datos
                        data_set_name:(string) nombre descriptivo del dataset
                        target: (string) nombre de la columna target
                Returns:
                        mejores hiperparametros
                        mejor score
                        tiempo de entrenamiento y de predicción 
                        metricas del mejor modelo
                        escritura en fichero de información
                        modelo entrenado guardado
                        plot de curva de aprendizaje, grafico de error y residuos
    '''
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123)
    modelo_grid = GridSearchCV(estimator= model, 
                                param_grid = parameters,
                                scoring = "r2", 
                                cv = cv, n_jobs= -1, verbose=0)

    start = dt.datetime.now()

    model_fit = modelo_grid.fit(df_train.drop(columns="target"), df_train["target"])
    

    print("*"*30, f"\nEntrenando {model.__class__.__name__} \n", "*"*30)
    print('\t Best Hyperparameters: %s' % model_fit.best_params_)
    print('\t Best Score: %s' % model_fit.best_score_)
    
    train_time = (dt.datetime.now() - start).seconds
    print("\t Training time: %0.3fs" % train_time)
    with open(f"data/subdata/model_info/times.csv", 'a') as out:
            writer = csv.writer(out)
            writer.writerow([f"{model.__class__.__name__}", train_time])
            print("Model Info saved")

    with open(f"utils/pickle/subdata/models/{model.__class__.__name__}_entrenado.pkl", "wb") as file:
        pickle.dump(model_fit, file)
        
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    
    X_train = df_train.drop(['target'], 1)
    y_train = df_train[["target"]].squeeze()

    learningCurve = LearningCurve(modelo_grid.best_estimator_, scoring='r2',cv=cv)
    learningCurve.fit(X_train, y_train)
    learningCurve.finalize()
    plt.savefig(f'fig/learning_curve_subdata/{model.__class__.__name__}_LC.png', dpi=300)

    return print(f"\nModelo {model.__class__.__name__} Entrenado \n", "*"*30)


######################################################################################################################################################

def guardar_en_fichero2(ruta_target, row):
    '''
    Guarda en fichero csv datos de las metricas de los modelos entrenados
    '''
    #comprobacion que exista fichero / directorio
    try:
        if os.path.isfile(ruta_target):
            print ("File exist")
            with open(f'{ruta_target}', 'a') as out:
                #out.write(f"{content[0]} , {content[1]} ,{content[2]} \n")
                writer = csv.writer(out)
                writer.writerow(row)
        else:
            print ("File not exist")
            with open(f'{ruta_target}', 'w') as out:
                out.write("model_name , hyper_parametros, best_score, R2,  MAE, MSE, RMSE, MAPE\n")
                writer = csv.writer(out)
                writer.writerow(row)
                print("Model Info saved")
    except Exception as e:
            print(e)

######################################################################################################################################################

def score2(model, df_test, path_model_info):
    y_pred = pd.DataFrame(model.predict(df_test_scaled.drop(columns='target', axis=1)))
    y_test=df_test[['target']]

    print("*"*30, f"\nScore de \n\t {model.get_params()['estimator']} \n")

    r2 = round(r2_score(y_test, y_pred), 4)
    print(f"\tR2 score cluster_{i}:" , r2 )

    mae = round(mean_absolute_error(y_test, y_pred), 4)
    print(f"\tMAE score cluster_{i}:" , mae )

    mse = round(mean_squared_error(y_test, y_pred), 4)
    print(f"\tMSE score cluster_{i}:" , mse)

    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    print(f"\tRMSE score cluster_{i}:" , rmse)

    mape = round(mean_absolute_percentage_error(y_test, y_pred), 4)
    print(f"\tMAPE score cluster_{i}:" , mape)
    print("."*30)

    row=[model.get_params()['estimator'], model.best_params_, round(model.best_score_ , 4), r2, mae, mse, rmse, mape]
    guardar_en_fichero(path_model_info, row)