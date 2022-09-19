
# Selección de Modelo de Machine Learning  para descripción de Nanopartículas Metálicas: 🌕🌟🌕

# project_resume_1

* Carga datos 185 columnas
* tiro ID y time
* . dropna()
* del_zeros
* `data["E_atom"] = data["Total_E"]/data["N_total"]`
* `data.drop("Total_E", axis=1)`
* `train, test = train_test_split(df, test_size=0.3, random_state=1234)`
* reset index!

## TRAIN
* Escalado
    * scaler=StandardScaler()
    * x_train=train.drop(["E_atom"], axis=1)
    * x_train_scaled = scaler.transform(x_train)
    * **utils/scaler.pkl**
* PCA
    * pca.transform(x_train_scaled)
    * df_train_scaled_pca
    * **utils/pca.pkl**
* KMeans
    * cluster_label = knn_3.predict(x_train_scaled_pca)
    * 
    * df_train_scaled_pca_cluster ["Cluster"]= knn_3.labels_
    * knn_3.predict(x_train_scaled_pca)
    * **utils/kmeans3.pk**
* df_train_scaled_pca_cluster['target']
* Modelos 
* columnas 
    * **utils/columnas.pkl**

## TEST
* Se carga
    * scaler -> `.transform(x_test)`
    * pca -> `.transform(x_test_scaled)`
    * kmeans -> `.predict(x_test_scaled_pca)`
    * df_test_scaled_pca_cluster 
        * columnas -> `pd.DataFrame(x_test_scaled_pca, columns=columnas[:-2])`
        * ["Cluster"]
        * ["target"]
    * modelado
        * n_cluster
        * y_preds={}
        * clusters
        * y_preds -> modelos_entrenados[f'Modelo_{i}'].predict(clusters[f'cluster_{i}']
        * y_test_clusters


