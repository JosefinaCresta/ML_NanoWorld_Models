
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


parameters_linear_regression = {'copy_X' : [True, False],
                                'fit_intercept' : [True, False]
                                }


parameters_Ridge = {'alpha': [0.1, 0.5, 1.0] ,
                    'fit_intercept': [True, False],
                    'max_iter': [100,1000],
                    'solver': ['auto', 'svd', 'cholesky'],
                    'random_state' : [0 , 1]
                    }


parameters_KernelRidge = {'kernel': ['polynomial', 'rbf'],
                        'alpha': [0.1, 0.5, 1 ] ,
                        'gamma': [0.01, 0.1] ,
                        'degree': [3, 2, 1] 
                        }


parameters_Lasso = {'alpha': [0.01, 1.0],
                    'copy_X': [True,False],
                    'fit_intercept': [True,False],
                    'max_iter': [100, 500, 1000],
                    'positive':[True,False],
                    'precompute': [True,False],
                    'selection': ['cyclic', 'random'],
                    'tol': [0.001, 0.01, 1],
                    'warm_start': [True,False]
                    }

parameters_ElasticNet = {'alpha': [0.01, 1.0],
                        'copy_X': [True,False],
                        'fit_intercept': [True,False],
                        'l1_ratio': [0, 0.5],
                        'max_iter': [100, 500, 1000],
                        'precompute': [True,False],
                        'random_state': [None, 0],
                        'tol': [0.0001, 0.01]
                        }  

parameters_DecisionTreeRegressor = {"splitter":["best","random"],
                                    "max_depth" : [1,3,5,7,9,11,12],
                                    "min_samples_leaf":[1,2,7,8,9,10],
                                    "min_weight_fraction_leaf":[0.1,0.2,0.5,0.7,0.8],
                                    "max_features":["auto","log2","sqrt",None],
                                    "max_leaf_nodes":[None,10,20,40,50,60] }
                        

parameters_AdaBoostRegressor = {'learning_rate': [0.01, 0.1, 1.0],
                                'loss': ['linear', 'exponential', 'square'],
                                'n_estimators': [50, 100, 500, 1000],
                                } 
 


parameters_GradientBoostingRegressor = {'alpha': [0.1, 0.9, 1],
                                        'learning_rate': [0.05, 0.1],
                                        'loss': ['squared_error','huber'],
                                        'max_depth': [1, 3],
                                        'min_samples_split': [2, 3],
                                        'n_estimators': [100, 500],
                                    }   

parameters_RandomForestRegressor = {'bootstrap': [True,False],
                                    'n_estimators': [10,50,100,500],
                                    'max_depth':[None ,10, 50, 100 ],               
                                    'min_samples_leaf': [1, 2],
                                    'random_state': [None, 0],
                                    }


parameters_ExtraTrees = {'ccp_alpha' : [0.0 ,0.1, 0.5],
                        'min_samples_leaf': [1, 2],
                        'min_samples_split' : [2, 3, 5]
                        }      

parameters_KNeighborsRegressor = {'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                                'leaf_size': [2, 5, 30],
                                'n_neighbors': [5, 6, 10, 14]
                                }                    
                      

params_BaggingRegressor = {'n_estimators' : [10, 50,100],
                'max_features' :[1,2,4,6,8],
                'max_samples' : [0.5,0.1, 1.0],
                'bootstrap' : [True, False]
                }

parameters_SVR = {'C': [0.1, 1.0],
                'degree': [1, 3],
                'gamma': ['scale',10.0, 100.0],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'shrinking': [True,False]
                }   

##########################################################################
models = { 
"LinearRegression" :  [LinearRegression(), parameters_linear_regression],

"Ridge" : [Ridge(), parameters_Ridge],

"Lasso" : [Lasso(), parameters_Lasso],

"ElasticNet" : [ElasticNet(), parameters_ElasticNet],

"KernelRidge" : [KernelRidge(), parameters_KernelRidge],

"DecisionTreeRegressor" : [DecisionTreeRegressor(), parameters_DecisionTreeRegressor],

"ExtraTreeRegressor" : [ExtraTreeRegressor(), parameters_ExtraTrees],

"RandomForestRegressor" : [RandomForestRegressor(), parameters_RandomForestRegressor],

"KNeighborsRegressor" : [KNeighborsRegressor(), parameters_KNeighborsRegressor], 

"GradientBoostingRegressor" : [GradientBoostingRegressor(), parameters_GradientBoostingRegressor],

"AdaBoostRegressor" : [AdaBoostRegressor(), parameters_AdaBoostRegressor],

"BaggingRegressor" : [BaggingRegressor(), params_BaggingRegressor],

"SVR" : [SVR(), parameters_SVR]

}