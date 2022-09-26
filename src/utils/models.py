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