import pandas as pd
import pydot   
from IPython.display import Image

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz

from copy import deepcopy

## classification importance with and ExtraTreesClassifier
def classification_feature_importance(X,
                                      y,
                                      n_estimators=100, 
                                      criterion='gini',
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.,
                                      max_features='auto', 
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.,
                                      min_impurity_split=1e-7, 
                                      bootstrap=True,
                                      oob_score=True,
                                      n_jobs=1, 
                                      random_state=0,
                                      verbose=0,
                                      class_weight=None):
    
    classifier = ExtraTreesClassifier(n_estimators=n_estimators, 
                                      criterion=criterion,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf,
                                      max_features=max_features, 
                                      max_leaf_nodes=max_leaf_nodes,
                                      min_impurity_decrease=min_impurity_decrease,
                                      min_impurity_split=min_impurity_split, 
                                      bootstrap=bootstrap,
                                      oob_score=oob_score,
                                      n_jobs=n_jobs, 
                                      random_state=random_state,
                                      verbose=verbose,
                                      class_weight=class_weight)
    classifier.fit(X,y)
    df_importance = pd.DataFrame({'feature':list(X.columns),
                                  'importance':list(classifier.feature_importances_)})
    df_importance.sort_values('importance',inplace=True,ascending=False)
    return df_importance



## plot a tree
def plot_tree(estimator,features_names):
    dot_data = StringIO()  
    export_graphviz(estimator, 
                    out_file=dot_data,
                    feature_names=features_names,
                    filled=True,rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph[0].create_png())  

## find correlations

def take_most_correlated(df,correlation_threshold = 0.9):
    aux = df.corr()[(df.corr()>=correlation_threshold)].reset_index().melt('index').dropna()
    aux = aux[aux['index'] != aux['variable']]
    aux['var_left'],aux['var_right'] = aux.apply(lambda r:sorted([r['index'],r['variable']]),axis=1).str
    aux = aux.loc[:,['var_left','var_right','value']].drop_duplicates()
    var_list = list(aux['var_left'].values) + list(aux['var_right'].values)
    return pd.Series(var_list).value_counts(normalize=True)



## map encoder
def map_encoder_categorical(x):
    set_ = sorted(set(x))
    range_ = list(range(1,len(set_)+1))
    return dict(zip(set_,range_))

def map_encoder_fit(df,categorical_features):
    return {i:map_encoder_categorical(df[i]) for i in categorical_features}

def map_encoder_transform(df,map_encoder):
    df_ = deepcopy(df) 
    for i in map_encoder.keys():
        map_encoder_variable = map_encoder[i]
        fill_ = int(max(map_encoder_variable.values()) + 1)
        df_[i] = df_[i].map(map_encoder_variable).fillna(fill_).astype(int)
    return df_

## onehot encoder
def onehot_encoder_fit(df,columns):
    dict_ = {}
    dict_['categorical_columns'] = columns
    dict_['output_columns'] = list(pd.get_dummies(data=df,columns=columns).columns)
    return dict_

def onehot_encoder_transform(df,onehot_encoder):
    df_ = deepcopy(df)
    df_ = pd.get_dummies(data=df_,columns=onehot_encoder['categorical_columns'])
    df_ = df_.loc[:,onehot_encoder['output_columns']].fillna(0)
    return df_
