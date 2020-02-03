def transform(df,label, map_encoder,onehot_encoder):
    df_ = deepcopy(df)
    df_.columns = [i.lower() for i in df_.columns]
    if 'customerid' in df.columns:
        df_ = df_.drop('customerid',axis=1)
    features = [i for i in df_.columns if i!=label]
    X,y = df_[features],df_[label]
    X_ = map_encoder_transform(X,map_encoder)
    X_ = onehot_encoder_transform(X_,onehot_encoder)
    return (X_,y)