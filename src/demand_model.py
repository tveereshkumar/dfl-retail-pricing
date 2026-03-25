from sklearn.ensemble import RandomForestRegressor

def train_models(df):
    models = {}

    for sku in df["sku"].unique():
        df_sku = df[df["sku"] == sku]

        X = df_sku[["price", "season", "promo"]]
        y = df_sku["demand"]

        model = RandomForestRegressor()
        model.fit(X, y)

        models[sku] = model

    return models