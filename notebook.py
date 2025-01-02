import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium", layout_file="layouts/notebook.slides.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Load Data & Import Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt

    from IPython.display import display

    from typing import List, Tuple
    import pickle
    return List, Tuple, display, mo, np, pd, pickle, plt, sns


@app.cell
def _(pd):
    # !pip install ucimlrepo
    # from ucimlrepo import fetch_ucirepo

    # fetch dataset
    # steel_industry_energy_consumption = fetch_ucirepo(id=851)

    # data (as pandas dataframes)
    # X = steel_industry_energy_consumption.data.features
    # y = steel_industry_energy_consumption.data.targets

    # df = X.copy()
    # df['Load_Type'] = y.copy()
    # df.to_csv('steel_industry_energy_consumption.csv')

    df = pd.read_csv('steel_industry_energy_consumption.csv')
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""### Data Preparation""")
    return


@app.cell
def _(df):
    df.head().T
    return


@app.cell
def _(df):
    del df['Unnamed: 0']
    return


@app.cell
def _(df):
    df.describe(include='all').T
    return


@app.cell
def _(df):
    # Convert columns name to lowercase and replace period to underscore
    df.columns = df.columns.str.lower().str.replace(".", "_")

    del df['weekstatus']

    # Get categorical columns name
    categorical = list(df.dtypes[(df.dtypes == 'object')].index)

    # Convert columns value to lowercase
    for col in categorical:
        df[col] = df[col].str.lower()

    # Get numerical columns name
    nums = list(df.dtypes[(df.dtypes != 'object')].index)



    nums, categorical
    return categorical, col, nums


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Exploratory Data Analyis""")
    return


@app.cell
def _(df, nums):
    df[nums].describe().T
    return


@app.cell
def _(categorical, df):
    df[categorical].describe().T
    return


@app.cell
def _(categorical, col, df):
    for column in categorical:
        print(f'{column}: {set(df[col])}')
        print()
    return (column,)


@app.cell
def _(categorical, col, df, display):
    for c in categorical:
        display(df[col].value_counts(normalize=True))
        print()
    return (c,)


@app.cell
def _(df, nums, plt, sns):
    sns.set(font_scale=0.8)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df[nums].corr(), annot=True)
    return


@app.cell
def _(categorical, df, plt, sns):
    for i, cat_f in enumerate(categorical):
        plt.figure(figsize=(8, 8))
        plt.subplot(3, 1, i+1)
        sns.violinplot(data=df, x=cat_f, y="usage_kwh", hue=cat_f)
        plt.xlabel(cat_f)
        plt.ylabel("Usage (kWh)")
        plt.tight_layout()
        plt.show()
    return cat_f, i


@app.cell
def _(df):
    df.head().T
    return


@app.cell
def _(df, sns):
    sns.histplot(x=df['usage_kwh'], bins=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Validation Framework""")
    return


@app.cell
def _(df, np):
    np.random.seed(8)

    n = df.shape[0]
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.shuffle(idx)

    data_copy = df.copy()
    df_train = data_copy.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = data_copy.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = data_copy.iloc[idx[n_train+n_val:]].reset_index(drop=True)

    y_train = df_train['usage_kwh'].values
    y_val = df_val['usage_kwh'].values
    y_test = df_test['usage_kwh'].values

    del df_train['usage_kwh']
    del df_val['usage_kwh']
    del df_test['usage_kwh']
    return (
        data_copy,
        df_test,
        df_train,
        df_val,
        idx,
        n,
        n_test,
        n_train,
        n_val,
        y_test,
        y_train,
        y_val,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Feature Engineering""")
    return


@app.cell
def _(nums):
    numerical = [_col for _col in nums if _col != 'usage_kwh']
    numerical
    return (numerical,)


@app.cell
def _(categorical, df):
    dicts_cat = {}
    for _col in categorical:
        dicts_cat[_col] = df[_col].value_counts().sort_values().index.to_list()
    return (dicts_cat,)


@app.cell
def _(dicts_cat, np, pd):
    def prepare_X(data: pd.DataFrame) -> np.ndarray:
        X = data.copy()

        if 'object' in X.dtypes.values:
            for k, v in dicts_cat.items():
                for value in v:
                    X[f'{k}={value}'] = (X[k] == value).astype(int)
                del X[k]

        return X.values
    return (prepare_X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Train Data""")
    return


@app.cell
def _(Tuple, np):
    def train_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X))
        XTX = np.dot(X.T, X)
        # XTX = XTX + 0.001 * np.eye(XTX.shape[0])
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv @ X.T @ y
        return w_full[0], w_full[1:]
    return (train_linear_regression,)


@app.cell
def _(np):
    def rmse(y_act, y_pred):
        se = (y_act - y_pred) ** 2
        mse = se.mean()
        return np.sqrt(mse)
    return (rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### *Train using only numerical features*""")
    return


@app.cell
def _(
    df_train,
    df_val,
    np,
    numerical,
    plt,
    prepare_X,
    rmse,
    sns,
    train_linear_regression,
    y_train,
    y_val,
):
    X_train = prepare_X(df_train[numerical])
    _intercept, _coef = train_linear_regression(X_train, y_train)
    y_train_pred = _intercept + np.dot(X_train, _coef)

    X_val = prepare_X(df_val[numerical])
    y_val_pred = _intercept + X_val.dot(_coef)

    print('intercept')
    print(f'{_intercept}')
    print()
    print('coef')
    print(f'{_coef}')
    print()
    print(f'rmse: {rmse(y_val, y_val_pred)}')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=y_val_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    plt.show()
    return X_train, X_val, y_train_pred, y_val_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md("""#### *Train using numerical + categorical features*""")
    return


@app.cell
def _(
    df_train,
    df_val,
    np,
    plt,
    prepare_X,
    rmse,
    sns,
    train_linear_regression,
    y_train,
    y_val,
):
    _X_train = prepare_X(df_train)

    _intercept, _coef = train_linear_regression(_X_train, y_train)
    _y_train_pred = _intercept + np.dot(_X_train, _coef)

    _X_val = prepare_X(df_val)
    _y_val_pred = _intercept + _X_val.dot(_coef)

    print('intercept')
    print(f'{_intercept}')
    print()
    print('coef')
    print(f'{_coef}')
    print()
    print(f'rmse: {rmse(y_val, _y_val_pred)}')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=_y_val_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    plt.show()


    # with open('lr_model_full_features.pickle', 'wb') as f:
    #    pickle.dump((intercept, coef), f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Regularization""")
    return


@app.cell
def _(Tuple, np):
    # training function with regularization
    def train_linear_regression_reg(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X))
        XTX = np.dot(X.T, X)
        XTX = XTX + alpha * np.eye(XTX.shape[0])
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv @ X.T @ y

        return w_full[0], w_full[1:]
    return (train_linear_regression_reg,)


@app.cell
def _(
    df_train,
    df_val,
    pd,
    prepare_X,
    rmse,
    train_linear_regression_reg,
    y_train,
    y_val,
):
    alphas = [0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    rmses = []

    for alpha in alphas:
        _X_train = prepare_X(df_train)
        _intercept, _coef = train_linear_regression_reg(_X_train, y_train, alpha)
        
        _X_val = prepare_X(df_val)
        _y_val_pred = _intercept + _X_val.dot(_coef)

        rmses.append(rmse(y_val, _y_val_pred))
        
        # print('%06s %f' % (alpha, rmse(y_val, y_val_pred)))
     
    alpha_df = pd.DataFrame()
    alpha_df['alpha_values'] = alphas
    alpha_df['rmses'] = rmses
    alpha_df = alpha_df.sort_values(by=['rmses']).reset_index(drop=True)

    alpha_df


    # with open('alpha_df.pickle', 'wb') as f:
    #    pickle.dump(alpha_df, f)
    return alpha, alpha_df, alphas, rmses


@app.cell
def _(
    df_train,
    df_val,
    np,
    plt,
    prepare_X,
    rmse,
    sns,
    train_linear_regression_reg,
    y_train,
    y_val,
    y_val_pred,
):
    _X_train = prepare_X(df_train)

    _intercept, _coef = train_linear_regression_reg(_X_train, y_train, 0.01)
    _y_train_pred = _intercept + np.dot(_X_train, _coef)

    _X_val = prepare_X(df_val)
    _y_val_pred = _intercept + _X_val.dot(_coef)

    print('intercept')
    print(f'{_intercept}')
    print()
    print('coef')
    print(f'{_coef}')
    print()
    print(f'rmse: {rmse(y_val, y_val_pred)}')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=y_val_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    plt.show()

    # with open('lr_model_regularized.pickle', 'wb') as f:
    #    pickle.dump((intercept, coef), f)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Using Model""")
    return


@app.cell
def _(
    df_test,
    df_train,
    df_val,
    np,
    pd,
    plt,
    prepare_X,
    rmse,
    sns,
    train_linear_regression_reg,
    y_test,
    y_train,
    y_val,
):
    df_train_full = pd.concat([df_train, df_val])
    df_train_full = df_train_full.reset_index(drop=True)

    y_train_full = np.concat((y_train, y_val))


    X_train_full = prepare_X(df_train_full)

    f_intercept, f_coef = train_linear_regression_reg(X_train_full, y_train_full, 0.01)

    _X_test = prepare_X(df_test)
    _y_test_pred = f_intercept + _X_test.dot(f_coef)

    print('intercept')
    print(f'{f_intercept}')
    print()
    print('coef')
    print(f'{f_coef}')
    print()
    print(f'rmse: {rmse(y_test, _y_test_pred)}')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=_y_test_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.show()

    # with open('lr_model_regularized_full.pickle', 'wb') as f:
    #    pickle.dump((intercept, coef), f)
    return X_train_full, df_train_full, f_coef, f_intercept, y_train_full


@app.cell
def _(df_test, f_coef, f_intercept, prepare_X, y_test):
    df_small = df_test.iloc[26:27]
    X_small = prepare_X(df_test.iloc[26:27])
    y_small = y_test[26]
    y_small_test = round(f_intercept + X_small.dot(f_coef)[0], 2)

    X_small, y_small, y_small_test
    return X_small, df_small, y_small, y_small_test


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
