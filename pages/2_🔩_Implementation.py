import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


st.set_page_config(page_title="Implementation", page_icon="🔩")
st.title("🔩Implementation")



df = pd.read_csv('steel_industry_energy_consumption.csv')
del df['Unnamed: 0']
df.columns = df.columns.str.lower().str.replace(".", "_")
del df['weekstatus']
categorical = list(df.dtypes[(df.dtypes == 'object')].index)
for col in categorical:
    df[col] = df[col].str.lower()
numerical = list(df.dtypes[(df.dtypes != 'object')].index)
numerical.remove('usage_kwh')
features = categorical + numerical
sns.set(font_scale=0.6)


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


numerical = [col for col in numerical if col != 'usage_kwh']
dicts_cat = {}
for col in categorical:
    dicts_cat[col] = df[col].value_counts().sort_values().index.to_list()

def prepare_X(data):
    X = data.copy()

    if 'object' in X.dtypes.values:
        for k, v in dicts_cat.items():
            for value in v:
                X[f'{k}_{value}'] = (X[k] == value).astype(int)
            del X[k]

    return X.values


def rmse(y_act, y_pred):
    se = (y_act - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


st.write("")
with st.expander('**Load libraries**'):
    st.code(
        """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    """, language='python'
    )

with st.expander('**Data preprocessing**'):
    st.code(
        """
    df = pd.read_csv('steel_industry_energy_consumption.csv')
    del df['Unnamed: 0']
    df.columns = df.columns.str.lower().str.replace(".", "_")
    categorical = list(df.dtypes[(df.dtypes == 'object')].index)
    for col in categorical:
        df[col] = df[col].str.lower()
    """, language='python'
    )


with st.expander('**Validation framework**'):
    st.code(
        """
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
    """, language='python'
    )

with st.expander('**Feature engineering**'):
    st.write("Dictionary of categorical features's name as keys and the unique value as it's items")
    st.code(
        """
    dicts_cat = {}
    for col in categorical:
        dicts_cat[col] = df[col].value_counts().sort_values().index.to_list()
    """, language='python'
    )

    st.write("One-hot encode the categorical features")
    st.code(
        """
    def prepare_X(data):
        X = data.copy()
        if 'object' in X.dtypes.values:
            for k, v in dicts_cat.items():
                for value in v:
                    X[f'{k}_{value}'] = (X[k] == value).astype(int)
                del X[k]
        return X.values
    """, language='python'
    )

with st.expander('**Train and evaluate the model using numerical features**'):
    st.markdown("* The train function to get the intercept (w0) and the coefficients (w) values:")
    st.latex(
        r"""
        \begin{align*}
        \mathbf{w} &= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
        \end{align*}
        """
    )
    st.code(
        """
    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X))
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv @ X.T @ y
        return w_full[0], w_full[1:]
    """, language='python'
    )

    st.markdown("* Calculating equation to get predicted values:")
    st.latex(
        r"""
        \begin{align*}
        \hat{y} = \mathbf{w0} + \mathbf{X} \cdot \mathbf{w}
        \end{align*}
        """
    )

    st.code(
        """
        y_pred = w0 + X @ w

    """, language='python'
    )

    st.markdown("* Root mean squared error for evaluating the model:")
    st.latex(
        r"""
        \begin{align*}
        RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
        \end{align*}
        """
    )
    st.code(
        """
    def rmse(y_act, y_pred):
        se = (y_act - y_pred) ** 2
        mse = se.mean()
        return np.sqrt(mse)
    """, language='python'
    )

    st.markdown("* Train and evaluate the model using train and validation dataset:")
    st.code(
        """
        X_train = prepare_X(df_train[numerical])
        intercept, coef = train_linear_regression(X_train, y_train)

        X_val = prepare_X(df_val[numerical])
        y_val_pred = intercept + X_val.dot(coef)

        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=y_val, y=y_val_pred, ax=ax1)
        plt.xlabel(f'y_val (actual)')
        plt.ylabel('y_val_pred (predicted)')
        plt.title('Actual vs Predicted Values')
        plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
        st.pyplot(fig1)

        print(f'rmse: {rmse(y_val, y_val_pred)}')
    """, language='python'
    )

    # Model 1
    with open('lr_model_num_features.pickle', 'rb') as f:
        intercept_m1, coef_m1 = pickle.load(f)

    X_train = prepare_X(df_train[numerical])
    X_val = prepare_X(df_val[numerical])
    y_val_pred = intercept_m1 + X_val.dot(coef_m1)

    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_val, y=y_val_pred, ax=ax1)
    plt.xlabel(f'y_val (actual)')
    plt.ylabel('y_val_pred (predicted)')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    
    st.pyplot(fig1)
    st.info(f"RMSE : {rmse(y_val, y_val_pred)}")


with st.expander('**Train and evaluate the model using all features**'):
    st.markdown("Use the same calculation as above:")
    st.code(
        """    
        X_train = prepare_X(df_train)
        intercept, coef = train_linear_regression(X_train, y_train)

        X_val = prepare_X(df_val)
        y_val_pred = intercept + X_val.dot(coef)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=y_val, y=y_val_pred, ax=ax2)
        plt.xlabel(f'y_val (actual)')
        plt.ylabel('y_val_pred (predicted)')
        plt.title('Actual vs Predicted Values')
        plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
        st.pyplot(fig2)

        print(f'rmse: {rmse(y_val, y_val_pred)}')
    """, language='python'
    )

    with open('lr_model_full_features.pickle', 'rb') as f:
        intercept_m2, coef_m2 = pickle.load(f)

    X_train = prepare_X(df_train)
    X_val = prepare_X(df_val)
    y_val_pred = intercept_m2 + X_val.dot(coef_m2)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=y_val, y=y_val_pred, ax=ax2)
    plt.xlabel(f'y_val (actual)')
    plt.ylabel('y_val_pred (predicted)')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
    
    st.pyplot(fig2)
    st.info(f"RMSE : {rmse(y_val, y_val_pred)}")
    st.info("Next step: fix the large RMSE issue by adjusting model calculation with regularization.")

with st.expander('**Train and evaluate the regularized model**'):
    st.markdown("* Modified version of the model training calculation:")
    st.latex(
        r"""
        \begin{align*}
        \mathbf{w} &= (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
        \end{align*}
        """
    )
    st.code(
        """
        def train_linear_regression_reg(X, y, alpha):
            ones = np.ones(X.shape[0])
            X = np.column_stack((ones, X))
            XTX = np.dot(X.T, X)
            XTX = XTX + alpha * np.eye(XTX.shape[0])
            XTX_inv = np.linalg.inv(XTX)
            w_full = XTX_inv @ X.T @ y
            return w_full[0], w_full[1:]
    """, language='python'
    )

    st.markdown("* Find the best alpha value (lowest RMSE):")
    st.code(
        """
        alphas = [0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]
        rmses = []

        for alpha in alphas:
            X_train = prepare_X(df_train)
            intercept, coef = train_linear_regression_reg(X_train, y_train, alpha)
            
            X_val = prepare_X(df_val)
            y_val_pred = intercept + X_val.dot(coef)

            rmses.append(rmse(y_val, y_val_pred))

        alpha_df = pd.DataFrame()
        alpha_df['alpha_values'] = alphas
        alpha_df['rmses'] = rmses
        alpha_df = alpha_df.sort_values(by=['rmses']).reset_index(drop=True)
    """, language='python'
    )

    with open('alpha_df.pickle', 'rb') as f:
        alpha_df = pickle.load(f)

    st.dataframe(alpha_df)
    st.info("The alpha value with lowest RMSE is 0.01")


    st.markdown("* Train and evaluate the regularized model:")
    st.code(
        """
        X_train = prepare_X(df_train)
        intercept, coef = train_linear_regression_reg(X_train, y_train, 0.01)
        X_val = prepare_X(df_val)
        y_val_pred = intercept + X_val.dot(coef)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=y_val, y=y_val_pred, ax=ax3)
        plt.xlabel(f'y_val (actual)')
        plt.ylabel('y_val_pred (predicted)')
        plt.title('Actual vs Predicted Values')
        plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
        
        st.pyplot(fig3)
        st.info(f"RMSE : {rmse(y_val, y_val_pred)}")
    """, language='python'
    )
    
    with open('lr_model_regularized.pickle', 'rb') as f:
        intercept_m3, coef_m3 = pickle.load(f)
    
    X_train = prepare_X(df_train)
    X_val = prepare_X(df_val)
    y_val_pred = intercept_m3 + X_val.dot(coef_m3)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=y_val, y=y_val_pred, ax=ax3)
    plt.xlabel(f'y_val (actual)')
    plt.ylabel('y_val_pred (predicted)')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')

    st.pyplot(fig3)
    st.info(f"RMSE : {rmse(y_val, y_val_pred)}")

with st.expander("**Train and evaluate the regularized model using train + validation and test dataset**"):
    st.markdown("The final model that used to predict electricity usage in the project:")
    st.code(
        """
        df_train_full = pd.concat([df_train, df_val])
        df_train_full = df_train_full.reset_index(drop=True)
        y_train_full = np.concat((y_train, y_val))
        
        X_train_full = prepare_X(df_train_full)
        intercept, coef = train_linear_regression_reg(X_train_full, y_train_full, 0.01)
        
        X_test = prepare_X(df_test)
        y_test_pred = intercept + X_test.dot(coef)

        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=y_test, y=y_test_pred, ax=ax4)
        plt.xlabel(f'y_test (actual)')
        plt.ylabel('y_test_pred (predicted)')
        plt.title('Actual vs Predicted Values')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

        st.pyplot(fig4)
        st.info(f"RMSE : {rmse(y_test, y_test_pred)}")
    """
    )

    with open('lr_model_regularized_full.pickle', 'rb') as f:
        intercept_m4, coef_m4 = pickle.load(f)
    
    X_test = prepare_X(df_test)
    y_test_pred = intercept_m4 + X_test.dot(coef_m4)

    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_test_pred, ax=ax4)
    plt.xlabel(f'y_test (actual)')
    plt.ylabel('y_test_pred (predicted)')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')

    st.pyplot(fig4)
    st.info(f"RMSE : {rmse(y_test, y_test_pred)}")