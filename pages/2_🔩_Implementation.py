import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Implementation", page_icon="🔩")
st.title("🔩Implementation")

df = pd.read_csv('steel_industry_energy_consumption.csv')
del df['Unnamed: 0']
df.columns = df.columns.str.lower().str.replace(".", "_")
categorical = list(df.dtypes[(df.dtypes == 'object')].index)
for col in categorical:
    df[col] = df[col].str.lower()
numerical = list(df.dtypes[(df.dtypes != 'object')].index)
numerical_f = numerical.copy()
numerical_f.remove('usage_kwh')
features = categorical + numerical_f
sns.set(font_scale=0.8)

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
    numerical = list(df.dtypes[(df.dtypes != 'object')].index)
    numerical = [col for col in numerical if col != 'usage_kwh']
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
    def prepare_X(data: pd.DataFrame) -> np.ndarray:
        X = data.copy()
        if 'object' in X.dtypes.values:
            for k, v in dicts_cat.items():
                for value in v:
                    X[f'{k}={value}'] = (X[k] == value).astype(int)
                del X[k]
        return X.values
    """, language='python'
    )

with st.expander('**Train data**'):
    st.write("Get the intercept and the coefficients value:")
    st.latex(
        r"""
        \begin{align*}
        \mathbf{w} &= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
        \end{align*}
        """
    )
    st.code(
        """
    def train_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        ones = np.ones(X.shape[0])
        X = np.column_stack((ones, X))
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv @ X.T @ y
        return w_full[0], w_full[1:]
    """, language='python'
    )

    st.write("Root mean squared error:")
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
        """
    )

