import pandas as pd
import numpy as np
df = pd.read_csv('/Users/annikaseo-yeonkim/Desktop/아무거나/kaggle/telecom_churn.csv')
print(df.head())
print(df.info())
df.describe()
print(df["Churn"].value_counts())
print(df["Total day calls"].mean())
# import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.countplot(x="International plan", hue="Churn", data=df))
sns.countplot(x="International plan", hue="Churn", data=df)
plt.show()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
import datetime # manipulating date formats
# Viz
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots


# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


# settings
import warnings
warnings.filterwarnings("ignore")

sales=pd.read_csv("/Users/annikaseo-yeonkim/Desktop/아무거나/sales data/sales_train.csv.zip")
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
sales.head
