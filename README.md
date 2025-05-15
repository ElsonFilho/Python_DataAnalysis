# Python Data Analysis
This repository provides examples and best practices for understanding data and utilizing Python libraries to import, explore, analyze, develop models, and evaluate them effectively. Each section below outlines key concepts and links to illustrative Jupyter Notebooks.

## 1. Importing Data Sets

This section focuses on how to bring data into your Python environment from various sources using popular libraries like pandas.

Learn how to import data from different file formats and sources, handle basic loading parameters, and perform initial inspections of the imported data.

**Key Operations and Code Examples:**
| Package/Method | Description | Code Example |
|---|---|---|
| `pd.read_csv()` | Read a CSV file into a pandas DataFrame. | `df = pd.read_csv(<CSV_path>, header = None)  # load without header` <br> `df = pd.read_csv(<CSV_path>, header = 0)   # load using first row as header` <br> **Note:** In JupyterLite, download the file locally and use the local path. In other environments, you can use the URL directly. |
| `df.head(n)` | Print the first few entries of the DataFrame (default 5). | `df.head(n)  # n = number of entries; default is 5` |
| `df.tail(n)` | Print the last few entries of the DataFrame (default 5). | `df.tail(n)  # n = number of entries; default is 5` |
| `df.columns = headers` | Assign appropriate header names to the DataFrame. | `df.columns = headers` |
| `df.replace("?", np.nan)` | Replace specific values (e.g., "?") with NumPy's `NaN` (Not a Number) for handling missing data. | `df = df.replace("?", np.nan)` |
| `df.dtypes` | Retrieve the data types of each column in the DataFrame. | `df.dtypes` |
| `df.describe()` | Generate descriptive statistics of the DataFrame. By default, it analyzes numerical columns. Use `include="all"` to include all data types. | `df.describe()` <br> `df.describe(include="all")` |
| `df.info()` | Provide a concise summary of the DataFrame, including data types, non-null values, and memory usage. | `df.info()` |
| `df.to_csv(<output CSV path>)` | Save the processed DataFrame to a CSV file at the specified path. | `df.to_csv(<output CSV path>)` |

**Example Notebook:**

* [notebooks/01_Importing_Data.ipynb](notebooks/01_Importing_Data.ipynb)

## 2. Data Wrangling

This section covers the essential steps involved in cleaning, transforming, and structuring your data for effective analysis.<BR>
Perform some fundamental data wrangling tasks that form the pre-processing phase of data analysis. These tasks include handling missing values in data, formatting data to standardize it and make it consistent, normalizing data, grouping data values into bins, and converting categorical variables into numerical quantitative variables to ensure data quality and consistency.


| Package/Method                 | Description                                                                                                | Code Example                                                                                                |
|--------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Replace Missing Data with Frequency** | Replaces missing values (NaN) in a specified column with the most frequently occurring value (mode). | MostFrequentEntry = df['attribute_name'].value_counts().idxmax() <br> df['attribute_name'].replace(np.nan, MostFrequentEntry, inplace=True)                                      |
| **Replace Missing Data with Mean** | Replaces missing values (NaN) in a specified column with the average (mean) of all non-missing values. | AverageValue = df['attribute_name'].astype(<data_type>).mean(axis=0) df['attribute_name'].replace(np.nan, AverageValue, inplace=True)                                        |
| **Fix Data Types** | Changes the data type of one or more columns in the DataFrame.                                            | df[['attribute1_name', 'attribute2_name', ...]] = df[['attribute1_name', 'attribute2_name', ...]].astype('data_type') <br> # data_type can be int, float, str, etc.                                                                     |
| **Data Normalization** | Scales the values in a specified column to a range between 0 and 1.                                      | df['attribute_name'] = df['attribute_name'] / df['attribute_name'].max()                                       |
| **Binning** | Groups data in a specified column into discrete intervals (bins) for analysis and visualization.         | bins = np.linspace(min(df['attribute_name']), max(df['attribute_name']), n) <br>  # n is the desired number of bins <br> GroupNames = ['Group1', 'Group2', 'Group3', ...] <br> df['binned_attribute_name'] = pd.cut(df['attribute_name'], bins, labels=GroupNames, include_lowest=True)     |
| **Change Column Name** | Renames a specified column in the DataFrame.                                                              | df.rename(columns={'old_name': 'new_name'}, inplace=True)                                                     |
| **Indicator Variables** | Creates new binary (0 or 1) columns for each unique category in a specified categorical column.           | dummy_variable = pd.get_dummies(df['attribute_name'])  <br> df = pd.concat([df, dummy_variable], axis=1)                                                                  |

**Example Notebook:**

* [notebooks/02_Data_Wrangling.ipynb](notebooks/02_Data_Wrangling.ipynb) 

## 3. Exploratory Data Analysis (EDA)

This section focuses on techniques for visually and statistically exploring your data to uncover patterns, relationships, and insights.

Discover how to use various plotting libraries (like Matplotlib and Seaborn) and statistical methods to understand the distribution of variables, identify correlations, and detect potential outliers.

| Package/Method                      | Description                                                                                                                                       | Code Example |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **Complete dataframe correlation** | Correlation matrix created using all the attributes of the dataset.                                                                               | `df.corr()` |
| **Specific Attribute correlation** | Correlation matrix created using specific attributes of the dataset.                                                                             | `df[['attribute1','attribute2',...]].corr()` |
| **Scatter Plot**                   | Create a scatter plot using the data points of the dependent variable along the x-axis and the independent variable along the y-axis.             |`from matplotlib import pyplot as plt plt.scatter(df[['attribute_1']], df[['attribute_2']])`|
| **Regression Plot**               | Uses the dependent and independent variables in a Pandas data frame to create a scatter plot with a generated linear regression line for the data. | `sns.regplot(x='attribute_1', y='attribute_2', data=df)` |
| **Box plot**                       | Create a box-and-whisker plot that uses the pandas dataframe, the dependent, and the independent variables.                                       | `sns.boxplot(x='attribute_1', y='attribute_2', data=df)` |
| **Grouping by attributes**         | Create a group of different attributes of a dataset to create a subset of the data.                                                               | `df_group = df[['attribute_1','attribute_2',...]]` |
| **GroupBy statements**             | a. Group the data by different categories of an attribute, displaying the average value of numerical attributes with the same category. <br> b. Group the data by different categories of multiple attributes, displaying the average value of numerical attributes with the same category. | `df_group = df.groupby(['attribute_1'], as_index=False).mean() df_group = df.groupby(['attribute_1','attribute_2'], as_index=False).mean()` |
| **Pivot Tables**                   | Create Pivot tables for better representation of data based on parameters.                                                                       | `grouped_pivot = df_group.pivot(index='attribute_1', columns='attribute_2')` |
| **Pseudocolor plot**               | Create a heatmap image using a PseudoColor plot (or pcolor) using the pivot table as data.                                                        | `from matplotlib import pyplot as plt<br>plt.pcolor(grouped_pivot, cmap='RdBu')` |
| **Pearson Coefficient and p-value**| Calculate the Pearson Coefficient and p-value of a pair of attributes.                                                                            | `from scipy import stats<br>pearson_coef, p_value = stats.pearsonr(df['attribute_1'], df['attribute_2'])` |


**Example Notebook:**

* [notebooks/03_Exploratory_Data_Analysis.ipynb](notebooks/03_Exploratory_Data_Analysis.ipynb)


## 4. Model Development

This section introduces the process of building predictive models using Python's powerful machine learning libraries (like scikit-learn).

Learn the fundamental steps in model development, including selecting appropriate algorithms, training models on your data, and making predictions.
| **Process**                          | **Description**                                                                                                                                                                                                                                                        | **Code Example**                                                                                                                                                                                                                                      |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linear Regression                   | Create a Linear Regression model object.                                                                                                                                                                                                                               | from sklearn.linear_model import LinearRegression  lr = LinearRegression()                                                                                                                                                                           |
| Train Linear Regression model       | Train the Linear Regression model on decided data, separating Input and Output attributes. When there is a single attribute in input, it is simple linear regression. When there are multiple attributes, it is multiple linear regression.                            | X = df[['attribute_1', 'attribute_2', ...]]  Y = df['target_attribute']  lr.fit(X, Y)                                                                                                                                                                 |
| Generate output predictions         | Predict the output for a set of Input attribute values.                                                                                                                                                                                                                | Y_hat = lr.predict(X)                                                                                                                                                                                                                                 |
| Identify the coefficient and intercept | Identify the slope coefficient and intercept values of the linear regression model.                                                                                                                                                                                      | coeff = lr.coef_  intercept = lr.intercept_                                                                                                                                                                                                          |
| Residual Plot                       | This function will regress y on x (possibly as a robust or polynomial regression) and then draw a scatterplot of the residuals.                                                                                                                                         | import seaborn as sns  sns.residplot(x=df[['attribute_1']], y=df[['attribute_2']])                                                                                                                                                                   |
| Distribution Plot                   | This function can be used to plot the distribution of data with respect to a given attribute.                                                                                                                                                                           | import seaborn as sns  sns.distplot(df['attribute_name'], hist=False)                                                                                                                                                                                 |
| Polynomial Regression               | Available under the numpy package, for single-variable feature creation and model fitting.                                                                                                                                       | f = np.polyfit(x, y, n)  p = np.poly1d(f)  Y_hat = p(x)                                                                                                                                                                                              |
| Multi-variate Polynomial Regression | Generate a new feature matrix consisting of all polynomial combinations of the features with degree ≤ specified degree.                                                                                                          | from sklearn.preprocessing import PolynomialFeatures  Z = df[['attribute_1', 'attribute_2', ...]]  pr = PolynomialFeatures(degree=n)  Z_pr = pr.fit_transform(Z)                                                                                      |
| Pipeline                            | Data Pipelines simplify the steps of processing the data by chaining transformations and models.                                                                                                                                | from sklearn.pipeline import Pipeline  from sklearn.preprocessing import StandardScaler  Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]  pipe = Pipeline(Input)  Z = Z.astype(float)  pipe.fit(Z, y)  ypipe = pipe.predict(Z) |
| R² value                            | R² (coefficient of determination) measures how close the data is to the fitted regression line.  a. For Linear Regression  b. For Polynomial Regression                                                                     | a)  X = df[['attribute_1', 'attribute_2', ...]]  Y = df['target_attribute']  lr.fit(X, Y)  R2_score = lr.score(X, Y)   b)  from sklearn.metrics import r2_score  f = np.polyfit(x, y, n)  p = np.poly1d(f)  R2_score = r2_score(y, p(x))                 |
| MSE value                           | The Mean Squared Error (MSE) measures the average of squared errors (difference between actual and predicted values).                                                                                                           | from sklearn.metrics import mean_squared_error  mse = mean_squared_error(Y, Y_hat)                                                                                                                                                                    |


**Example Notebook:**

* [notebooks/04_Model_Development.ipynb](notebooks/04_Model_Development.ipynb) 

## 5. Model Refinement and Evaluation

This section focuses on assessing the performance of your developed models and techniques for improving their accuracy and generalization.

Understand various evaluation metrics for different types of models and explore methods for hyperparameter tuning and model selection.

| Process                        | Description                                                                                                                                                            | Code Example |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| Splitting data for training and testing | Separate the target attribute from the rest of the data. Then split the input and output datasets into training and testing subsets. | `from sklearn.model_selection import train_test_split`<br>`y_data = df['target_attribute']`<br>`x_data = df.drop('target_attribute', axis=1)`<br>`x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)` |
| Cross validation score        | When data is limited, use cross validation to evaluate model performance across multiple folds using R² scores.                     | `from sklearn.model_selection import cross_val_score`<br>`from sklearn.linear_model import LinearRegression`<br>`lre = LinearRegression()`<br>`Rcross = cross_val_score(lre, x_data[['attribute_1']], y_data, cv=n)`<br>`Mean = Rcross.mean()`<br>`Std_dev = Rcross.std()` |
| Cross validation prediction   | Use a cross-validated model to predict output values.                                                                             | `from sklearn.model_selection import cross_val_predict`<br>`from sklearn.linear_model import LinearRegression`<br>`lre = LinearRegression()`<br>`yhat = cross_val_predict(lre, x_data[['attribute_1']], y_data, cv=4)` |
| Ridge Regression and Prediction | Use Ridge regression with alpha to avoid overfitting in polynomial models.                                                        | `from sklearn.linear_model import Ridge`<br>`pr = PolynomialFeatures(degree=2)`<br>`x_train_pr = pr.fit_transform(x_train[['attribute_1', 'attribute_2', ...]])`<br>`x_test_pr = pr.fit_transform(x_test[['attribute_1', 'attribute_2', ...]])`<br>`RidgeModel = Ridge(alpha=1)`<br>`RidgeModel.fit(x_train_pr, y_train)`<br>`yhat = RidgeModel.predict(x_test_pr)` |
| Grid Search                   | Use Grid Search with cross-validation to find the best alpha value for Ridge regression.                                          | `from sklearn.model_selection import GridSearchCV`<br>`from sklearn.linear_model import Ridge`<br>`parameters = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, ...]}]`<br>`RR = Ridge()`<br>`Grid1 = GridSearchCV(RR, parameters, cv=4)`<br>`Grid1.fit(x_data[['attribute_1', 'attribute_2', ...]], y_data)`<br>`BestRR = Grid1.best_estimator_`<br>`BestRR.score(x_test[['attribute_1', 'attribute_2', ...]], y_test)` |


**Example Notebook:**

* [notebooks/05_Model_Evaluation_and_Refinement.ipynb](notebooks/05_Model_Evaluation_and_Refinement.ipynb) 
