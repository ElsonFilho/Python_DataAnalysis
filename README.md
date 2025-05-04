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

**Description:** Discover how to use various plotting libraries (like Matplotlib and Seaborn) and statistical methods to understand the distribution of variables, identify correlations, and detect potential outliers.

**Example Notebook:**

* [notebooks/03_Exploratory_Data_Analysis.ipynb](notebooks/03_Exploratory_Data_Analysis.ipynb) (Showcase different types of plots, correlation analysis, and basic statistical tests)

## 4. Model Development

This section introduces the process of building predictive models using Python's powerful machine learning libraries (like scikit-learn).

**Description:** Learn the fundamental steps in model development, including selecting appropriate algorithms, training models on your data, and making predictions.

**Example Notebook:**

* [notebooks/04_Model_Development.ipynb](notebooks/04_Model_Development.ipynb) (Demonstrate a simple model building process, including splitting data into training and testing sets)

## 5. Model Evaluation and Refinement

This section focuses on assessing the performance of your developed models and techniques for improving their accuracy and generalization.

**Description:** Understand various evaluation metrics for different types of models and explore methods for hyperparameter tuning and model selection.

**Example Notebook:**

* [notebooks/05_Model_Evaluation_and_Refinement.ipynb](notebooks/05_Model_Evaluation_and_Refinement.ipynb) (Show examples of model evaluation metrics and basic refinement techniques)
