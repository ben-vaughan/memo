## **Machine Learning with Python Foundations**
https://www.linkedin.com/learning/machine-learning-with-python-foundations/
https://www.youtube.com/watch?v=D9sU1hLT0QY

Supervised Learning
- Linear Regression
- Logistic Regression
- KNN (k-nearest-neighbours) Algorithm
- Naive Bayes
- Classification

Unsupervised
- Clustering

TEST!

<br/>
<br/>

**Contents**:
1.	**Machine Learning**
    -	What is ML?
    -	What is not ML?
    -	What is Unsupervised Learning?
    -	What is Supervised Learning?
    -	What is Reinforcement Learning?
    -	What are the Steps to ML?
2.	**Collecting Data for ML**
    -	Things to Consider when Collecting Data
    -	How to Import Data in Python
3.	**Understanding Data for ML**
    -	Describe your Data
    -	How to Summarize Data in Python
    -	Visualize your Data
    -	How to Visualize Data in Python
4.	**Preparing Data for Machine Learning**
    -	Common Data Quality Issues
    -	How to Resolve Missing Data in Python
    -	Normalizing your Data
    -	How to Normalize Data in Python
    -	Sampling your Data
    -	How to Sample Data in Python
    -	Reducing the Dimensionality of Your Data
5.	**Types of Machine Learning Models**
    -	Classification vs. Regression Problems
    -	How to Build a Machine Learning Model in Python
<br>
<br>

 
### **1. Machine Learning**
**What is ML?** <br/>
Traditional way to program a computer – input, process, output. Computer scientist Arthur Samuel thought of a different approach – wondered if computers could infer logic instead of being given explicit instructions. What if we gave a computer just the input and output, and the computer could figure out the process that was used to transform the data?<br/>
After we train a model, going forward, we can just give it input data and allow it to use its process to give us output. This is supervised learning. With unsupervised learning, we ask the machine to analyse the input data, and identify a pattern in that input. In reinforcement learning, there are two entities: the agent and the environment. This is when the agent performs an action and gets either positive or negative feedback back from its environment. This will allow the agent to eventually figure out what actions are required to accomplish the task at hand. Commonly used in robotics, computer game engines, self-driving cars.<br/>
<br/>

**What is not ML?** <br/>
ML borrows a lot of concepts from statistics, but also from information theory, calculus, algebra… it’s a combination of several different fields of mathematics. ML is mostly concerned with the future results of a dataset – this is prediction. Statistics, however, is mostly interested in the relationship between two variables – this is called inference. The overlap between the two is referred to statistical modelling. ML, Data Mining and Optimization are all under the Data Science field. Data Mining is focused on the discovery of previously unknown properties of the data. Business analytics: ML = predictive analytics, data mining = descriptive analytics, optimization = prescriptive analytics. Used to recommend actions based on prior performance.<br/> 
Deep learning is a ML approach that falls under the umbrella of supervised learning. The three major branches of ML are reinforcement learning, unsupervised learning, and supervised learning. Machine Learning is not the same thing as AI; it is a subfield of AI. AI also includes translation, computer vision, etc.<br/>
<br/>

**What is Unsupervised Learning?** <br/>
Unsupervised learning is the process of building a descriptive model. Descriptive models are used to summarise and group unlabelled data in new and interesting ways. Unsupervised model as there is no pre-existing rubric in which the data can be matched to – there is no ‘supervising’ dataset to determine whether the output is correct or not. Identifying unknown patterns in unlabelled data.<br/>
<br/>

**What is Supervised Learning?** <br/>
Supervised learning is the process of training a predictive model. Predictive models are used to assign labels to unlabelled data based on patterns learned from previously labelled historical data.<br/>
Before predicting, we need to train the model. In ML, we call the input the independent variable, and the output the dependent variable. These make up our training data. For example, independent variables would be the size of the loan, the grade of the loan, or the stated purpose of the loans, whereas the dependent variable will be the outcome variable – default. The predictive accuracy of a model is how well it can predict the expected outcome of an event, using its independent variables.<br/>
One of the most popular definitions of supervised machine learning is a quote by Tom Mitchell: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.”, essentially meaning that if a model’s performance improves with experience, then that model is learning. Experience = training data. Task = who will default. Performance Measure = predictive accuracy.<br/>
<br/>

**What is Reinforcement Learning?** <br/>
Reinforcement learning is the science of learning to make decisions from interaction, or the process of learning through feedback. Very similar to early childhood learning – gets positive or negative feedback dependent on what action they took.<br/>
Reinforcement learning attempts to tackle two distinct learning objectives: <br/>
    -	Finding previously unknown solutions to a problem. Example includes a ML model built to play chess against a human. <br/>
    -	Finding online solutions to problems that arise due to unforeseen circumstances. Examples include finding alternative routes based upon a road being closed. <br/>
    
Reinforcement learning involves two primary entities: the agent and the environment. The agent takes actions in the environment, and the environment responds to these actions by giving feedback. This feedback contains two components: the state and reward. The state describes the impact of the agent’s previous actions on the environment, and the possible actions that the agent can take. The agent’s primary objective is to maximize the sum of rewards it receives over the long term.<br/>
Exploitation: evaluating the policy table and choosing the action that will give the agent the highest reward. Can randomly decide if some actions return the same reward. Terminal state = the current cycle of learning has ended.<br/>
Challenge: the exploitation vs exploration trade-off. If left unchecked, the agent will always try to take the actions it had taken in the past and found to be effective in maximizing reward. To discover a new sequence of actions with potentially higher rewards, the agent must sometimes choose actions with little to no consideration for their associated award – this is exploration. An agent that only solves problems through exploitation will only be able to solve problems that it has previously encountered. An agent that focuses only on exploration will not learn from prior experience. A balanced approach is needed.
<br/>

**What are the Steps to Machine Learning?**
1. Data Collection: identify and acquire the data you need for machine learning
2.	Data Exploration: a process of describing, visualising, and analysing your data to better understand it. Size of table, duplicate values, types of data, outliers…
3.	Data Preparation: modify your data so it works for the type of machine learning you intend to do. Modifying or transforming the structure of data to make it easier to work with – may involve normalization. Garbage in, garbage out. 80% of our time should be focused on the data.
4.	Modelling: apply a machine learning approach to your data.
5.	Evaluation: assess how well your machine learning approach worked.
6.	Actionable Insight: identifying a potential course of action depending on the results of the machine learning approach you used.
<br>
<br>


 
### **2. Collecting Data for Machine Learning**
**Things to Consider when Collecting Data** <br/>
Primary objective is to identify and gather data we need for machine learning. 5 key characteristics that we’re looking for when looking at data.
    -	Accuracy: supervised learning algorithms use this as a baseline, or a source of truth. Often referred to as ‘ground truth data’.
    -	Relevance: the type of data we collect should be relevant to the conclusions we’re trying to form.
    -	Quantity: understanding the ML algorithm itself and how much data it requires.
    -	Variability: our ground truth data must vary somehow so our model can gain a broader understanding of the data.
    -	Ethics: avoid bias, protect individual’s data and ensure you have their consent to utilise their information.

**How to Import Data in Python** <br/>
Pandas package – provides easy to use functions for creating, structuring and ordering data. Import it with the ‘import’ command. Create a series with `pd.series().` Can use the `type()` function to check the type of an object. A pandas dataframe is a collection of pandas series, all sharing the same index type.

We can also import data from an external source rather than constructing one ourselves. We can use `pd.read_csv()` to read a CSV (comma-separated-value) documnet. There is also a `pd.read_excel()` function which can import excel .xlsx files. Note that if there are multiple sheets contained within the excel file, only the first sheet will be read. To import a different sheet, we can specify it using `sheet_name` as a parameter.
<br>
<br>

### **3. Understanding Data for Machine Learning**
**Describe your Data** <br/>
Data exploration is a process where we describe, visualise and analyse data in order to better understand it.
In machine learning, we've terms to describe our data.
  - Instance: this refers to a row of data. An individual tuple, or record, or observation.
  - Feature: refers to a column of data. A property, field, variable, characteristic.
  - Continuous Feature: holds data in the form of an integer or real number. Infinite number of possible values between lower and upper bounds.
  - Categorical Feature: feature is an attribute that holds data stored in a discrete form. Set scope of values.
In supervised machine learning, we use the values of a set of features known as 'independent variables' to predict 'dependent variables'.
If the dependent variable is categorical, it is referred to as a class.
If the dependent feature is continuous, it is referred to as a response.
The dimensionality of a dataset refers to the number of features within a dataset. Higher dimensionality indicates higher complexity.
Sparsity and density referred to the degree to which data exists in a dataset. 20% of missing data: 20% sparse or 80% dense.

**How to Summarize Data in Python** <br/>
One method to summarise data is the `.info()` method. This outputs the count and general information.
  - If we want the first couple of rows, we can use the `.head()` method. First 5 rows.
  - `washers[['BrandName']].describe()` will describe a certain field / feature.
  - For numeric columns, the describe method will return average, standard deviation, percentiles...
  - We can also count the number of occurrences of a particular feature using `.value_count()`.
  - We can also find the % of occurrences using `.value_count(normalize = True)`.
  - We can also find the statistical information of a particular feature, e.g. `washers[['Volume']].mean()`
  - We can also get specific aggregations at group level, e.g. `washers.groupby('BrandName')[['Volume']].mean`
  - Compute multiple aggregates: `washers.groupby('BrandName)[['Volume']].agg(['mean', 'median'])`

**Visualise your Data** <br/>
Data visualisations are necessary for analysing, asking and answering question about data.
Primary types of data visualisations:
  - *Comparison*: illustrate the difference between two or more items, e.g. boxplot. Answers questions such as:
    - Is a feature important?
    - Does the median value of a feature differ between subgroups?
    - Does a feature have outliers?
  - *Relationship*: illustrates the correlation between two or more variables, e.g. scatterplots, line charts. Answers questions such as:
    - How do two features interact with each other?
    - Is a feature important?
    - Does a feature have outliers?
  - *Distribution*: illustrate the statistical distribution of the values of a feature, e.g. histogram.
    - Does a feature have outliers?
    - How spread out are the values of a feature?
    - Are the values of a feature symmetric or skewed?
  - *Composition*: shows the component makeup of the data, e.g. stacked bar charts, pie charts.
    - How much does a subgroup contribute to the total?
    - What is the relative or absolute change in the composition of a subgroup over time?

**How to Visualise Data in Python** <br/>
`matplotlib` is one of the most popular visualisation packages in Python.
  - `df.plot(kind ='scatter', x = 'x_axis_values, x = 'y_axis_values'` for a scatterplot.
  - `df.plot['desired_field'].plot(kind = 'hist')` for a histogram.
  - `df.pivot(columns = 'categorical_feature', value ='continuous_feature').plot(kind = 'box', figsize = (10,6))` continuous feature vs. categorical feature.
  - `vehicles.groupby('year')['drive'].value_count().unstack().plot(kind = 'bar', stacked = TRUE, figsize = (10,6)`
<br>
<br>

### **4. Preparing Data for Machine Learning**
**Common Data Quality Issues** <br/>
Data preparation is a process that makes sure that our data is in such a way that it is suitable for the machine learning model that we are using.
Garbage in, garbage out. Most common issue is missing data. Several approaches:
  - Remove tuples with misssing values.
  - Replace the missing values with a set term - 'NA' or '-'
  - Use *imputation*: systematic approach to fill in missing data by using the most suitable value. Can use median.

Another issue is *outliers*. Values that are unusual with respect to typical values.
For most classes (categorical dependent), their values are not uniformly distributed. Class imbalance.

**How to Resolve Missing Data in Python** <br/>
How do we detect missing values using Python? <br/>
- Data Deletion: <br/>
    `mask = students['State'].isnull()`. <br/>
    Returns a boolean series. Then, <br/>
    `students[mask]` - this returns the tuples in which there are missing values. <br/>
    We can then drop these tuples: <br/>
    `students.dropna()` <br/>
    We can remove rows with missing data in certain columns only: <br/>
    `students.dropna(subset = ['State', ['Zip'], how = 'all')` <br/>
    We can also remove columns with missing data: <br/>
    `students.dropna(axis = 1)` <br/>
    Often, we want to drop columns if there's a threshold of them missing. <br/>
    `students.dropna(axis = 1, thresh = 10)` <br/>

- Data Replacement: <br/>
    `students = students.fillna({'Gender', 'Female'})` <br/>
    Replaces all NA entries in the 'Gender' column with 'Female. <br/>
    We can also use a function to replace NA figures. <br/>
    `students = students.fillna({'Age': students['Age'].median})` <br/>
    Replacing an individual cell: <br/>
    `mask = (students['City'] == 'Granger') & (students['State'] == 'IN'` <br/>
    `student.loc[mask, 1]` <br/>
    `student.loc[mask, 1] = 49120`
<br/>

**Normalizing your Data** <br/>
An ideal dataset has no missing values or no values that deviate from the norm. <br/>
Data preparation is a process of ensuring that data is suitable for the ML approach we are intending to do.<br/>
Involves modifying, or transforming the data to make it easy to work with.
*Normalisation*: ensures that values share a common property. Often involves scaling data to fall within a small or specified range. Often required, it reduces complexity, and improves interpretability.
- **Z-Score Normalisation**<br/>
  Transform the data so that it has a mean of 0 and standard deviation of 1. The normalized value $v'$ is computed as:
  $$ v^{'} = \frac{v-\bar{F}}{\sigma_F} $$
  where $\bar{F}$ and $\sigma_{F}$ are the mean and standard deviation of feature $F$, respectively.
- **Min-Max Normalization**<br/>
  Transform the data from measured units to a new interval, which goes from $lower_F$ to $upper_F$ for feature $F$:
  $$ v^{'} = \frac{v-min_F}{max_F - min_F}(upper_F - lower_F) + lower_F $$
  where $v$ is the current value of feature $F$ and $v^{'}$ is the normalised value. Essentially, the lowest value in the dataset will have a score of 0, whereas the highest will have a score of 1.
- **Log Transformation**<br/>
  Transform the data by replacing the values of the original data with its logarithm, such that:
  $$ v^{'} = log(v) $$
  where $v$ is the original value and $v^{'}$ is the normalized value. Can be $log_2$ or $log_10$. Important to note that this only works for positive values. Minimises the difference between the outliers.
<br/>

**How to Normalize Data in Python** <br/>
Here we can utilise maplibplot, scikit and pandas to normalise data.
- **Performing Min-Max Normalisation**<br/>
  ```python
  from sklearn.preprocessing import MinMaxScaler
  co2emissions_mm = MinMaxScaler().fit_transform(vehicles[['co2emissions']])
  co2emissions_mm = pd.DataFrame(co2emissions_mm, columns = ['co2emissions'])
  co2emissions_mm.describe()
  co2emissions_mm.plot(kind = 'hist', bins = 20, figsize = (10, 6))
  ```
  Basic structure and shape of the histogram stays the same regardless of the transformation. <br/>
  X-axis and y-axis differ in scales.
- **Performing Z-Score Normalisation**<br/>
  ```python
    from sklearn.preprocessing import StandardScaler
    ... # from above
  ```
<br>

**Sampling your Data** <br/>
In supervised machine learning, we want to map a given input to a given output. <br/>
We must first split our datasets into training and testing datasets. <br/>
Sampling is a process of selecting a subset of the instances in a dataset as a proxy for the whole.<br/>
In statistics, the entire dataset is the population, and the subset is called the sample.<br/>
Can sample with or without replacement. Sampling with replacement is important - called bootstrapping in machine learning.<br/>
Stratified sampling is a modification of simple random sampling. Ensures that the distribution of values for a particular feature matches the distribution in the overall feature. <br/>
- First divided into homogenous subgroups ('strata'). Can group by gender, for example.
<br/>

**How to Sample Data in Python** <br/>
Again, we are using the scikit module / library. Before splitting data, we must differentiate the independent and the dependent variables.<br/>
```python
response = 'co2emissions'
y = vehicles[[response]]                # our dependent variable.
predictors = list(vehicles.columns)     # gives us a list of all the columns in the dataset.
predictors.remove(response)             # remove the dependent variable from the dataframe.
x = vehicles[predictors]                # create a dataframe with the dependent variables.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train.shape                           # holds independent variables of the dataset.
y_train.shape                           # holds dependent variables of the dataset.
x_test.shape                            # holds the independent variables of the test set.
y_test.shape                            # holds the dependent variables of the test set.
```
By default, the `train_test_split` holds 25% of the original dataset. Can change this:
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)
```
If we're using Stratified approach, add a parameter called `stratify = x['drive']`.
<br/>

**Reducing the Dimensionality of your Data** <br/>
Dimensionality Reduction involves the process of reducing the number of features in a dataset prior to modelling. Reduces complexity and helps avoid 'the curse of dimensionality'. <br/>
The curse of dimensionality is a phenomenom that describes the eventual reduction in the performance of a model as the number of dimensions increases. <br/>
As complexity $(p)$ increases, the amount of data $(n)$ needed to generalise accurately grows exponentially. If $n$ is held constant, performance will eventually diminish as $p$ increases. <br/>

Feature Selection: identify the minimal set of features needed to build a good model. Remove features that do not contribute significantly to the performance of the model. Also known as variable subset selection.<br/>
Feature Extraction: the use of mathematical functions to transform high-dimensional data into lower dimensions. Also known as feature projection. One notable disadvantage is that interpretting the values are difficult. <br/>
<br>
<br>

### **5. Types of Machine Learning Models**
**Classification vs Regression Problems** <br/>
Apply a machine learning approach to your data. <br/>
In supervised machine learning, our model must map our independent variables to an output, the dependent variable.<br/>
Can either be called a 'classification problem' or a 'regression problem'. Classification problems are used when the dependent variable is *categorical*. Regression problems are used when the dependent variable is *continuous*.<br/>
Most popular ML algorithms such as KNN, Neural Networks, Naive Bayes, Decision Trees and Support Vector Machines can be used to solve both regression and classification problems. <br/>
Some algorithms are suited for regression problems only (logistic regression, simple linear regression, multiple linear regerssion, poisson regression, polynomial regression).<br/>

The evaluation stage is when we assess how well our model has worked. We must use the 'test' data to evaluate the performance of our model.
If our model is solving a regression problem, we must use the Mean Absolute Error (MAE) to measure the accuracy of our model.
$$ MAE = \frac{\sum{|Predicted - Actual|}}{Number\ of\ Test\ Instances} $$ 
<br>

**How to Build a Machine Learning Model in Python**
Import the data, explore the data, prepare the data.<br/>
To build a linear regression model, we must import it from the sklearn package.
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
model = LinearRegression().fit(x_train, y_train)
model.intercept_                    # returns the intercept of the regression line
model.coef_                         # returns the coefficients for each feature
model.score(x_test, y_test)         # returns r^2 of the model.
y_pred = model.predict(x_test)
mean_absolute_error(y_test, y_pred) # returns the MAE of the model.
```
The closer $R^2$ is to 1, the better the correlation.
