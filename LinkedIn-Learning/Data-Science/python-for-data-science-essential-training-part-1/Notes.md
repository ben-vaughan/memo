### **1. Introduction to the Data Professions**
**Introduction to the data professions** <br/>
Data science, data engineering, data analytics and business intelligence all work together to generate business value from data. Focus on mastering one area at a time. <br/>
Data science is the systematic study of the structure and behaviour of data in order to quantifiably understand past and current occurences, as well as predict the future behaviour of data. <br/>
Traits of data scientists:
- Are interested in *why* rather than *how*. Once we understand why something happens the way it does, we can make predictions on *how* it will happen.
- Derive insights from data, including big data (data from data engineered systems).
- Uncover correlations and causations in business data to support business decision-making
- Generate predictions from data and communicate predictions through data visualization.
<br/>

Data engineering is the design, construction and maintenance of data systems. More interested in *how* rather than *why*.<br/>
Typical tasks of data engineers:
- Design systems to collect, handle and store big datasets.
- Build modular, scalable platforms for data processing.
- Design, build and maintain systems that store and move big data.
<br/>

Data analytics are data products that describe data and how it behaves. These data products are generated from data analysis and visualisation processes. Less software engineering involved.<br/>
- Use applications to analyze data.
- Solid in basic mathematics.
- Understand the inner workings of a business very well.
- Enjoys using data insights to improve business.
- Not into deep analyses.
<br/>
<br/>

**The four flavours of data analysis**
- *Data analysis*: a process for making sense of data. Include data cleaning, reformatting, and recombining data. Carried out with the express intention of discovering trends, patterns and findings in data that describe real-life phenomena.
- *Data science*: the systematic study of the structure and behaviour of data in order to quantifiably understand past and current occurences, as well as predict the future behaviour of that data.
- *Artificial intelligence*: a machine or application with the capacity to autonomously execute upon predictions it makes from data. Two main elements: prediction (predictive modeling from data science), and execution (autonomous response, from engineering).
- *Deep learning*: a set of predictive methodologies that borrows its structure from the neural network structures of the brain. This class of methods is particularly effective for making predictons from big data. A sub-field within data science. Can be used as a decision model with applications to produce deep learning AI.
<br/>
<br/>

**Why use Python for analytics?**
- Easy to learn, human readable
- Extensive array of well-supported data science libraries
- Biggest user base of all data science languages
- Useful in data engineering
<br/>
Useful for data science, data analytics, data engineering. Useful in both a professional and an academic environment. Open-source, can be used in both web and application development.<br/>
Main Python libraries for data science include:
- Advanced Data Analysis: `NumPy`, `SciPy`, `s`
- Data Visualisation: `Matplotlib`, `Seaborn`
- Machine Learning: `scikit-learn`, `TensorFlow`, `Keras`
<br/>
<br/>

**High-level course roadmap**
```mermaid
flowchart TB;
    id1(Introduction to the<br/>Data Professions);
    id2(Data Preparation<br/>Basics);
    id3(Data Visualization<br/>101);
    id4(Practical Data<br/>Visualization);
    id5(Basic Math<br/>and Statistics);
    id6(Data Sourcing via<br/>Web Scraping);
    id7(Build interact graphs<br/>with Plotly)

    subgraph 1
        direction LR
        id1 --> id2 --> id3
    end
    
    subgraph 2
        direction LR
        id4 --> id5 --> id6
    end

    subgraph 3
        direction LR
        id7
    end
    
    1 --> 2 --> 3    
```
<br/>
<br/>
<br/>

### **1. Data Preparation Basics**
**Filtering and selecting**<br/>
Pandas: a data analytics library.<br/>
*Why pandas?* - fast data cleaning, preparation, and analysis; easy to use for data visualisation and machine learning.<br/>
*What is pandas?* - built on top of NumPy, makes it easy to work with arrays and matrices.<br/> *Indexing in pandas:* An index is a list of integers or labels you use to uniquely identify rows or columns. This course will use a set of square brackets `[...]`, or the `.loc[]` indexer.<br/>
*DataFrame Object:* acts like a spreadsheet in Excel, made of a set of Series objects, and are indexable.<br/>
*Series Object:* a single row or column. Always indexed.<br/>
Refer to the Jupyter notebook for code examples.
<br/>
<br/>

**Treating missing values**<br/>
By default, missing values are represented with NaN: 'Not a Number'<br/>
Warning: if your dataset has `0s`, `99s`, or `999s`, be sure to either drop or approximate them as you would with missing values.<br/>
Refer to the Jupyter notebook for code examples.
<br/>
<br/>

**Removing duplicates**<br/>
We remove duplicates to maintain accurate, consistent datasets and to avoid producing erroneous or misleading statistics.
<br/>
<br/>

**Concatenating and transforming**<br/>
Concatenating - combining. Transforming - changing to suit our purposes.
<br/>
<br/>

**Subgrouping and aggregation**<br/>
Useful for categorizing data.<br/>
Can group data to compare subsets, deduce reasons why subgroups differ, and can subset your data for analysis.