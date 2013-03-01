Homework 4:  Linear Regression with GSS Data
============================================

You will use your linear regression module from last week to analyze the [General Social Survey](http://www3.norc.org/gss+website/) (GSS) data.  This is a yearly social science survey that "takes the pulse of America."

**Written HW Due Monday March 11, presentation Wednesday March 13**

---

Data directory layouts
----------------------

You will have to transform your data by cleaning/cutting out certain columns.  This can lead to a mess of different data files.  Here are some suggestions.

You have the following layout by default:

    notebooks/
    scripts/
    src/

    data/
      raw/
      processed/

* Remember not to commit any data to the repository.
* I like to keep the data in `raw` completely untouched copies from websites, or common transformations of them (e.g. the csv files that result of `Getting the data` above).  The key point is that once data goes into `raw` I *never* change it.
* The `processed` directory is for the altered versions of the `raw` directory.
* There is a `scripts` directory that can be used to store scripts (Python or Bash) that transform data from the type in `raw` to the type in `processed`.  
* T also use ipython notebooks (stored in `notebooks/` to transform data from `raw` to `processed`.  I commit these scripts and notebooks to the repo and other people can run them to get copies of my processed data.
* For longer projects I create snapshots of the `processed` directory that have a timestamp on their name.  E.g. `processed-2013-02-11`.

---

Project deliverables
--------------------

1. Use the 2006 data to train models, test those models on 2010 data.
2. Predict `income06` as a function of other variables.  For simplicity, this should be one single model that includes everyone...in other words, don't segment your data.  Note that `income06` is missing in about 15% of responses.  You don't have to predict `income06` for these people.  This model should work, even in the presence of missing data (with the exception of missing `income06`)!  So, you should probably fill the missing values with something.
3. Find one other relation to predict.  Make sure it is appropriate for linear regression.  Segment or do whatever you want.  The model can work for the whole population or subpopulations.
4. Make a 15 minute slide-show presentation documenting your work.  This is what you turn in.  Two randomly chosen groups will present this in class on Wednesday March 13.  The intended audience is the class...so present at the appropriate technical level.

---

Getting started
---------------

### Useful links

* The [2008 GSS Codebook](/misc/2008_GSS_Codebook.pdf) will be useful for variable definitions.
* The [GSS User's Guide](http://publicdata.norc.org/GSS/DOCUMENTS/OTHR/GSS_NESSTAR_Guide.pdf) shows you how to search for variable description using the website.  The website is very very very slow.


### Basic workflow

1. Get data
2. Inspect data
3. Clean data
4. Explore relationships (EDA)
5. Fit model
6. Inspect results
7. Repeat  2-7

#### Get data

1. Download the 2006 and 2010 datasets from [this site](http://www3.norc.org/GSS+Website/Download/STATA+v8.0+Format/)
2. Convert these STATA dataset into Pandas DataFrames using [these instructions](/Extras/2013/02/15/convert-stata/)
3. Store them as csv files using (assuming the DataFrame is named `df`):

    df.to_csv('filename', index=False)

#### Inspect data

* Inspect the csv files with `less` and see what they look like.  Remember `Ctrl-f`, `Ctrl-b` to move forward and backward.
* Use `head` to create a file (probably located in `/tmp/`) containing the first 100 lines.  Look at this file in excel (or `libreoffice` in ubuntu).  You may get an error about too many columns...that's ok, just look at what you can!

#### Clean data

* Start your ipython notebook with `ipython notebook --pylab inline`.  
* Create a new notebook named `cleaning-your-name`.  This will be used for cleaning data.
* See `notebooks/HW4_cleaning_EDA` for an example.
* Read in the 2006 and 2010 datasets into DataFrames named `df2006`, `df2010`.  
* We will probably have little use for columns that are mostly NaN.  Use `df.count().order()` to figure out which columns have lots of missing values.  Chop off these columns by creating a boolean mask that will be true if a column has enough good entries and then using `df.ix[:, mask]`.
* We are only interested in variables that are in both datasets, so use pandas reindex to modify and align the columns like so

    col = df2006.columns.intersection(df2010.columns)
    df2006 = df2006.reindex(columns=col)
    df2010 = df2010.reindex(columns=col)

#### EDA

See `notebooks/HW4_cleaning_EDA` or goto [this link](http://nbviewer.ipython.org/url/columbia-applied-data-science.github.com/misc/HW4_cleaning_EDA.ipynb) for an example.


#### Build your model

See `notebooks/HW4_regression_example` or goto [this link](http://nbviewer.ipython.org/url/columbia-applied-data-science.github.com/misc/HW4_regression_example.ipynb) for an example.

You will have to add variables and see if it improves your fit.  Make sure your variables make sense intuitively.  Do EDA and read about the data to gain intuition.
