# Enhancing Performance Bug Prediction Using Performance Code Metrics #

**Preliminary_RQ**
* **Requirements**
* R version 3.4.4 or newer
* Library [beanplot] (https://cran.r-project.org/web/packages/beanplot/beanplot.pdf) version 1.2 or newer
* Library [effsize] (https://pandas.pydata.org) version 0.8.1 or newer

* **Files description**
	*  **non-perf_bugs and perf_bugs** folders contain the sample data of the non-performance bug reports and performance bug reports from GitHub project [Elastisearch] (https://github.com/elastic/elasticsearch). The full dataset of bug reports from the 80 studied GitHub projects is hosted on [Onedrive](https://queensuca-my.sharepoint.com/:f:/r/personal/17gz2_queensu_ca/Documents/Bug%20Reports?csf=1&web=1&e=Yg5dQo).
	*  **issue_characteristics.csv** file contains the extracted information (e.g., time taken to fix bug reports and number of comments from developers) of bug reports from the 80 studied GitHub projects.
	*  **Preliminary_RQ.R** file contains the R scripts to conduct the Wilcoxon Rank Sum tests and Cliff’s delta tests to compare the characteristics of non-performance bug reports and performance bug reports. 


**RQ1_RQ2**
* **Requirements**
* Python 3.5 or newer
* Library [pandas](https://pandas.pydata.org) version 0.23.1 or newer
* Library [sklearn](https://scikit-learn.org/stable) version 0.22.2 or newer
* Library [scipy](https://www.scipy.org) version 1.4.1 or newer
* Library [numpy](https://numpy.org) version 1.18.5 or newer
* Library [xgboost](https://xgboost.readthedocs.io/en/latest/get_started.html) version 1.1.1 or newer
* Library [Imbalanced-learn](https://imbalanced-learn.org/stable/index.html) version 0.8.1 or newer

**Files description**
	*  **data** folder contains the labeled dataset of clean and buggy files that have performance bugs in the 80 studied GitHub projects. Each project has a separate folder.
	*  **evaluate_functions.py** file contains the script to conduct out-of-sample bootstrap, create training and testing datasets, optimize parameters of machine learning algorithms, and evaluate the performance of machine learning algorithms. 
	*  **all_metrics.py** file contains the script to test the performance of the seven studied machine learning algorithms for predicting performance bugs using all three types of metrics (i.e., performance code metrics, code metrics, and process metrics). 
	*  **no_proposed_metrics.py** file contains the script to test the performance of the seven studied machine learning algorithms for predicting performance bugs without using proposed performance code metrics. 
	*  **no_code_metrics.py** file contains the script to test the performance of the seven studied machine learning algorithms for predicting performance bugs without using code metrics. 
	*  **no_process_metrics.py** file contains the script to test the performance of the seven studied machine learning algorithms for predicting performance bugs without using process metrics. 

For any questions, please send email to g.zhao@queensu.ca

