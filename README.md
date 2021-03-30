# Enhancing Performance Bug Prediction Using Anti-patterns #

**Requirements**
* Python 3.5 or newer
* Library [pandas](https://pandas.pydata.org) library 0.23.1 or newer
* Library [sklearn](https://scikit-learn.org/stable) library 0.22.2 or newer
* Library [scipy](https://www.scipy.org) library 1.4.1 or newer
* Library [numpy](https://numpy.org) library 1.18.5 or newer
* Library [xgboost](https://xgboost.readthedocs.io/en/latest/get_started.html) library 1.1.1 or newer

**Files description**
*  **data** 
	*  **experiment_dataset** folder contains the labeled dataset of clean and buggy files that have performance bugs in the 80 studied GitHub projects. Each project has a separate folder.
	* **issue_reports** folder contains the sample data of the non-performance bug reports and performance bug reports. The full dataset of bug reports is hosted on [Onedrive](https://queensuca-my.sharepoint.com/:f:/g/personal/17gz2_queensu_ca/Ej4FCkF34ZtHvimgPoMHwAMBtnHyQIv3cqIAOGt4zS5rzQ?e=EaVQdA).
*  **python scripts** folder contains the scripts for evaluating the performance machine learning algorithms and calculating the effects of metrics. 
    * **algorithms_comparison.py** contains the script for training and evaluating the performance of machine learning algorithms for predicting performance bugs.
    * **metrics_effects** contains the script for calculating the effects of metrics on the performance bug prediction models.

For any questions, please send email to g.zhao@queensu.ca

