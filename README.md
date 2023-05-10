# stance-detection-infusing-wiki-knowledge
To reproduce the results, all the folders and files above should be in the same directory
## Dataset
In the data folder we have the three datasets, Pstance, Covid 19 and Vast.
The new dataset COVID CQ is also in the data/covid19-stance as hcq_train, hcq_test and hcq_val
translated_datasets contains all the translated datasets for VAST and Covid 19 tweets data.

## Run
To get the result for PStance, target-specific stance detection, Biden
```angular2html
python run_pstance_biden.py
```
To get the result for PStance, target-specific stance detection, Sanders
```angular2html
python run_pstance_sanders.py
```
To get the result for PStance, target-specific stance detection, Trump
```angular2html
python run_pstance_trump.py
```

To get the result for PStance, cross-target stance detection, Biden $\rightarrow$ Sanders
```angular2html
python run_pstance_biden2sanders.py
```
To get the result for COVID19-Stance, target-specific stance detection, face mask
```angular2html
python run_covid_mask.py
```
To get the result for COVID19-Stance, target-specific stance detection, HCQ
```angular2html
python run_covid_hcq.py
```
To get the result for VAST, zero/few-shot stance detection
```angular2html
python run_vast.py
```
## Running XLM R
To run XLM R, we have modified the codes and created a src_xlm_r folder.
To get the result for school closures Covid data on XLM R
```angular2html
python run_covid_school_xlm_r.py
```
To get the result for stay at home Covid data on XLM R
```angular2html
python run_covid_home_xlm_r.py
```
## Running on translated_datasets
To run on translated_datasets, first download the respective train,test and val files from translated_datasets folder and replace the existing train,test and val files in data folder.
For example, to run VAST on German data, first download the train,test and dev files from german folder in translated_datasets folder and replace them with train,test and dev files in data folder
## Error Analysis
To perform error analysis, we have modified the Engine.py file and created a src_error_analysis folder. We performed the error analysis for PStance, target-specific stance detection, Biden. To get the error analysis results, run the following command:
```angular2html
python run_err_biden.py
```

