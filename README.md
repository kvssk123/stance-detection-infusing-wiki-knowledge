# stance-detection-infusing-wiki-knowledge
To reproduce the results, all the folders and files above should be in the same directory
## Dataset
In the data folder we have the three datasets, Pstance, Covid 19 and Vast.
The new dataset COVID CQ is also in the data/covid19-stance a hcq_train, hcq_test and hcq_val
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

To get the result for VAST, zero/few-shot stance detection
```angular2html
python run_vast.py
```
## Error Analysis
To perform error analysis, we have modified the Engine.py file and created a src_error_analysis folder. We performed the error analysis for PStance, target-specific stance detection, Biden. To get the error analysis results, run the following command:
```angular2html
python run_err_biden.py
```
## Running XLM R
To run XLM R, we have modified the codes and created a src_xlm_r folder. We
