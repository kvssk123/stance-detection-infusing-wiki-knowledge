# stance-detection-infusing-wiki-knowledge
## Dataset
In the dataset folder we have the three datasets, Pstance, Covid 19 and Vast.
For the reproduction, we have used Pstance and Vast datasets
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
To perform error analysis, we have modified the Engine.py file and created a src_new folder. We performed the error analysis for PStance, target-specific stance detection, Biden. To get the error analysis results, run the following command:
```angular2html
python run_err_biden.py
```
