<img width="401" height="82" alt="image" src="https://github.com/user-attachments/assets/78d8eaeb-53fa-48db-b51b-90299bfe6fce" /># AI-in-EHR-
Use machine learning models to analyze vital signs, laboratory data, past medical records, and demographic information to predict disease risk in advance.

* Design data flow diagrams and organize the MIMIC-IV dataset, including time-series processing, missing value imputation, and outlier handling, to complete data integration and feature extraction.

## final result

**without timeline**

| 模型    | AUC         | F1          | Precision   | Recall      |
| ----- | ----------- | ----------- | ----------- | ----------- |
|MLP	|0.755264974|	0.341463415	|0.725225225	|0.223300971|
RNN|	0.755749711|	0.305775764	|0.833333333|	0.187239945|
SAINT	|0.756148694|	0.33916849	|0.803108808|	0.214979196|




**with timeline**

| 模型    | AUC         | F1          | Precision   | Recall      |
| ----- | ----------- | ----------- | ----------- | ----------- |
| MLP   | 0.776492138 | 0.345782614 | 0.728912347 | 0.230845972 |
| RNN   | 0.779015274 | 0.312958421 | 0.836472915 | 0.195628314 |
| SAINT | 0.783214895 | 0.341672495 | 0.807254612 | 0.218947203 |
