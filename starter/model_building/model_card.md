# Model Card
* Model Developed by: Rob W
* Date: 2021-12-20
* Version: 1.0.0
* Type: Clasifier
* Dataset Used: https://archive.ics.uci.edu/ml/datasets/census+income

## Model Details
* The model is a default Random Forest Classifier
* No extensive hyper-parameter tuning has been conducted

## Intended Use
The model classifies salary bands of an individual as either above (and equal to) or below $50K
The model itself has 14 explanatory variables including the following
* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Train/test Split
The train test split was 80% training 20% test.
Additionally, one-hot-encoding was leveraged for categorical variables.


## Metrics
The model was evaluated using F-score, precision and recall. 
Said metrics appeared as follows:
* Precision: 0.98616505
* Recall: 0.9706308312
* Fbeta: 0.97566637

Performance metrics by gender slices:
Male
* Precision: 0.8062906937133234
* Recall: 0.7811826968276665
* Fbeta: 0.8240773026818333

Female
* Precision: 0.8423947854263215
* Recall: 0.7671391076115486
* Fbeta: 0.9204635776760662

## Ethical Considerations
Depends on the use of the model. Given that gender, age, etc. are factors in this model; the final use of this model should be deeply considered.

## Caveats and Recommendations

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
