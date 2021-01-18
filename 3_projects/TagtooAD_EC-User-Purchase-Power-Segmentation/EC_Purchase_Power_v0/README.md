# EC Purchase Power Label
2020.06.24

## Definition
`Avg. Item Value`: **Average value of all products** buyer bought in the shift duration.
> value.sum() / num_item.sum()
1. `0-500`
2. `500-`


`Avg. Order Value`: **Average value of unique orders** buyer made in the shift duration.
> value.sum() / transcation_id.count()
1. `0-1000`
2. `1000-`



### Querying data
`make-train-predict-query.py`
Search and export data table from BigQuery with the target of both train and test data daterange at once.


#### Decide daterange
For example: We want to predict buyer purchase-power label in `2020.05`, 
input date of testing data will be: 10 shifts-backward, each shifts are 1-month-based with 2-week-shifting, starts from `2020.04.30`.
Training data for will shift-backward starts from `2020.03.31`, do validation on `2020.04`.

[Input] 
- train & test data daterange

[Output] 
- generate sql script 
- finish BigQuery inquire and table export through python libraries 



### Cleansing & transform
`transform-train-val-predict.py`
Do data cleansing, shifts generation and buyer info at ecs/inds.

[Input] 
- raw transcation data

[Output]
- labeled users with 2 kinds of purchase power label
- labeled records by shift for each 4 labels
- shifts of buyer purchased in each ECs / Industries



### Train
`train.py`
model: Logistic Regression
explanation: Try find the pattern of a **more stable** reference from each shifts information. 
Since only when the buyer had transactions before, we have enough information to predict if he/she bought at what value level next time. 
We assume the past 10 shifts of purchase records may reflect the buyer's consume level, and we want to know which shifts indicate more important imformation.

[Input]
- frequency of ec/ind transaction in each shift + shift purchased records in dummy format.
- labeled validation data

[Output]
- a set of weights (LR coefficients)



### Predict
`predict.py`

[Input] 
- test data with the same format as train data.
(frequency of ec/ind transaction in each shift + shift purchased records in dummy format)

[Output]
- label prediction of each buyer who has showed up in past 10 shifts.




