#!/bin/sh
python3 make-query-train-predict.py --from_date=20190401 --to_date=20200401 --media_id=1604 --media_active_date=20200101 --media_deactive_date=2020331
python3 transform_train_val_predict.py --from_date=20190401 --to_date=20200401 --predict_month=3 --predict_year=2020 --itemValueCut=500 --orderValueCut=1000
python3 train.py --from_date=20190401 --to_date=20200401 --predict_month=3 --predict_year=2020 --threshold=0.8
python3 predict.py --from_date=20190401 --to_date=20200401 --predict_month=3 --predict_year=2020 --threshold=3.5