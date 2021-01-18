import csv
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

hmpcrt_no = [3376,7146,10380,15521,16827,18651,23278,32813,44488,45934,
            56075,57830,63579,90496,96961,119298,139980,142533,184883,216995,221211,221472,258260] #31384
                
#Patient_data_file_pretranslate_path
path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/Patient_oCare_data_file/')

#bp_data_path
bp_data_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient_all_data_set/bp_data_1024.csv')

#machine_data_path
# machine_data_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient_all_data_set/machine_data_1024_aggregate_by10min.csv')
machine_data_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient_all_data_set/machine_data_1024.csv')

#output_path
# patient_aggregate_format_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient-aggregate-features--format-IV.csv')
# patient_aggregate_format_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient-aggregate-features--format-IV.2.csv')
# patient_aggregate_format_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient-aggregate-features--format-IV.3.csv') # add previous_systole
patient_aggregate_format_path = ('AN-NAN-Hypotension-datasets/Prediction_Hypo_IRB_Data_set/patient-aggregate-features--format-IV.4.csv') # 2mins before systole measured, for numeric prediction


#bp_datas
with open(bp_data_path, encoding='utf-8') as f:
    inp = f.readlines()

bp_data = pd.DataFrame([row.strip().split(',') for row in inp[1:]], columns=inp[0].strip().split(','))
print(f'bp_data: {bp_data.shape}')
for c in bp_data.columns:
    bp_data[c] = bp_data[c].apply(lambda x: x.encode('utf-8').decode('utf-8-sig'))
bp_data.columns = [c.encode('utf-8').decode('utf-8-sig') for c in bp_data.columns]
bp_data['diastolic'] = np.where(bp_data['diastolic']=='', '68', bp_data['diastolic'])
for c in bp_data.columns:
    if c not in ['hmpcrtno','btrim','watch']:
        bp_data[c] = bp_data[c].astype(int)

#machine_data
machine_data = pd.read_csv(machine_data_path)
print(f'machine_data: {machine_data.shape}')
# for c in machine_data.columns:
#     if c not in ['hmpcrtno','btrim','ufprofile','sodiumpfile','watch']:
#         machine_data[c] = machine_data[c].astype(float)



#main
collect = dict()

for hmpcrtno in tqdm(hmpcrt_no):

    collect[str(hmpcrtno)] = list()
    #存放該patient有oCare_data檔案紀錄的日期
    oCare_filedate_list = list()
    
    
    bp_data_hmpcrt_no = bp_data[bp_data['hmpcrtno']==str(hmpcrtno)]
    machine_data_hmpcrt_no = machine_data[machine_data['hmpcrtno']==hmpcrtno]
    oCare_data_hmpcrt_no = Path(path + str(hmpcrtno))
    oCare_data_hmpcrt_no_filelist = list(oCare_data_hmpcrt_no.glob("**/*.csv"))
    for File in oCare_data_hmpcrt_no_filelist: 
        file_date = (File.name)[13:20]
        oCare_filedate_list.append(file_date)
    print(oCare_filedate_list)
    
    for daidate in machine_data_hmpcrt_no.daidate.unique():
        
        bp_data_hmpcrt_no_daidate = bp_data_hmpcrt_no.query(" daidate == @daidate ")
        bp_data_hmpcrt_no_daidate_daitimes = bp_data_hmpcrt_no_daidate.daitime.values.tolist()
        machine_data_hmpcrt_no_daidate = machine_data_hmpcrt_no.query(" daidate == @daidate ")
        machine_data_hmpcrt_no_daidate_daitimes = machine_data_hmpcrt_no_daidate.daitime.values.tolist()
        
        if str(int(daidate)) in oCare_filedate_list:
            print(oCare_filedate_list.index(str(int(daidate))))
            oCare_data_hmpcrt_no_daidate = pd.read_csv(oCare_data_hmpcrt_no_filelist[oCare_filedate_list.index(str(int(daidate)))])
            for c in oCare_data_hmpcrt_no_daidate.columns:
                oCare_data_hmpcrt_no_daidate[c] = oCare_data_hmpcrt_no_daidate[c].astype(float)
                oCare_data_hmpcrt_no_daidate[c] = oCare_data_hmpcrt_no_daidate[c].astype(int)
            print(f'shape of oCare_data_hmpcrt_no_daidate: {oCare_data_hmpcrt_no_daidate.shape}')
#             oCare_data_hmpcrt_no_daidate_daitimes = oCare_data_hmpcrt_no_daidate.daitime.values.tolist()
        else:
            oCare_data_hmpcrt_no_daidate = pd.DataFrame(np.array([[0,0,0,0,0]]), columns=['hmpcrtno', 'daidate', 'daitime', 'SpO2', 'HR'])
        
        for point in range(len(machine_data_hmpcrt_no_daidate_daitimes)):
            
            if (machine_data_hmpcrt_no_daidate_daitimes[point] > min(bp_data_hmpcrt_no_daidate_daitimes)) & (machine_data_hmpcrt_no_daidate_daitimes[point] < max(bp_data_hmpcrt_no_daidate_daitimes)):
                # index = next(x[0] for x in enumerate(bp_data_hmpcrt_no_daidate_daitimes) if x[1] > machine_data_hmpcrt_no_daidate_daitimes[point]) - 1
                # index 不該 -1 因為要對照下一個時間點
                index = next(x[0] for x in enumerate(bp_data_hmpcrt_no_daidate_daitimes) if x[1] > machine_data_hmpcrt_no_daidate_daitimes[point])
                
            elif machine_data_hmpcrt_no_daidate_daitimes[point] <= min(bp_data_hmpcrt_no_daidate_daitimes):
                # index = -1
                # 對到第一個時間點的測量值
                index = 0
                
            elif machine_data_hmpcrt_no_daidate_daitimes[point] >= max(bp_data_hmpcrt_no_daidate_daitimes):
                # index = len(bp_data_hmpcrt_no_daidate_daitimes) - 1
                # 接下來的測量值未知
                index = -1
                
            else:
                print(hmpcrtno, daidate, machine_data_hmpcrt_no_daidate_daitimes[point], bp_data_hmpcrt_no_daidate_daitimes)

                
            if point == 0:
                period = oCare_data_hmpcrt_no_daidate[oCare_data_hmpcrt_no_daidate['daitime'] < machine_data_hmpcrt_no_daidate_daitimes[point]]
                print(period)
            else:
                period = oCare_data_hmpcrt_no_daidate[(oCare_data_hmpcrt_no_daidate['daitime'] > machine_data_hmpcrt_no_daidate_daitimes[point - 1] - 1) & (oCare_data_hmpcrt_no_daidate['daitime'] < machine_data_hmpcrt_no_daidate_daitimes[point])]
                print(period)
                

            HR_mean = np.mean(period.HR)
            HR_cv = (np.std(period.HR) / HR_mean)
            SpO2_mean = np.mean(period.SpO2)
            SpO2_cv = (np.std(period.SpO2) / SpO2_mean)
            print(HR_mean, HR_cv, SpO2_mean, SpO2_cv)

            
            #append
            ##machine_data
            to_append = machine_data_hmpcrt_no_daidate[machine_data_hmpcrt_no_daidate['daitime']==machine_data_hmpcrt_no_daidate_daitimes[point]].iloc[0, :].values.tolist()
            ##bp_data
            if index == -1:
                for v in bp_data_hmpcrt_no_daidate.iloc[index, 4:].values:
                    to_append.append(np.nan)
                to_append.append(bp_data_hmpcrt_no_daidate.iloc[index, 4]) # append systole from previous timing
            else:
                for v in bp_data_hmpcrt_no_daidate.iloc[index, 4:].values:
                    to_append.append(v)
                # append systole from previous timing
                if index == 0:
                    to_append.append(np.nan) 
                else:
                    to_append.append(bp_data_hmpcrt_no_daidate.iloc[index-1, 4])
            ##oCare_data
            for v in [HR_mean, HR_cv, SpO2_mean, SpO2_cv]:
                to_append.append(v)
                
            ##append index for check systole change
            to_append.append(index)
                
            collect[str(hmpcrtno)].append(to_append)


            
#output
    result = pd.DataFrame(collect[str(3376)], columns=machine_data_hmpcrt_no_daidate.columns.tolist()+bp_data_hmpcrt_no_daidate.columns.tolist()[4:]+['previous_systole', 'oCare_HR_mean', 'oCare_HR_cv', 'oCare_SpO2_mean', 'oCare_SpO2_cv', 'match_index'])
    print(result.head())

for hmpcrtno in hmpcrt_no[1:]:
    result = pd.concat([result, pd.DataFrame(collect[str(hmpcrtno)], columns=machine_data_hmpcrt_no_daidate.columns.tolist()+bp_data_hmpcrt_no_daidate.columns.tolist()[4:]+['previous_systole', 'oCare_HR_mean', 'oCare_HR_cv', 'oCare_SpO2_mean', 'oCare_SpO2_cv', 'match_index'])])


print(f'result: {result.shape}')
print(result.head())
print(result.isnull().sum())
result.to_csv(patient_aggregate_format_path, index=False)









