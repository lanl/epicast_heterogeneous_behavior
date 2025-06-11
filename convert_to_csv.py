import numpy as np
import pandas as pd
import glob


fips_codes = np.array([1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
                    39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56])
start_dates = np.array([20200901, 20201001, 20201101, 20201201, 20210101, 20210201, 20210301, 20210401, 20210501, 20210601, 
                     20210701, 20210801, 20210901, 20211001, 20211101, 20211201, 20220101, 20220201, 20220301, 20220401,
                    20220501, 20220601])
ages = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75plus']
genders = ['Female', 'Male']


df_avoid = pd.DataFrame()

fip_data = []
cfip_data = []
date_data = []
age_data = []
gender_data = []
behavior_data = []
n_data = []

behavior = 'avoid_contact'
for fip in fips_codes:
    cfip_codes = [x.split('/')[-2] for x in glob.glob(f'IPF_results/{behavior}/{fip}/*/')]
    for cfip in cfip_codes:
        for date in start_dates:
        
            # print(fip, cfip, date)
            
            arr = np.load(f"IPF_results/{behavior}/{fip}/{cfip}/{date}.npy")  #gender (female,male) X age (0...6) X behavior (yes,no)
            females = arr[0]
            males = arr[1]

            for age,val,nmale in zip(ages,(males/np.atleast_2d(males.sum(axis=1)).T)[:,0],males.sum(axis=1)):
                # print(fip, cfip, age, 'male', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Male')
                behavior_data.append(val)
                n_data.append(nmale)
            
            for age,val,nfemale in zip(ages,(females/np.atleast_2d(females.sum(axis=1)).T)[:,0],females.sum(axis=1)):
                # print(fip, cfip, age, 'female', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Female')
                behavior_data.append(val)
                n_data.append(nfemale)
            

df_avoid['date'] = date_data
df_avoid['state_fip'] = fip_data
df_avoid['county_fip'] = cfip_data
df_avoid['gender'] = gender_data
df_avoid['age'] = age_data
df_avoid['N'] = n_data
df_avoid[behavior] = behavior_data

df_mask = pd.DataFrame()

fip_data = []
cfip_data = []
date_data = []
age_data = []
gender_data = []
behavior_data = []
n_data = []

behavior = 'wear_mask'
for fip in fips_codes:
    cfip_codes = [x.split('/')[-2] for x in glob.glob(f'IPF_results/{behavior}/{fip}/*/')]
    for cfip in cfip_codes:
        for date in start_dates:
        
            # print(fip, cfip, date)
            
            arr = np.load(f"IPF_results/{behavior}/{fip}/{cfip}/{date}.npy")  #gender (female,male) X age (0...6) X behavior (yes,no)
            females = arr[0]
            males = arr[1]

            for age,val,nmale in zip(ages,(males/np.atleast_2d(males.sum(axis=1)).T)[:,0],males.sum(axis=1)):
                # print(fip, cfip, age, 'male', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Male')
                behavior_data.append(val)
                n_data.append(nmale)
            
            for age,val,nfemale in zip(ages,(females/np.atleast_2d(females.sum(axis=1)).T)[:,0],females.sum(axis=1)):
                # print(fip, cfip, age, 'female', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Female')
                behavior_data.append(val)
                n_data.append(nfemale)
            

df_mask['date'] = date_data
df_mask['state_fip'] = fip_data
df_mask['county_fip'] = cfip_data
df_mask['gender'] = gender_data
df_mask['age'] = age_data
df_mask['N'] = n_data
df_mask[behavior] = behavior_data

df_time = pd.DataFrame()

fip_data = []
cfip_data = []
date_data = []
age_data = []
gender_data = []
behavior_data = []
n_data = []


behavior = 'spend_time'
for fip in fips_codes:
    cfip_codes = [x.split('/')[-2] for x in glob.glob(f'IPF_results/{behavior}/{fip}/*/')]
    for cfip in cfip_codes:
        for date in start_dates:
        
            # print(fip, cfip, date)
            
            arr = np.load(f"IPF_results/{behavior}/{fip}/{cfip}/{date}.npy")  #gender (female,male) X age (0...6) X behavior (yes,no)
            females = arr[0]
            males = arr[1]

            for age,val,nmale in zip(ages,(males/np.atleast_2d(males.sum(axis=1)).T)[:,0],males.sum(axis=1)):
                # print(fip, cfip, age, 'male', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Male')
                behavior_data.append(val)
                n_data.append(nmale)
            
            for age,val,nfemale in zip(ages,(females/np.atleast_2d(females.sum(axis=1)).T)[:,0],females.sum(axis=1)):
                # print(fip, cfip, age, 'female', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Female')
                behavior_data.append(val)
                n_data.append(nfemale)
            

df_time['date'] = date_data
df_time['state_fip'] = fip_data
df_time['county_fip'] = cfip_data
df_time['gender'] = gender_data
df_time['age'] = age_data
df_time['N'] = n_data
df_time[behavior] = behavior_data

df_transit = pd.DataFrame()

fip_data = []
cfip_data = []
date_data = []
age_data = []
gender_data = []
behavior_data = []
n_data = []

behavior = 'public_transit'
for fip in fips_codes:
    cfip_codes = [x.split('/')[-2] for x in glob.glob(f'IPF_results/{behavior}/{fip}/*/')]
    for cfip in cfip_codes:
        for date in start_dates:
        
            # print(fip, cfip, date)
            
            arr = np.load(f"IPF_results/{behavior}/{fip}/{cfip}/{date}.npy")  #gender (female,male) X age (0...6) X behavior (yes,no)
            females = arr[0]
            males = arr[1]

            for age,val,nmale in zip(ages,(males/np.atleast_2d(males.sum(axis=1)).T)[:,0],males.sum(axis=1)):
                # print(fip, cfip, age, 'male', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Male')
                behavior_data.append(val)
                n_data.append(nmale)
            
            for age,val,nfemale in zip(ages,(females/np.atleast_2d(females.sum(axis=1)).T)[:,0],females.sum(axis=1)):
                # print(fip, cfip, age, 'female', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Female')
                behavior_data.append(val)
                n_data.append(nfemale)
            

df_transit['date'] = date_data
df_transit['state_fip'] = fip_data
df_transit['county_fip'] = cfip_data
df_transit['gender'] = gender_data
df_transit['age'] = age_data
df_transit['N'] = n_data
df_transit[behavior] = behavior_data

df_work = pd.DataFrame()

fip_data = []
cfip_data = []
date_data = []
age_data = []
gender_data = []
behavior_data = []
n_data = []

behavior = 'work_outside_home'
for fip in fips_codes:
    cfip_codes = [x.split('/')[-2] for x in glob.glob(f'IPF_results/{behavior}/{fip}/*/')]
    for cfip in cfip_codes:
        for date in start_dates:
        
            # print(fip, cfip, date)
            
            arr = np.load(f"IPF_results/{behavior}/{fip}/{cfip}/{date}.npy")  #gender (female,male) X age (0...6) X behavior (yes,no)
            females = arr[0]
            males = arr[1]

            for age,val,nmale in zip(ages,(males/np.atleast_2d(males.sum(axis=1)).T)[:,0],males.sum(axis=1)):
                # print(fip, cfip, age, 'male', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Male')
                behavior_data.append(val)
                n_data.append(nmale)
            
            for age,val,nfemale in zip(ages,(females/np.atleast_2d(females.sum(axis=1)).T)[:,0],females.sum(axis=1)):
                # print(fip, cfip, age, 'female', val)
                fip_data.append(fip)
                cfip_data.append(cfip)
                date_data.append(date)
                age_data.append(age)
                gender_data.append('Female')
                behavior_data.append(val)
                n_data.append(nfemale)
            

df_work['date'] = date_data
df_work['state_fip'] = fip_data
df_work['county_fip'] = cfip_data
df_work['gender'] = gender_data
df_work['age'] = age_data
df_work['N'] = n_data
df_work[behavior] = behavior_data

df_tot = df_mask.copy()
df_tot['avoid_contact'] = df_avoid['avoid_contact']
df_tot['spend_time'] = df_time['spend_time']
df_tot['public_transit'] = df_transit['public_transit']
df_tot['work_outside_home'] = df_work['work_outside_home']
df_tot.N = df_tot.N.astype(int)
df_tot.rename(columns={"gender" : "sex"},inplace=True)

df_tot.to_csv('behavior_uptake.csv',index=False)