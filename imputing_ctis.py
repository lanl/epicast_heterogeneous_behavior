import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# load the original CTIS data set
df = pd.read_csv('./data/monthly_state_all_indicators_agefull_gender.csv')

# ages and genders will be used to remove rows that are missing age or gender
# information. Ages, genders, and dates will be used to add any missing rows
ages = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75plus']
genders = ['Female', 'Male']
earliest_date = 20200901
latest_date = 20220601

# select only rows that are within the date range and have valid age and gender info
df = df[(df.period_start >= earliest_date) & (df.period_start <= latest_date) & (np.isin(df.agefull, ages)) & (np.isin(df.gender, genders))]

# reduce the orignal dataset to only the following columns
df_reduced = df[['period_start', 'state_fips', 'agefull', 'gender', 
                 'val_pct_wearing_mask_5d', 'val_pct_wearing_mask_7d', 
                 'val_pct_work_outside_home_1d', 'val_pct_work_outside_home_indoors_1d', 
                 'val_pct_avoid_contact', 'val_pct_avoid_contact_7d','val_pct_public_transit_1d',
                 'val_pct_spent_time_1d', 'val_pct_spent_time_indoors_1d',

                 'sample_size_pct_wearing_mask_5d', 'sample_size_pct_wearing_mask_7d', 
                 'sample_size_pct_work_outside_home_1d', 'sample_size_pct_work_outside_home_indoors_1d', 
                 'sample_size_pct_avoid_contact', 'sample_size_pct_avoid_contact_7d','sample_size_pct_public_transit_1d',
                 'sample_size_pct_spent_time_1d', 'sample_size_pct_spent_time_indoors_1d'
                ]]
# reset the index
df_reduced.index = range(len(df_reduced))

for fip in df.state_fips.unique():
    for start in np.array([20200901, 20201001, 20201101, 20201201, 20210101, 20210201, 20210301, 20210401, 20210501, 20210601, 20210701, 20210801,
                           20210901, 20211001, 20211101, 20211201, 20220101, 20220201,
                           20220301, 20220401, 20220501, 20220601]):
        for age in ages:
            for gender in genders:
            
                try:
                    df_reduced[(df_reduced.state_fips==fip) & (df_reduced.agefull == age) & (df_reduced.gender == gender) & (df_reduced.period_start ==start)].val_pct_wearing_mask_5d.iloc[0]
                except:
                    print(start, fip, age, gender)
                    df_reduced.loc[len(df_reduced)] = [start, fip, age, gender] + [np.nan]*18

df_reduced.sort_values(by=['period_start', 'state_fips', 'agefull', 'gender'],inplace=True)


df_reduced.index = range(len(df_reduced))

df_combined = pd.DataFrame()
df_combined['period_start'] = df_reduced.period_start
df_combined['year'] = df_reduced.period_start.apply(lambda x: int(str(x)[:4]))
df_combined['month'] = df_reduced.period_start.apply(lambda x: int(str(x)[4:6]))
df_combined['state_fips'] = df_reduced.state_fips
df_combined['agefull'] = df_reduced.agefull
df_combined['gender'] =  df_reduced.gender
df_combined['val_pct_wear_mask'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210101)]['val_pct_wearing_mask_5d'],df_reduced[(df_reduced.period_start>20210101) & (df_reduced.period_start<=20220601)]['val_pct_wearing_mask_7d']])
df_combined['sample_size_pct_wear_mask'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210101)]['sample_size_pct_wearing_mask_5d'],df_reduced[(df_reduced.period_start>20210101) & (df_reduced.period_start<=20220601)]['sample_size_pct_wearing_mask_7d']])

df_combined['val_pct_work_outside_home'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210201)]['val_pct_work_outside_home_1d'],df_reduced[(df_reduced.period_start>20210201) & (df_reduced.period_start<=20220601)]['val_pct_work_outside_home_indoors_1d']])
df_combined['sample_size_pct_work_outside_home'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210201)]['sample_size_pct_work_outside_home_1d'],df_reduced[(df_reduced.period_start>20210201) & (df_reduced.period_start<=20220601)]['sample_size_pct_work_outside_home_indoors_1d']])


df_combined['val_pct_avoid_contact'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210501)]['val_pct_avoid_contact'],df_reduced[(df_reduced.period_start>20210501) & (df_reduced.period_start<=20220601)]['val_pct_avoid_contact_7d']])
df_combined['sample_size_pct_avoid_contact'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210501)]['sample_size_pct_avoid_contact'],df_reduced[(df_reduced.period_start>20210501) & (df_reduced.period_start<=20220601)]['sample_size_pct_avoid_contact_7d']])


df_combined['val_pct_public_transit'] = df_reduced[(df_reduced.period_start>=10100901) & (df_reduced.period_start<=20220601)]['val_pct_public_transit_1d']
df_combined['sample_size_pct_public_transit'] = df_reduced[(df_reduced.period_start>=10100901) & (df_reduced.period_start<=20220601)]['sample_size_pct_public_transit_1d']

df_combined['val_pct_spend_time'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210201)]['val_pct_spent_time_1d'],df_reduced[(df_reduced.period_start>20210201) & (df_reduced.period_start<=20220601)]['val_pct_spent_time_indoors_1d']])
df_combined['sample_size_pct_spend_time'] = pd.concat([df_reduced[(df_reduced.period_start>=20200901) & (df_reduced.period_start<=20210201)]['sample_size_pct_spent_time_1d'],df_reduced[(df_reduced.period_start>20210201) & (df_reduced.period_start<=20220601)]['sample_size_pct_spent_time_indoors_1d']])


def impute_column(missing_df):
    '''
    It is assumed that the last column is the one being imputed
    '''
    
    X = missing_df.copy()

    # ordinal encoding for ages
    X[X=='18-24'] = 0
    X[X=='25-34'] = 1
    X[X=='35-44'] = 2
    X[X=='45-54'] = 3
    X[X=='55-64'] = 4
    X[X=='65-74'] = 5
    X[X=='75plus'] = 6

    # ordinal encoding for genders
    X[X=='Male'] = 0
    X[X=='Female'] = 1
    
    X = X.astype(float)
    
    imputer = IterativeImputer(RandomForestRegressor(verbose=0),verbose=10, max_value = 100, min_value = 0, tol=1e-3, max_iter=20, imputation_order='ascending')
    imp_X = imputer.fit_transform(X)
    
    
    return imp_X[:,5]


imp_mask = impute_column(df_combined[['year', 'month', 'state_fips', 'agefull', 'gender', 
                'val_pct_wear_mask']])

imp_work = impute_column(df_combined[['year', 'month', 'state_fips', 'agefull', 'gender', 
                'val_pct_work_outside_home']])

imp_transit = impute_column(df_combined[['year', 'month', 'state_fips', 'agefull', 'gender', 
                'val_pct_public_transit']])

imp_spend = impute_column(df_combined[['year', 'month', 'state_fips', 'agefull', 'gender', 
                'val_pct_spend_time']])

imp_avoid = impute_column(df_combined[['year', 'month', 'state_fips', 'agefull', 'gender', 
                'val_pct_avoid_contact']])


imp_df = pd.DataFrame()
imp_df['period_start'] = df_combined.period_start
imp_df['year'] = df_combined.year
imp_df['month'] = df_combined.month
imp_df['state_fips'] = df_combined.state_fips
imp_df['agefull'] = df_combined.agefull
imp_df['gender'] = df_combined.gender

imp_df['val_pct_wear_mask'] = imp_mask
imp_df['val_pct_spend_time'] = imp_spend
imp_df['val_pct_work_outside_home'] = imp_work
imp_df['val_pct_public_transit'] = imp_transit
imp_df['val_pct_avoid_contact'] = imp_avoid


imp_df['sample_size_pct_wear_mask'] = df_combined.sample_size_pct_wear_mask
imp_df['sample_size_pct_spend_time'] = df_combined.sample_size_pct_spend_time
imp_df['sample_size_pct_work_outside_home'] = df_combined.sample_size_pct_work_outside_home
imp_df['sample_size_pct_public_transit'] = df_combined.sample_size_pct_public_transit
imp_df['sample_size_pct_avoid_contact'] = df_combined.sample_size_pct_avoid_contact



imp_df.fillna(100).to_csv('./data/imputed_behaviors.csv',index=False)