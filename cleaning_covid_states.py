import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import datetime
from scipy import interpolate

# Load data from Covid States
cs_df = pd.read_csv('./data/Health_Behavior.csv')

# FIPS for state (55: WI)
# state_of_interest = 55

# ctis_df = ctis_df[ctis_df.state_fips == state_of_interest]

# cs_df = cs_df[cs_df.StateFIPS == state_of_interest]

# Set the orgin for all time domains
day_zero=datetime.datetime(2020,4,1)

# add columns to COVID States dataframe maping start/end dates to days from April 1, 2020
cs_df['start_day'] = cs_df.Start_Date.apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')-day_zero).days)
cs_df['end_day'] = cs_df.End_Date.apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')-day_zero).days)


def cs_interpolater(df,state,behavior):

    # new date format to write in file
    start_dates = np.array([20200401, 20200501, 20200601, 20200701, 20200801, 20200901,
       20201001, 20201101, 20201201, 20210101, 20210201, 20210301,
       20210401, 20210501, 20210601, 20210701, 20210801, 20210901,
       20211001, 20211101, 20211201, 20220101, 20220201, 20220301,
       20220401, 20220501, 20220601])

    # new date format to write in file
    end_dates = np.array([20200430, 20200531, 20200630, 20200731, 20200831, 20200930,
       20201031, 20201130, 20201231, 20210131, 20210228, 20210331,
       20210430, 20210531, 20210630, 20210731, 20210831, 20210930,
       20211031, 20211130, 20211231, 20220131, 20220228, 20220331,
       20220430, 20220531, 20220630])

    # days since April 1, 2020
    date_ranges = [(0,30),(30,61),(61,91),(91,122),(122,153),(153,183),(183,214),(214,244),
        (244,275),(275,306),(306,334),(334,365),(365,395),(395,426),(426,456),(456,487),
        (487,518),(518,548),(548,579),(579,609),(609,640),(640,671),(671,699),(699,730),
        (730,760),(760,791),(791,821)]
    
    
    # array that stores each states 
    
    state_behavior_inters = np.zeros(len(start_dates))
    
    # load a specific states data
    df_ = df[df.State == state]
    # select out the start times
    x = df_.start_day.to_numpy()
        
    y = df_[behavior].to_numpy()

    f = interpolate.interp1d(x, y, kind=1, fill_value='extrapolate')

    xnew = np.arange(0, 1217, 1)
    ynew = f(xnew)   # use interpolation function returned by `interp1d`

    # new_behavior = np.zeros(len(date_ranges))
    
    for _,d_range in enumerate(date_ranges):
        val = (ynew[d_range[0]:d_range[1]].mean())
        if val < 0:
            val = 0
        if val > 100:
            val = 100
        state_behavior_inters[_] = val
    
    # for sdate, edate, val in zip(start_dates, end_dates, new_behavior):

        # print(state,'\t',sdate,'\t', edate, '\t', val, '\t')
    
    return state_behavior_inters


states = ['United States', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
          'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
 'Iowa',
 'Kansas',
 'Kentucky',
 'Louisiana',
 'Maine',
 'Maryland',
 'Massachusetts',
 'Michigan',
 'Minnesota',
 'Mississippi',
 'Missouri',
 'Montana',
 'Nebraska',
 'Nevada',
 'New Hampshire',
 'New Jersey',
 'New Mexico',
 'New York',
 'North Carolina',
 'North Dakota',
 'Ohio',
 'Oklahoma',
 'Oregon',
 'Pennsylvania',
 'Rhode Island',
 'South Carolina',
 'South Dakota',
 'Tennessee',
 'Texas',
 'Utah',
 'Vermont',
 'Virginia',
 'Washington',
 'West Virginia',
 'Wisconsin',
 'Wyoming']

fips= [ 0,  1,  2,  4,  5,  6,  8,  9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
       20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
       37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55,
       56]



def cs_state_df_generator(df,state,fips):
    behaviors=['Go to work', 'Go to the gym', 'Go visit a friend',
           'Go to a cafe, bar, or restaurant',
           'Go to a doctor or visit a hospital',
           'Go to church or another place of worship',
           'Take mass transit (e.g. subway, bus or train)',
           'Avoiding contact with other people',
           'Avoiding public or crowded places', 'Frequently washing hands',
           'Wearing a face mask when outside of your home',
           'Been in a room with someone outside of \n household in the past 24 hours',
           'Yes, 5-10 people', 'Yes, 11-50 people', 'Yes, 50 or more people']
    new_df = pd.DataFrame()
    # these dates are taken from CTIS since it has a smaller time range
    new_df['start_date'] = np.array([20200401, 20200501, 20200601, 20200701, 20200801, 20200901,
           20201001, 20201101, 20201201, 20210101, 20210201, 20210301,
           20210401, 20210501, 20210601, 20210701, 20210801, 20210901,
           20211001, 20211101, 20211201, 20220101, 20220201, 20220301,
           20220401, 20220501, 20220601],dtype=str)
    
    # these dates are taken from CTIS since it has a smaller time range
    new_df['end_date'] = np.array([20200430, 20200531, 20200630, 20200731, 20200831, 20200930,
           20201031, 20201130, 20201231, 20210131, 20210228, 20210331,
           20210430, 20210531, 20210630, 20210731, 20210831, 20210930,
           20211031, 20211130, 20211231, 20220131, 20220228, 20220331,
           20220430, 20220531, 20220630],dtype=str)
    # Days since April 1, 2020
    new_df['start_day'] = [0,30,61,91,122,153,183,214,244,275,306,334,365,395,426,456,487,518,
                            548,579,609,640,671,699,730,760,791]
    
    new_df['state'] = np.array(len(new_df)*[state])
    new_df['fips'] = np.array(len(new_df)*[fips])
    
    for behavior in behaviors:
        # print(cs_interpolater(df,state,behavior))
        new_df[behavior] = cs_interpolater(df,state,behavior)

    return new_df


new_cs_df = pd.DataFrame()
for state,fip in zip(states,fips):
    
    new_cs_df = pd.concat([new_cs_df,cs_state_df_generator(cs_df,state,fip)])

new_cs_df.to_csv('./data/Interpolated_COVID_States_Data.csv',index=False)



