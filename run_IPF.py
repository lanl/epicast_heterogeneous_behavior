import numpy as np
import itertools
import pandas as pd
import os

def IPF_iterationNd(seed,marginals):
    '''
    1-step Iterative proportional fitting algorithm that will automatically detect dimensions
    '''
    # create a copy of the seed table
    X_copy = np.copy(seed)

    # grab the number of elements per dimension for iterating over later
    ranges = X_copy.shape

    # loop over each axis
    for dim in range(len(ranges)):
        # find the marginal of the current iteration of the seed table
        X_marginal = X_copy.sum(axis=dim)

        # this for loop will automatically create a nested for loop structure where each loop
        # corresponds to one of the axes and loops over the proper number of elements from ranges
        for indices in itertools.product(*[range(r) for r in ranges]):
            # this applies the update to the seed table
            X_copy[indices] = X_copy[indices]/X_marginal[*np.delete(indices, dim)]*marginals[dim][*np.delete(indices, dim)]
    return X_copy


covid_states_df = pd.read_csv('./data/Interpolated_COVID_States_Data.csv')
# country = pd.read_csv('./data/cc-est2023-agesex-all.csv',encoding='latin-1')
country = pd.read_csv('./data/cc-est2020-2023-agesex-all.csv',encoding='latin-1')


census_df = pd.DataFrame()
census_df['STATE'] = country['STATE']
census_df['COUNTY'] = country['COUNTY']
census_df['YEAR'] = country['YEAR']
census_df['18-24Male'] = country['AGE1824_MALE']
census_df['25-34Male'] = country['AGE2529_MALE'] + country['AGE3034_MALE']
census_df['35-44Male'] = country['AGE3539_MALE'] + country['AGE4044_MALE']
census_df['45-54Male'] = country['AGE4549_MALE'] + country['AGE5054_MALE']
census_df['55-64Male'] = country['AGE5559_MALE'] + country['AGE6064_MALE']
census_df['65-74Male'] = country['AGE6569_MALE'] + country['AGE7074_MALE']
census_df['75plusMale'] = country['AGE7579_MALE'] + country['AGE8084_MALE'] + country['AGE85PLUS_MALE']

census_df['18-24Female'] = country['AGE1824_FEM']
census_df['25-34Female'] = country['AGE2529_FEM'] + country['AGE3034_FEM']
census_df['35-44Female'] = country['AGE3539_FEM'] + country['AGE4044_FEM']
census_df['45-54Female'] = country['AGE4549_FEM'] + country['AGE5054_FEM']
census_df['55-64Female'] = country['AGE5559_FEM'] + country['AGE6064_FEM']
census_df['65-74Female'] = country['AGE6569_FEM'] + country['AGE7074_FEM']
census_df['75plusFemale'] = country['AGE7579_FEM'] + country['AGE8084_FEM'] + country['AGE85PLUS_FEM']

census_df['18-24'] = country['AGE1824_TOT']
census_df['25-34'] = country['AGE2529_TOT'] + country['AGE3034_TOT']
census_df['35-44'] = country['AGE3539_TOT'] + country['AGE4044_TOT']
census_df['45-54'] = country['AGE4549_TOT'] + country['AGE5054_TOT']
census_df['55-64'] = country['AGE5559_TOT'] + country['AGE6064_TOT']
census_df['65-74'] = country['AGE6569_TOT'] + country['AGE7074_TOT']
census_df['75plus'] = country['AGE7579_TOT'] + country['AGE8084_TOT'] + country['AGE85PLUS_TOT']

census_df['Male'] = country['POPEST_MALE'] - country['AGE1417_MALE'] - country['AGE513_MALE'] - country['UNDER5_MALE']
census_df['Female'] = country['POPEST_FEM'] - country['AGE1417_FEM'] - country['AGE513_FEM'] - country['UNDER5_FEM']

census_df['Total'] = census_df['Male'] + census_df['Female']


ctis_df = pd.read_csv('./data/imputed_behaviors.csv')


# THIS FILE NEEDS TO HAVE COLUMNS COMBINED
county_overall_raw = pd.read_csv('./data/monthly_county_theme_behavior_and_mental_health_overall.csv',dtype={'county_fips':str})
county_overall_raw['cfip'] = county_overall_raw['county_fips'].apply(lambda x: int(x[2:]))

county_overall = pd.DataFrame()
county_overall['period_start'] = county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20220601)].period_start
county_overall['year'] = county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20220601)].period_start.apply(lambda x: int(str(x)[:4]))
county_overall['month'] = county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20220601)].period_start.apply(lambda x: int(str(x)[4:6]))
county_overall['state_fips'] = county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20220601)].state_fips
county_overall['cfip'] = county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20220601)]['cfip']

county_overall['val_pct_wear_mask'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210101)]['val_pct_wearing_mask_5d'],county_overall_raw[(county_overall_raw.period_start>20210101) & (county_overall_raw.period_start<=20220601)]['val_pct_wearing_mask_7d']])
county_overall['sample_size_pct_wear_mask'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210101)]['sample_size_pct_wearing_mask_5d'],county_overall_raw[(county_overall_raw.period_start>20210101) & (county_overall_raw.period_start<=20220601)]['sample_size_pct_wearing_mask_7d']])

county_overall['val_pct_work_outside_home'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210201)]['val_pct_work_outside_home_1d'],county_overall_raw[(county_overall_raw.period_start>20210201) & (county_overall_raw.period_start<=20220601)]['val_pct_work_outside_home_indoors_1d']])
county_overall['sample_size_pct_work_outside_home'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210201)]['sample_size_pct_work_outside_home_1d'],county_overall_raw[(county_overall_raw.period_start>20210201) & (county_overall_raw.period_start<=20220601)]['sample_size_pct_work_outside_home_indoors_1d']])

county_overall['val_pct_avoid_contact'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210501)]['val_pct_avoid_contact'],county_overall_raw[(county_overall_raw.period_start>20210501) & (county_overall_raw.period_start<=20220601)]['val_pct_avoid_contact_7d']])
county_overall['sample_size_pct_avoid_contact'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210501)]['sample_size_pct_avoid_contact'],county_overall_raw[(county_overall_raw.period_start>20210501) & (county_overall_raw.period_start<=20220601)]['sample_size_pct_avoid_contact_7d']])

county_overall['val_pct_public_transit'] = county_overall_raw[(county_overall_raw.period_start>=10100901) & (county_overall_raw.period_start<=20220601)]['val_pct_public_transit_1d']
county_overall['sample_size_pct_public_transit'] = county_overall_raw[(county_overall_raw.period_start>=10100901) & (county_overall_raw.period_start<=20220601)]['sample_size_pct_public_transit_1d']

county_overall['val_pct_spend_time'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210201)]['val_pct_spent_time_1d'],county_overall_raw[(county_overall_raw.period_start>20210201) & (county_overall_raw.period_start<=20220601)]['val_pct_spent_time_indoors_1d']])
county_overall['sample_size_pct_spend_time'] = pd.concat([county_overall_raw[(county_overall_raw.period_start>=20200901) & (county_overall_raw.period_start<=20210201)]['sample_size_pct_spent_time_1d'],county_overall_raw[(county_overall_raw.period_start>20210201) & (county_overall_raw.period_start<=20220601)]['sample_size_pct_spent_time_indoors_1d']])


fips_codes = np.array([1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
                    39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56])
dates = np.array([20200901, 20201001, 20201101, 20201201, 20210101, 20210201,
       20210301, 20210401, 20210501, 20210601, 20210701, 20210801,
       20210901, 20211001, 20211101, 20211201, 20220101, 20220201,
       20220301, 20220401, 20220501, 20220601])

genders=['Female','Male']
ages=['18-24','25-34','35-44','45-54','55-64','65-74','75plus']

epsilon=1e-6

urb_rur_df = pd.read_excel('./data/2020_UA_COUNTY.xlsx', sheet_name=0)
urb_rur_df = urb_rur_df[np.isin(urb_rur_df.STATE,fips_codes)]
urb_rur_df.sort_values(by=['STATE', 'COUNTY'],inplace=True)
urb_rur_df.index=range(len(urb_rur_df))



for col_of_insterst, cs_col_of_interest in [('wear_mask', 'Wearing a face mask when outside of your home'), ('work_outside_home', 'Go to work'), ('avoid_contact', 'Avoiding contact with other people'), ('public_transit', 'Take mass transit (e.g. subway, bus or train)'), ('spend_time','Been in a room with someone outside of \n household in the past 24 hours')]:
    print(f'Behavior:{col_of_insterst}')
    ## THIS IS PRODUCING AVERAGES FOR URB OR RURAL COUNTIES BY STATE FOR A GIVEN BEHAVE
    urban_rural_state_average = pd.DataFrame()

    st_fip = []
    urb_pct = []
    rur_pct = []
    av_date = []

    for fip in fips_codes:
        for date in dates:
            st_fip.append(fip)
            av_date.append(date)
            
            # get all of the county fips for the state
            cfips = census_df[(census_df.YEAR==2)&(census_df.STATE==fip)].COUNTY.to_numpy()
            # get the urban and rural county fips for the state
            urban_counties = urb_rur_df[(urb_rur_df.STATE == fip)&(urb_rur_df.POPPCT_URB>.5)].COUNTY.unique()
            rural_counties = urb_rur_df[(urb_rur_df.STATE == fip)&(urb_rur_df.POPPCT_RUR>.5)].COUNTY.unique()
        
            # select from the CTIS data countys that are urban or rural
            urban_county_data = county_overall[(county_overall.period_start==date)&(county_overall.state_fips==fip)&(np.isin(county_overall.cfip,urban_counties))]['val_pct_'+col_of_insterst].mean()
            rural_county_data = county_overall[(county_overall.period_start==date)&(county_overall.state_fips==fip)&(np.isin(county_overall.cfip,rural_counties))]['val_pct_'+col_of_insterst].mean()
            
            # print(fip,date)
            # print(urban_county_data,rural_county_data, covid_states_df[(covid_states_df.start_date==date)&(covid_states_df.fips==fip)]['Wearing a face mask when outside of your home'].iloc[0])
            # print(fip,len(cfips),len(urban_counties),len(rural_counties))
            urb_pct.append(urban_county_data)
            rur_pct.append(rural_county_data)
    urban_rural_state_average['state_fips'] = st_fip
    urban_rural_state_average['period_start'] = av_date
    urban_rural_state_average['urb_pct'] = urb_pct
    urban_rural_state_average['rur_pct'] = rur_pct





    county_issue = 0
    gender_issue = 0
    age_issue = 0

    for FIP in fips_codes:
    # for FIP in [13]:

        # cfips = census_df[(census_df.STATE==FIP) & (census_df.YEAR==YEAR)].COUNTY.unique()
        cfips = census_df[(census_df.STATE==FIP)].COUNTY.unique()
        for CFIP in cfips:
            os.makedirs(f'./IPF_results/{col_of_insterst}/{FIP}/{CFIP}/',exist_ok=True)
            
        # os.makedirs(f'./IPF_results/{col_of_interest}/{FIP}/county/',exist_ok=True)
        # print(FIP)
        for DATE in dates:
        # for DATE in [20210701]:
            # print(DATE)
            # use the correct year's census data
            if str(DATE)[:4] == '2020':
                YEAR = 2
            elif str(DATE)[:4] == '2021':
                YEAR = 3
            elif str(DATE)[:4] == '2022':
                YEAR = 4
            elif str(DATE)[:4] == '2023':
                YEAR = 5
            else:
                print("something is wrong")

            
            county_age_behavior = np.zeros((len(cfips), 7, 2))
            county_gender_behavior = np.zeros((len(cfips), 2, 2))
            county_gender_age = np.zeros((len(cfips), 2, 7))


            
            
            # COVID STATES
            # perc_mask = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)]['Wearing a face mask when outside of your home'].iloc[0]/100

            ###############
            # create gender by age marginal for each county (county number, gender[Female, Male], age)
            for ci, c in enumerate(cfips):
                for ai, a in enumerate(ages):
                    for gi, g in enumerate(genders):
                        county_gender_age[ci,gi,ai] = census_df[(census_df.STATE==FIP)&(census_df.YEAR==YEAR)&(census_df.COUNTY==c)][a+g].iloc[0]
            ###############



            # Generating percent mask for county while maintaining differences
            ##############
            percent_pop_county = county_gender_age.sum(axis=1).sum(axis=1)/county_gender_age.sum()

            diff_county_wear_mask = np.zeros(len(percent_pop_county)-1)

            for i in range(len(cfips)-1):
                
                try:
                    # county_next = county_overall[(county_overall.period_start==DATE) & (county_overall.state_fips ==FIP) & (county_overall.cfip==cfips[i+1])].val_pct_wearing_mask_5d.iloc[0]/100
                    county_next = county_overall[(county_overall.period_start==DATE) & (county_overall.state_fips ==FIP) & (county_overall.cfip==cfips[i+1])]['val_pct_'+col_of_insterst].iloc[0]/100
                    int(county_next)
                    
                except:
                    try:
                        if urb_rur_df[(urb_rur_df.STATE==FIP)&(urb_rur_df.COUNTY==cfips[i+1])].POPPCT_URB.iloc[0]>.5:
                            county_next = urban_rural_state_average[(urban_rural_state_average.state_fips==FIP)&(urban_rural_state_average.period_start==DATE)].urb_pct.iloc[0]/100
                            int(county_next)
                        if urb_rur_df[(urb_rur_df.STATE==FIP)&(urb_rur_df.COUNTY==cfips[i+1])].POPPCT_URB.iloc[0]<.5:
                            county_next = urban_rural_state_average[(urban_rural_state_average.state_fips==FIP)&(urban_rural_state_average.period_start==DATE)].rur_pct.iloc[0]/100
                            int(county_next)
                        
                    except:
                        # print(fip,cfip,date)
                        # county_next = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)]['Wearing a face mask when outside of your home'].iloc[0]/100
                        county_next = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)][cs_col_of_interest].iloc[0]/100
                        # int(county_next)
                        
            
                try:
                    # county_curr = county_overall[(county_overall.period_start==DATE) & (county_overall.state_fips ==FIP) & (county_overall.cfip==cfips[i])].val_pct_wearing_mask_5d.iloc[0]/100
                    county_curr = county_overall[(county_overall.period_start==DATE) & (county_overall.state_fips ==FIP) & (county_overall.cfip==cfips[i])]['val_pct_'+col_of_insterst].iloc[0]/100
                    int(county_curr)
                except:
                    try:
                        if urb_rur_df[(urb_rur_df.STATE==FIP)&(urb_rur_df.COUNTY==cfips[i])].POPPCT_URB.iloc[0]>.5:
                            county_curr = urban_rural_state_average[(urban_rural_state_average.state_fips==FIP)&(urban_rural_state_average.period_start==DATE)].urb_pct.iloc[0]/100
                            int(county_curr)
                        if urb_rur_df[(urb_rur_df.STATE==FIP)&(urb_rur_df.COUNTY==cfips[i])].POPPCT_URB.iloc[0]<.5:
                            county_curr = urban_rural_state_average[(urban_rural_state_average.state_fips==FIP)&(urban_rural_state_average.period_start==DATE)].rur_pct.iloc[0]/100
                            int(county_curr)
                    except:
                        # print(fip,cfip,date)
                        # county_curr = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)]['Wearing a face mask when outside of your home'].iloc[0]/100
                        county_curr = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)][cs_col_of_interest].iloc[0]/100
                        # int(county_curr)
                # print(county_curr,county_next)
                diff_county_wear_mask[i] = county_next-county_curr



            
            tot = 0
            for i in range(len(cfips)-1):
                tot += (percent_pop_county[i+1]*diff_county_wear_mask[list(range(i+1))].sum())
            
            # cs = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)]['Wearing a face mask when outside of your home'].iloc[0]/100
            cs = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)][cs_col_of_interest].iloc[0]/100
            county_mask_vals = np.zeros(len(cfips))
            county_mask_vals[0] = cs - tot
            for i in range(1,len(cfips)):
                county_mask_vals[i] = county_mask_vals[i-1] + diff_county_wear_mask[i-1]


            # YOU ADDED THIS MAKE SURE IT IS WHAT YOU WANT
            # if you can't maintain differences in coutnies reset to use CS for all
            if np.any(county_mask_vals<0) or np.any(county_mask_vals>1):
                # print('cant handle')
                county_issue += 1
                county_mask_vals = cs*np.ones(len(cfips))
            # county_mask_vals[county_mask_vals<0] = 0
            ##############


            

            

            ###############
            # create gender by behavior marginal for each county (county number, gender[Female, Male], mask [yes,no])
            for ci, c in enumerate(cfips):
                perc_mask = np.copy(county_mask_vals[ci])
                # try:
                    # perc_mask = county_overall[(county_overall.period_start==DATE)&(county_overall.cfip==c)&(county_overall.state_fips==FIP)].val_pct_wearing_mask_5d.iloc[0]/100
                    # int(perc_mask)
                    # print('county')
                # except:
                    # perc_mask = covid_states_df[(covid_states_df.fips==FIP)&(covid_states_df.start_date==DATE)]['Wearing a face mask when outside of your home'].iloc[0]/100
                    # print('state')
            
                percent_female, percent_male = county_gender_age[ci].sum(axis=1)/county_gender_age[ci].sum() # percent females and males
                
                # calculate the difference
                # diff_female_male_wear_mask = (ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.gender=='Female')].val_pct_wearing_mask_5d.mean() - ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.gender=='Male')].val_pct_wearing_mask_5d.mean())/100
                diff_female_male_wear_mask = (ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.gender=='Female')]['val_pct_'+col_of_insterst].mean() - ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.gender=='Male')]['val_pct_'+col_of_insterst].mean())/100
                # construct percentages so that we preserve this difference
                u = (perc_mask - diff_female_male_wear_mask*percent_female)
                v = u+diff_female_male_wear_mask
                gender_mask_percents = np.array([v,u]) #np.array([66.4944875753551, 66.32154236121224])#

                # check for negative percentages
                if np.any(gender_mask_percents<0) or np.any(gender_mask_percents>1):
                    gender_issue+=1
                    gender_mask_percents = perc_mask*np.ones(2)
            
                county_gender_behavior[ci,0] = [gender_mask_percents[0]*county_gender_age[ci,0,:].sum(), (1-gender_mask_percents[0])*county_gender_age[ci,0,:].sum()]
                county_gender_behavior[ci,1] = [gender_mask_percents[1]*county_gender_age[ci,1,:].sum(), (1-gender_mask_percents[1])*county_gender_age[ci,1,:].sum()]
            ###############

            ###############
            # create the age by behavior marginal for each county (county number, age, behavior [yes,no])
            for ci, c in enumerate(cfips):

                perc_mask = np.copy(county_mask_vals[ci])
                    
                percent_ages = county_gender_age[ci].sum(axis=0)/county_gender_age[ci].sum()
            
                # array to store empical differences in mask wearing by age
                diff_age_wear_mask = np.zeros(len(ages)-1)
                for i in range(len(ages)-1):
                
                    # what percent of the older age class wears a mask
                    # older = ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.agefull==ages[i+1])].val_pct_wearing_mask_5d.mean()
                    older = ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.agefull==ages[i+1])]['val_pct_'+col_of_insterst].mean()
                    # what percent of the younger age class wears a mask
                    # younger = ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.agefull==ages[i])].val_pct_wearing_mask_5d.mean()
                    younger = ctis_df[(ctis_df.period_start==DATE) & (ctis_df.state_fips == FIP) & (ctis_df.agefull==ages[i])]['val_pct_'+col_of_insterst].mean()
                
                    # for each difference store the difference that the older versus younger age classes wear masks
                    diff_age_wear_mask[i] = (older-younger)/100
                
                # to calculate the weighting for the first age class, subtract from measured effets of all others
                # a0 = z - y1x1 - (y1+y2)x2 - ...
                tot = 0
                for i in range(6):
                    tot += (percent_ages[i+1]*diff_age_wear_mask[list(range(i+1))].sum())
                
                
                # each of these are the weights each age class needs in order for the total to match covid states
                a0 = perc_mask - tot
                a1 = a0 + diff_age_wear_mask[0]
                a2 = a1 + diff_age_wear_mask[1]
                a3 = a2 + diff_age_wear_mask[2]
                a4 = a3 + diff_age_wear_mask[3]
                a5 = a4 + diff_age_wear_mask[4]
                a6 = a5 + diff_age_wear_mask[5]
                
                age_mask_percents = np.array([a0,a1,a2,a3,a4,a5,a6])
                
                # check for negative percetages
                if np.any(age_mask_percents<0) or np.any(age_mask_percents>1):
                    age_issue += 1
                    age_mask_percents = perc_mask*np.ones(7)
                # print(age_mask_percents,perc_mask)
                
                # Constructing the gender age marginal
                for age_i, age in enumerate(ages):
                
                    county_age_behavior[ci,age_i, 0] = (age_mask_percents[age_i])*census_df[(census_df.STATE==FIP) & (census_df.YEAR==YEAR)&(census_df.COUNTY==c)][age].iloc[0]
                    county_age_behavior[ci,age_i, 1] = (1-age_mask_percents[age_i])*census_df[(census_df.STATE==FIP) & (census_df.YEAR==YEAR)&(census_df.COUNTY==c)][age].iloc[0]

                    
            ###############
            

            # State-level IPF
            ###############
            # create the seed table for the state by summing marginal data
            seed_table = np.zeros((2,7,2)) # gender (female,male) X age (0...6) X behavior (yes,no)
            for ai, a in enumerate(ages):
                for gi, g in enumerate(genders):
                    # seed_table[gi,ai,0] = ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)].val_pct_wearing_mask_5d.iloc[0]/100 * ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)].sample_size_pct_wearing_mask_5d.iloc[0]
                    seed_table[gi,ai,0] = ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)]['val_pct_'+col_of_insterst].iloc[0]/100 * ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)]['sample_size_pct_'+col_of_insterst].iloc[0]
                    # seed_table[gi,ai,1] = (1 - ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)].val_pct_wearing_mask_5d.iloc[0]/100) * ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)].sample_size_pct_wearing_mask_5d.iloc[0]
                    seed_table[gi,ai,1] = (1 - ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)]['val_pct_'+col_of_insterst].iloc[0]/100) * ctis_df[(ctis_df.state_fips==FIP)&(ctis_df.period_start==DATE)&(ctis_df.gender==g)&(ctis_df.agefull==a)]['sample_size_pct_'+col_of_insterst].iloc[0]
            
            # Run IPF for the state, summing each above marginal along the county dimensions (axis 0)
            
            res = np.copy(seed_table) + 1e-10
            for iteration in range(50):
                res = IPF_iterationNd(res,[county_age_behavior.sum(axis=0), county_gender_behavior.sum(axis=0), county_gender_age.sum(axis=0)])
            
                axis_0_error = np.linalg.norm((county_age_behavior.sum(axis=0) - res.sum(axis=0)).flatten(),2)
                axis_1_error = np.linalg.norm((county_gender_behavior.sum(axis=0) - res.sum(axis=1)).flatten(),2)
                axis_2_error = np.linalg.norm((county_gender_age.sum(axis=0) - res.sum(axis=2)).flatten(),2)
                
                # print(f'Iteration: {iteration}, Axis 0 error = {axis_0_error:.5f}, Axis 1 error = {axis_1_error:.5f}, Axis 2 error = {axis_2_error:.5f}')
                
                if (axis_0_error < epsilon) and (axis_1_error < epsilon) and (axis_2_error < epsilon):
                    print(f'FIP: {FIP}, Date: {DATE}, State IPF converged')
                    break
            if iteration == 49:
                print(f'FIP: {FIP}, Date: {DATE}, State IPF failed, {axis_0_error,axis_1_error,axis_2_error}')
            
            np.save(f'./IPF_results/{col_of_insterst}/{FIP}/{DATE}',res)
            # print()
            ###############

            # County-level IPF
            ###############
            seed_table = np.ones((len(cfips),2,7,2))
            gender_age_behavior = np.copy(res)


            epsilon=1e-6
            new_res = np.copy(seed_table)
            for iteration in range(50):
                new_res = IPF_iterationNd(new_res,[gender_age_behavior, county_age_behavior+1e-10, county_gender_behavior+1e-10, county_gender_age+1e-10])
            
                axis_0_error = np.linalg.norm((gender_age_behavior - new_res.sum(axis=0)).flatten(),2)
                axis_1_error = np.linalg.norm((county_age_behavior - new_res.sum(axis=1)).flatten(),2)
                axis_2_error = np.linalg.norm((county_gender_behavior - new_res.sum(axis=2)).flatten(),2)
                axis_3_error = np.linalg.norm((county_gender_age - new_res.sum(axis=3)).flatten(),2)
                
                # print(f'Iteration: {iteration}, Axis 0 error = {axis_0_error:.5f}, Axis 1 error = {axis_1_error:.5f}, Axis 2 error = {axis_2_error:.5f}, Axis 3 error = {axis_3_error:.5f}')
            
                if (axis_0_error < epsilon) and (axis_1_error < epsilon) and (axis_2_error < epsilon) and (axis_2_error < epsilon):
                    print(f'FIP: {FIP}, Date: {DATE}, County IPF converged')
                    break
            if iteration == 49:
                print(f'FIP: {FIP}, Date: {DATE}, County IPF failed, {axis_0_error,axis_1_error,axis_2_error,axis_3_error}')

            for ci,CFIP in enumerate(cfips):
                np.save(f'./IPF_results/{col_of_insterst}/{FIP}/{CFIP}/{DATE}',new_res[ci])
            
            # print()
            # print(new_res.min())