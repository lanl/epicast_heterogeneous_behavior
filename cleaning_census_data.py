import pandas as pd
import numpy as np

census2021 = pd.read_csv('./data/cc-est2021-agesex-all.csv', encoding='latin-1') # 2: 2020 3: 2021
census2023 = pd.read_csv('./data/cc-est2023-agesex-all.csv', encoding='latin-1') # 2: 2020 3: 2021 4: 2022 5: 2023


missing_ct_4 = pd.concat([census2023[census2023.STATE!=9],census2021[census2021.STATE==9]], ignore_index=True)
temp_ct = census2021[(census2021.STATE==9)&(census2021.YEAR==3)]
temp_ct['YEAR']=[4]*len(temp_ct)
final_census_df = pd.concat([missing_ct_4,temp_ct], ignore_index=True)
final_census_df.to_csv('./data/cc-est2020-2023-agesex-all.csv',index=False)