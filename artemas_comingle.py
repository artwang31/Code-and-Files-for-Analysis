#!/usr/bin/env python
# coding: utf-8

# In[1]:

# # Addressable EOC Co-Mingle Script
# This script brings together impression and reach data from up to 4 MVPDs - Verizon, Comcast, Spectrum and Cox, standardizes the data, merges it and then outputs it as one formatted file. Will output one internal file, which breaks down metrics by MVPD, as well as one external facing file that includes a total number for the full Ampersand order.

# ### Bring in libraries needed

# In[2]:

import os
import pandas as pd
import numpy as np
import datetime as dt322
from datetime import datetime
import warnings
from argparse import ArgumentParser
from os import path


# ### define brand and whether this is a Matterkind daypart definition report

# In[4]:

parser = ArgumentParser()
parser.add_argument("--brand", dest="brand", required = True, help="Folder where files are")
parser.add_argument("--mk_dp", dest="mk_dp", required = False, default="0", help="Is the request for a Matterkind Daypart definitions? 1/0. Default is 0.")
args = parser.parse_args()

brand = getattr(args,'brand')
mk = getattr(args,'mk_dp')


# ### Matterkind Daypart
# - Daytime - 9AM-4PM
# - Early Fringe - 4PM-8PM
# - Early Morning - 6AM-9AM
# - Late Fringe – 12AM-2AM
# - Overnight – 2AM-6AM
# - Prime - 8PM-12AM
# 
# ### Standard Dayparts
# (Defined by Spectrum; all other files use hours to define dayparts)
# - Overnight = Midnight-6AM
# - Early Morning = 6AM-9AM
# - Daytime = 9AM-4PM
# - Early Fringe = 4PM-6PM
# - Prime Access = 6PM-8PM
# - Prime = 8PM-11PM
# - Late Fringe = 11PM-12AM

# # Define paths and create mapping data
# Individual files can include MVPD name in the file name, or be in separate folders by MVPD. Separate Reach / Frequency files for Spectrum and Comcast are necessary, and they need to include "Reach" in their file name, or be in the reach/frequecy folder and include "Paralell" for comcast and "Spectrum" for the Spectrum file.

# In[21]:

#define paths
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)
files = [c for c in os.listdir(brand) if (c.lower().find('~$')==-1) & (c.lower().find('ds_store')==-1) & (c.lower().find('xls')!=-1)]
if len(files)>0:
    com_path = brand #'files/comcast'
    sp_path = brand #'files/spectrum'
    cx_path = brand #'files/cox'
    vz_path = brand #'files/verizon'
    rf_path = brand #'files/reach_freq'
    new_com_path = brand #'files/audienceaddressable'  #ADW
    comcast_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('comcast')!=-1) & (c.lower().find('reach')==-1)]
    spectrum_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('spectrum')!=-1) & (c.lower().find('reach')==-1)]
    sp_all_files = spectrum_files
    cox_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('cox')!=-1) & (c.lower().find('reach')==-1)]
    verizon_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('verizon')!=-1) & (c.lower().find('reach')==-1)]
    rf_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('reach')!=-1)]
    new_com_files = [c for c in files if (c.lower().find('ds_store')==-1) & (c.lower().find('audienceaddressable')!=-1) & (c.lower().find('reach')==-1) & (c.lower().find('xls')!=-1)] #ADW
else:
    com_path = 'files/comcast'
    sp_path = 'files/spectrum'
    cx_path = 'files/cox'
    vz_path = 'files/verizon'
    rf_path = 'files/reach_freq'
    new_com_path = 'files/audienceaddressable' #ADW
    #create file lists
    comcast_files = [c for c in os.listdir(com_path) if (c.lower().find('ds_store')==-1)]
    spectrum_files = [c for c in os.listdir(sp_path) if (c.lower().find('ds_store')==-1)]
    sp_all_files = spectrum_files
    cox_files = [c for c in os.listdir(cx_path) if (c.lower().find('ds_store')==-1)]
    verizon_files = [c for c in os.listdir(vz_path) if (c.lower().find('ds_store') == -1)]
    rf_files = [c for c in os.listdir(rf_path) if (c.lower().find('ds_store')==-1)]
    new_com_files = [c for c in os.listdir(new_com_path) if (c.lower().find('ds_store')==-1)] #ADW
## BRING IN mapping/cleaning files, and create mapping dataframes
#Networks
net_map = pd.read_csv('/Users/artemasw/Ampersand_20-21/test_networks.csv', encoding = "ISO-8859-1") #read in network mapping
net_map = net_map[['Network', 'corrected_network']] #limit to cols needed
net_map.columns = ['feature_value', 'corrected_network']
#DMAs
dma_map = pd.read_csv('/Users/artemasw/Ampersand_20-21/full_dma_mapping.csv')
dma_map = dma_map[['Geography', 'corrected_dma']]
dma_map.columns = ['feature_value', 'corrected_dma']
dma_map['feature_value'] = dma_map['feature_value'].str.lower() #make lowercase
dma_map = dma_map.drop_duplicates() #drop duplicates
#create daypart mapping dataframe
daypart_map = pd.DataFrame({'feature_value': ['Overnight','Early Morning', 'Daytime','Early Fringe','Prime Access', 'Prime', 'Prime Time','Late Fringe', 'Access', 'AM Day', 'PM Day'],
                   'new_fv': ['Overnight - Midnight-6AM', 'Early Morning - 6AM-9AM', 'Daytime - 9AM-4PM','Early Fringe - 4PM-6PM',
 'Prime Access - 6PM-8PM', 'Prime - 8PM-11PM', 'Prime - 8PM-11PM','Late Fringe - 11PM-12AM', 'Prime Access - 6PM-8PM', 'Daytime - 9AM-4PM', 'Daytime - 9AM-4PM']})
mk_dp_map = pd.DataFrame({'Daypart': ['DA', 'EF', 'EM', 'LF', 'ON', 'PR'], 
                          'feature_value': ['Daytime - 9AM-4PM', 'Early Fringe - 4PM-8PM', 'Early Morning - 6AM-9AM', 'Late Fringe - 12AM-2AM', 'Overnight – 2AM-6AM', 'Prime - 8PM-12AM']})
#create day of week mapping dataframe.
dow_map = pd.DataFrame({'day_num': [0,1,2,3,4,5,6],
                   'feature_value': ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']})

da_map = pd.DataFrame({'day_abc': ['Mon', 'Tue', 'Wed','Thur', 'Fri', 'Sat', 'Sun', 'Thu'], 
                      'feature_value': ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday', 'Thursday']})


# ## New_Comcast Artemas Wang added to original co-mingle script

# In[8]:
if len(new_com_files)>0:
    new_com_all = pd.ExcelFile(new_com_path+'/'+new_com_files[0])
    new_com_df_data = pd.read_excel(new_com_all)

    # subsetting data needed from original file
    newcom_networks = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Network') &
                                                  (new_com_df_data['Dimension Value'] != 'Total'), ['Dimension Name','Dimension Value','Impressions']])
    newcom_market = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Market') &
                                                  (new_com_df_data['Dimension Value'] != 'Total'), ['Dimension Name','Dimension Value','Impressions']])
    newcom_day = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Day of Week') &
                                                  (new_com_df_data['Dimension Value'] != 'Total'), ['Dimension Name','Dimension Value','Impressions','Frequency']])
    newcom_daypart = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Hour') &
                                              (new_com_df_data['Dimension Value'] != 'Total'), ['Dimension Name','Dimension Value','Impressions']])
    newcom_daypart['Impressions'] = newcom_daypart['Impressions'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    newcom_daypart['Dimension Value'] = newcom_daypart['Dimension Value'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    newcom_week = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Week') &
                                                  (new_com_df_data['Dimension Value'] != 'Total'), ['Dimension Name','Dimension Value','Impressions','Frequency']])
    newcom_summary = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Campaign') &
                                                  (new_com_df_data['Dimension Value'] == 'Total'), ['Dimension Name','Dimension Value','Impressions']])
    newcom_campaign = pd.DataFrame(new_com_df_data.loc[(new_com_df_data['Dimension Name'] == 'Campaign') &
                                              (new_com_df_data['Dimension Value'] != 'Total'), ['Impressions','Uniques']])

    # creating reach and frequency to calculate later
    newcom_campaign['MVPD'] = 'Comcast'
    newcom_campaign.rename(columns = {'Impressions':'Total Impressions','Uniques':'Total HHs'}, inplace = True)


    # merging networks from original data and corrected network data
    network_merged1 = pd.merge(newcom_networks, net_map, left_on='Dimension Value', right_on = ['feature_value'], how = 'left')
    network_merged1.columns = ['data_type','Dimension Valule','Impressions','vlook','feature_value']
    newcom_network = network_merged1[['data_type','feature_value', 'Impressions']]
    newcom_network.loc[pd.isnull(newcom_network.feature_value), 'feature_value'] = 'Unassigned'
    newcom_network = newcom_network.groupby(['feature_value', 'Impressions', 'data_type']).sum().reset_index()

    # adding dma code to dma
    newcom_map = pd.read_csv('/Users/artemasw/Ampersand_20-21/full_dma_mapping.csv')
    newcom_map = newcom_map[['Geography', 'corrected_dma','dma_code']]
    newcom_map['dma_code'] = newcom_map['dma_code'].apply(str)
    newcom_map['dma_and_code'] = newcom_map['corrected_dma'] + "|" + newcom_map['dma_code']
    full_mapping = pd.read_csv('/Users/artemasw/Ampersand_20-21/full_dma_mapping.csv')
    full_mapping = full_mapping[['corrected_dma','dma_code']]

    # conducting string split to create new new column
    newcom_market['City'] = newcom_market['Dimension Value'].str.split()
    newcom_market['City'] = newcom_market['Dimension Value'].str.split(', ', expand = True)

    # merging markets from original data and corrected market data
    dma_merged = pd.merge(newcom_market, newcom_map, left_on = 'City', right_on = ['Geography'], how = 'left')

    # selecting columns we want
    dma_merged = dma_merged[['Dimension Name','dma_and_code', 'Impressions']]
    dma_merged.columns = ['data_type','feature_value','Impressions']
    newcom_market = dma_merged[['data_type','feature_value', 'Impressions']]
    newcom_market = newcom_market.drop_duplicates()

    # string split a column to create dma_code
    newcom_market[['feature_value', 'dma_code']] = newcom_market['feature_value'].str.split('|', 1, expand = True)
    dma_values = newcom_market[['feature_value','dma_code']]
    newcom_market = newcom_market[['data_type','feature_value','Impressions']]

    # cleaning day of week
    # dropping unnecessary columns
    newcom_day = newcom_day.drop(['Frequency'], axis=1)
    newcom_day['Dimension Name'] = 'Day'
    newcom_day.columns = ['data_type','feature_value','Impressions']

    # cleaning dayparts
    if mk=='1': 
        conditions = [
            (newcom_daypart['Dimension Value'] < 2),
            (newcom_daypart['Dimension Value'] >= 2) & (newcom_daypart['Dimension Value'] <= 5),
            (newcom_daypart['Dimension Value'] > 5) & (newcom_daypart['Dimension Value'] <= 8),
            (newcom_daypart['Dimension Value'] > 8) & (newcom_daypart['Dimension Value'] <= 15),
            (newcom_daypart['Dimension Value'] > 15) & (newcom_daypart['Dimension Value'] <= 19),
            (newcom_daypart['Dimension Value'] > 19) & (newcom_daypart['Dimension Value'] <= 23),
            ]
        values = ['Late Fringe - 12AM-2AM', 'Overnight – 2AM-6AM','Early Morning - 6AM-9AM',
                     'Daytime - 9AM-4PM','Early Fringe - 4PM-8PM','Prime - 8PM-12AM']
        newcom_daypart['feature_value'] = np.select(conditions, values)
    else:
        conditions = [
            (newcom_daypart['Dimension Value'] <= 5),
            (newcom_daypart['Dimension Value'] > 5) & (newcom_daypart['Dimension Value'] <= 8),
            (newcom_daypart['Dimension Value'] > 8) & (newcom_daypart['Dimension Value'] <= 15),
            (newcom_daypart['Dimension Value'] > 15) & (newcom_daypart['Dimension Value'] <= 17),
            (newcom_daypart['Dimension Value'] > 17) & (newcom_daypart['Dimension Value'] <= 19),
            (newcom_daypart['Dimension Value'] > 19) & (newcom_daypart['Dimension Value'] <= 22),
            (newcom_daypart['Dimension Value'] > 22)
            ]
        values = ['Overnight - Midnight-6AM', 'Early Morning - 6AM-9AM', 'Daytime - 9AM-4PM', 'Early Fringe - 4PM-6PM',
                  'Prime Access - 6PM-8PM','Prime - 8PM-11PM','Late Fringe - 11PM-12AM']
        newcom_daypart['feature_value'] = np.select(conditions, values)
    newcom_daypart.columns = ['data_type','Dimension Valule','Impressions','feature_value']
    newcom_daypart = newcom_daypart[['data_type','feature_value', 'Impressions']]
    newcom_daypart['data_type'] = 'Daypart'

    # cleaning week
    # dropping unnecessary columns
    newcom_week = newcom_week.drop(['Frequency'], axis=1)
    newcom_week.columns = ['data_type','feature_value','Impressions']

    # cleaning summary
    # dropping unnecessary columns
    newcom_summary['Dimension Name'] = 'Summary'   
    newcom_summary.columns = ['data_type','feature_value','Impressions']

    # combining dataframes into list
    newcom_df = [newcom_summary, newcom_network, newcom_market, newcom_day, newcom_daypart, newcom_week]

    # creating columns for brand and mvpd
    ni = 0
    while ni < len (newcom_df):
        newcom_df[ni]['brand'] = brand
        newcom_df[ni]['mvpd'] = 'Comcast'
        ni+=1

    # appending all files so it is long
    art_df = pd.DataFrame()
    if len(new_com_df_data) > 0:
        # New Comcast
        art_df = art_df.append(newcom_week, sort=False).append(newcom_day, sort=False).append(newcom_network, sort=False).append(newcom_market, sort=False).append(newcom_daypart, sort=False).append(newcom_summary, sort=False)

    # writing to csv for preview
    art_df.to_csv(r'/Users/artemasw/Ampersand_20-21/Addressable_Comingled_EOC/art_df.csv', index = False)

# ## Comcast

# In[ ]:

if len(comcast_files)>0:
    #LOOP 1: Create dictionary for all comcast files and sheets files = {} #create dictionary for all files
    ### Read Comcast Files into a dictionary
    #Create a dictionary for all files, and then for each file, a dictionary for the sheets. This will allow us to use one file to then create a set of dataframes for each of the types of data sets we need (Summary, Network, Week, Hour (for daypart), Creative and DMA.
    files = {}
    fi = 0
    while fi<len(comcast_files):
        xl = pd.ExcelFile(com_path+'/'+comcast_files[fi])
        sheets = xl.sheet_names
        #read in all sheets for the file
        dfs = {}
        si = 0
        while si<len(sheets):
            #read in data
            raw_data = pd.read_excel(xl, sheet_name=sheets[si])
            #look for header row, as first row with any data
            i=0
            for i, row in raw_data.iterrows():
                if row.notnull().all():
                    data = raw_data.iloc[(i+1):].reset_index(drop=True)
                    data.columns = raw_data.iloc[i].values
                    break
            #transform to numeric where possible
            for c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='ignore')
            #drop NaN rows, as well as remove the total row
            data = data[~(data.iloc[:,0].astype(str).str.contains("Total", na=False))].dropna(axis=1, how='all')
            data = data.dropna()
            data['file'] = comcast_files[fi]
            dfs.update({sheets[si]: data})
            si=si+1
        files.update({comcast_files[fi]: dfs})
        fi+=1
    #LOOP 2: Create dataframes for Comcast files from Dictionary: Creates dataframes for Summary, Network, week, hour (for Daypart creation), Creative and DMAs
    com_summary = pd.DataFrame()
    com_network = pd.DataFrame()
    com_dma = pd.DataFrame()
    com_week = pd.DataFrame()
    com_dp_hour = pd.DataFrame()
    com_weekhour = pd.DataFrame()
    com_day = pd.DataFrame()
    com_creative = pd.DataFrame()
    fi = 0
    while fi<len([*files]):
        com_summary = com_summary.append(files[[*files][fi]][sheets[0]], sort=False).reset_index(drop=True)
        com_network = com_network.append(files[[*files][fi]][sheets[1]], sort=False).reset_index(drop=True)
        com_dma = com_dma.append(files[[*files][fi]][sheets[2]], sort=False).reset_index(drop=True)
        com_week = com_week.append(files[[*files][fi]][sheets[3]], sort=False).reset_index(drop=True)
        com_dp_hour = com_dp_hour.append(files[[*files][fi]][sheets[4]], sort=False).reset_index(drop=True)
        com_weekhour = com_weekhour.append(files[[*files][fi]][sheets[5]], sort=False).reset_index(drop=True)
        com_day = com_day.append(files[[*files][fi]][sheets[6]], sort=False).reset_index(drop=True)
        com_creative = com_creative.append(files[[*files][fi]][sheets[7]], sort=False).reset_index(drop=True)
        fi+=1
    #subloop 2: Create dayparts
    com_dp_hour['Display By'] = com_dp_hour['Display By'].astype(str).str[-8:] #save the hours as a string
    com_dp_hour['Display By'] = pd.to_datetime('2020-01-01 '+(com_dp_hour['Display By']), format='%Y-%m-%d %H/%M', errors='ignore') #create date with time strin
    com_dp_hour.index = pd.DatetimeIndex(com_dp_hour['Display By']) #set time as index
    #set the times by hours
    if mk=='1': 
        com_dp_hour.loc['2020-01-01 02:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight – 2AM-6AM'
        com_dp_hour.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        com_dp_hour.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        com_dp_hour.loc['2020-01-01 16:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Early Fringe - 4PM-8PM'
        com_dp_hour.loc['2020-01-01 20:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Prime - 8PM-12AM'
        com_dp_hour.loc['2020-01-01 00:00:00': '2020-01-01 01:00:00', 'Daypart'] = 'Late Fringe - 12AM-2AM'
    else:
        com_dp_hour.loc['2020-01-01 00:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight - Midnight-6AM'
        com_dp_hour.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        com_dp_hour.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        com_dp_hour.loc['2020-01-01 16:00:00': '2020-01-01 17:00:00', 'Daypart'] = 'Early Fringe - 4PM-6PM'
        com_dp_hour.loc['2020-01-01 18:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Prime Access - 6PM-8PM'
        com_dp_hour.loc['2020-01-01 20:00:00': '2020-01-01 22:00:00', 'Daypart'] = 'Prime - 8PM-11PM'
        com_dp_hour.loc['2020-01-01 23:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Late Fringe - 11PM-12AM'
    com_dp_hour = com_dp_hour.reset_index(drop=True) #remove the unnessary index
    com_dp_hour['brand'] = brand
    com_dp_hour['Impressions'] = com_dp_hour['Impressions'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    com_daypart = com_dp_hour[['Daypart', 'Impressions', 'brand']].groupby(['Daypart','brand']).sum().reset_index() #create final dayparts dataframe
    com_daypart.columns = ['feature_value', 'brand', 'Impressions']
    com_daypart['mvpd'] = 'Comcast'
    com_daypart['data_type'] = 'Daypart'
    ### LOOP 3: Clean up and group, limiting to Impressions
    #This part creates a list of DataFrames, then loops through them and creates a cleaner version of the brand, then creates a grouped version of the data by all values except the Impressions (brand and also network, week, creative and DMA). Then it puts those cleaned dataframes back into the original names
    comcast_dfs = [com_summary,
    com_network,
    com_week,
    com_creative, 
    com_dma,
    com_day]
    df_name = ['Summary', 'Network', 'Week', 'Creative', 'Market', 'Day']
    ci = 0
    while ci<len(comcast_dfs):
        comcast_dfs[ci]['brand'] = brand
        # limit to columns needed
        if ci==0:
            #summary
            comcast_dfs[ci] = comcast_dfs[ci].iloc[:,[5,8]].dropna()
            
        elif ci==6:
            comcast_dfs[ci] = comcast_dfs[ci].iloc[:,[0,4,8]]
        else:
            comcast_dfs[ci] = comcast_dfs[ci].iloc[:,[0,3,6]]
        cols = comcast_dfs[ci].columns
        cols = list(map(lambda x: x.strip(), cols)) #remove excess spaces
        cols = [c for c in cols if c!='Impressions']
        comcast_dfs[ci]['Impressions'] = comcast_dfs[ci]['Impressions'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        comcast_dfs[ci] = comcast_dfs[ci].groupby(cols).sum().reset_index()
        cols = comcast_dfs[ci].columns.values
        cols = list(map(lambda x: x if x!='Impressions' else 'Impressions', cols))
        cols = cols[1:]
        cols.insert(0,'feature_value')
        comcast_dfs[ci].columns = cols
        comcast_dfs[ci]['mvpd'] = 'Comcast'
        comcast_dfs[ci]['data_type'] = df_name[ci]
        ci+=1
    com_summary = comcast_dfs[0]
    com_network = comcast_dfs[1]
    com_week = comcast_dfs[2]
    com_creative = comcast_dfs[3]
    com_dma = comcast_dfs[4]
    com_day = comcast_dfs[5]
    com_summary.columns = ['brand', 'Impressions', 'mvpd','data_type']
    com_summary.insert(0,'feature_value', 'Total')
    #Comcast Cleanup - map to network and dma files
    com_network=com_network.merge(net_map, how='left', on=['feature_value'])
    com_unmatched_networks = com_network[pd.isnull(com_network.corrected_network)]
    com_network = com_network[['corrected_network', 'brand', 'Impressions', 'mvpd', 'data_type']]
    com_network.columns = ['feature_value','brand', 'Impressions', 'mvpd', 'data_type']
    com_network.loc[pd.isnull(com_network.feature_value), 'feature_value'] = 'Unassigned'
    com_network = com_network.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    com_network = com_network[['feature_value', 'brand','Impressions', 'mvpd', 'data_type']]
    com_dma['feature_value'] = com_dma['feature_value'].str.lower()
    com_dma=com_dma.merge(dma_map, how='left', on=['feature_value'])
    com_unmatched_dmas = com_dma[pd.isnull(com_dma.corrected_dma)]
    com_dma = com_dma[['corrected_dma', 'brand', 'Impressions', 'mvpd', 'data_type']]
    com_dma.columns = ['feature_value','brand', 'Impressions', 'mvpd', 'data_type']
    com_dma.loc[pd.isnull(com_dma.feature_value), 'feature_value'] = 'Unassigned'
    com_dma = com_dma.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    com_dma = com_dma[['feature_value','brand', 'Impressions', 'mvpd', 'data_type']]
    if len(com_unmatched_dmas)>0 or len(com_unmatched_networks)>0:
        print('Warning! Unmapped Networks or Markets! Exported as unmapped_Comcast_'+brand+'.xlsx')
        writer = pd.ExcelWriter('unmapped_Comcast_'+brand+'.xlsx', engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        com_unmatched_dmas.to_excel(writer, sheet_name='DMAs', index=False)
        com_unmatched_networks.to_excel(writer, sheet_name='Networks', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


# ## Spectrum

# In[ ]:


#Spectrum
#This portion of the script reads the Spectrum excel files in the files/pdfs folder relative to the script.
if len(spectrum_files)>0:
    sp_hourly_file = [c for c in sp_all_files if (c.lower().find('hour')!=-1)]
    spectrum_files = [c for c in spectrum_files if (c.lower().find('hour')==-1)]
    #FIRST LOOP - create dictionary for all Spectrum files and sheets.
    sp_files = {} #create dictionary for all sp_files
    fi = 0
    while fi<len(spectrum_files):
        sp_xl = pd.ExcelFile(sp_path+'/'+spectrum_files[fi])
        sp_sheets = sp_xl.sheet_names
        #read in all sheets for the file
        dfs = {}
        si = 0
        while si<len(sp_sheets):
            #read in data
            data = pd.read_excel(sp_xl, sheet_name=sp_sheets[si])
            #transform to numeric where possible
            #data = raw_data
            for c in data.columns:
                if c.lower().find('date')==-1: 
                    data[c] = pd.to_numeric(data[c], errors='ignore')
            #drop NaN rows, as well as remove the total row
            data = data[~(data.iloc[:,0].astype(str).str.contains("Total", na=False))].dropna(axis=1, how='all') #first remove total row and any columns with missing data
            data = data.dropna() #remove rows with missing data
            data['file'] = spectrum_files[fi]
            dfs.update({sp_sheets[si]: data})
            si+=1
        sp_files.update({spectrum_files[fi]: dfs})
        fi+=1
    #SECOND LOOP - Create dataframes for necessary parts
    sp_daily = pd.DataFrame()
    sp_dma = pd.DataFrame()
    sp_creative = pd.DataFrame()
    sp_network = pd.DataFrame()
    sp_daypart = pd.DataFrame()
    sp_day = pd.DataFrame()
    so = pd.DataFrame(sp_sheets, columns=['sheet_name']).reset_index()
    daily_num = so[so['sheet_name'].str.lower().str.contains('daily|date')]['index'].values[0]
    dma_num = so[so['sheet_name'].str.lower().str.contains('geo|market|dma')]['index'].values[0]
    nets_num = so[so['sheet_name'].str.lower().str.contains('net')]['index'].values[0]
    try:
        hour_num = so[so['sheet_name'].str.lower().str.contains('hour')]['index'].values[0]
    except:
        pass
    dp_num = so[so['sheet_name'].str.lower().str.contains('daypart')]['index'].values[0]
    dow_num = so[so['sheet_name'].str.lower().str.contains('day of week')]['index'].values[0]
    try:
        creative_num = so[so['sheet_name'].str.lower().str.contains('creative')]['index'].values[0]
    except:
        pass
    fi = 0
    while fi<len([*sp_files]):
        sp_daily = sp_daily.append(sp_files[[*sp_files][fi]][sp_sheets[daily_num]], sort=False).reset_index(drop=True)
        sp_dma = sp_dma.append(sp_files[[*sp_files][fi]][sp_sheets[dma_num]], sort=False).reset_index(drop=True)
        sp_creative = sp_creative.append(sp_files[[*sp_files][fi]][sp_sheets[creative_num]], sort=False).reset_index(drop=True)
        sp_network = sp_network.append(sp_files[[*sp_files][fi]][sp_sheets[nets_num]], sort=False).reset_index(drop=True)
        sp_daypart = sp_daypart.append(sp_files[[*sp_files][fi]][sp_sheets[dp_num]], sort=False).reset_index(drop=True)
        sp_day = sp_day.append(sp_files[[*sp_files][fi]][sp_sheets[dow_num]], sort=False).reset_index(drop=True)
        fi+=1
    cols_drop = ['DMA Code', 'Boundary Type']
    if cols_drop[0] in sp_dma.columns:
        sp_dma.drop(sp_dma[[cols_drop[0]]],axis=1,inplace=True)
    if cols_drop[1] in sp_dma.columns:
        sp_dma.drop(sp_dma[[cols_drop[1]]],axis=1,inplace=True)      
#THIRD LOOP: add brand, limit columns, clean up column names (trim and update any Impressions names with Spectrum Impression, make sure all numbers are saves as numeric)
    spectrum_dfs = [sp_network,
    sp_creative,
    sp_daypart,
    sp_dma,
    sp_day]
    df_name = ['Network', 'Creative', 'Daypart', 'Market', 'Day']
    ci = 0
    while ci<len(spectrum_dfs):
        spectrum_dfs[ci]['brand'] = brand
        spectrum_dfs[ci] = spectrum_dfs[ci].iloc[:,[0,1,-1]]
        cols = spectrum_dfs[ci].columns
        cols = list(map(lambda x: x.strip(), cols)) #remove excess spaces
        spectrum_dfs[ci].columns = cols
        cols = [c for c in cols if c!='Impressions']
        spectrum_dfs[ci]['Impressions'] = spectrum_dfs[ci]['Impressions'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        spectrum_dfs[ci] = spectrum_dfs[ci].groupby(cols).sum().reset_index()
        cols = spectrum_dfs[ci].columns.values
        cols = list(map(lambda x: x if x!='Impressions' else 'Impressions', cols))
        cols = cols[1:]
        cols.insert(0,'feature_value')
        spectrum_dfs[ci].columns = cols
        spectrum_dfs[ci]['mvpd'] = 'Spectrum'
        spectrum_dfs[ci]['data_type'] = df_name[ci]
        ci+=1
    sp_network = spectrum_dfs[0]
    sp_creative = spectrum_dfs[1]
    sp_daypart = spectrum_dfs[2]
    sp_dma = spectrum_dfs[3]
    sp_day = spectrum_dfs[4]
    #dayparts mapping
    if mk=='1':
        sp_daypart.columns = ['Daypart', 'brand', 'Impressions', 'mvpd', 'data_type']
        sp_daypart = sp_daypart.merge(mk_dp_map, how='left', on=['Daypart'])
        sp_daypart = sp_daypart[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
        sp_daypart.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    else:
        sp_daypart = sp_daypart.merge(daypart_map, how='left', on=['feature_value'])
        sp_daypart = sp_daypart[['new_fv', 'brand', 'Impressions', 'mvpd', 'data_type']]
        sp_daypart.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    #if there is an hourly file, use it to create the hourly info
    if len(sp_hourly_file)>0:
        sp_xl = pd.ExcelFile(sp_path+'/'+sp_hourly_file[0]) #read in excel file
        sp_sheets = sp_xl.sheet_names #excel sheet names
        sp_hr_df = pd.read_excel(sp_xl, skiprows=1) #read excel from dataframe 
        cols = sp_hr_df.columns
        #create list of columns for each element
        dp_cols = [c for c in cols if (c.lower().find('daypart')!=-1)]
        start_cols = [c for c in cols if (c.lower().find('start')!=-1)]
        end_cols = [c for c in cols if (c.lower().find('end')!=-1)]
        imps_cols = [c for c in cols if (c.lower().find('impressions')!=-1)]
        imps_cols = imps_cols[-len(dp_cols):] #select only the impressions columns in the correct portions
        #create dataframes for each version of columns
        dp = sp_hr_df[dp_cols].dropna() 
        st = sp_hr_df[start_cols].dropna()
        end = sp_hr_df[end_cols].dropna()
        imps = sp_hr_df[imps_cols].dropna()
        #loop through however many iterations there are and create a full dataframe for each version, then append it to the final sp_dp_hourly dataframe
        i = 0 
        col_num = len(dp_cols)
        sp_dp_hour = pd.DataFrame()
        while i<col_num: 
            tmp_dp = dp.iloc[:,i]
            tmp_st = st.iloc[:,i]
            tmp_end = end.iloc[:,i]
            tmp_imps = imps.iloc[:,i]
            tmp_df = pd.DataFrame(data=[tmp_dp, tmp_st, tmp_end, tmp_imps]).transpose()
            tmp_df.columns = ['Daypart', 'Start', 'End', 'Impressions']
            sp_dp_hour = sp_dp_hour.append(tmp_df)
            i+=1
        #clean up by adding impressions together
        sp_dp_hour = sp_dp_hour.groupby(['Daypart', 'Start', 'End']).sum().reset_index()  
        sp_dp_hour = sp_dp_hour.merge(mk_dp_map, how='left', on=['Daypart'])
        sp_dp_hour['Impressions'] = sp_dp_hour['Impressions'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        sp_dp_hour['brand'] = brand
        sp_daypart = sp_dp_hour[['feature_value', 'brand', 'Impressions']].groupby(['feature_value','brand']).sum().reset_index() #create final dayparts dataframe
        sp_daypart['mvpd'] = 'Spectrum'
        sp_daypart['data_type'] = 'Daypart'        
    #create summary from daily data
    if ('brand' in sp_daily)==False:
        sp_daily.insert(1,'brand', brand)
        sp_day = sp_day[sp_day.feature_value!='Total']
        sp_daily = sp_daily[sp_daily.Date!='Total']
        sp_summary = sp_daily[['brand', 'Impressions']].groupby(['brand']).sum().reset_index()
        sp_summary.columns = ['brand', 'Impressions']
        sp_summary['mvpd'] = 'Spectrum'
        sp_summary['feature_value'] = 'Total'
        sp_summary['data_type'] = 'Summary'
        sp_summary = sp_summary[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    #Spectrum Cleanup - map to network and dma files
    sp_network = sp_network[~(sp_network.feature_value=='Total')]
    sp_network=sp_network.merge(net_map, how='left', on=['feature_value'])
    sp_unmatched_networks = sp_network[pd.isnull(sp_network.corrected_network)]
    sp_network = sp_network[['corrected_network', 'brand', 'Impressions', 'mvpd', 'data_type']]
    sp_network.columns = ['feature_value','brand', 'Impressions', 'mvpd', 'data_type']
    sp_network.loc[pd.isnull(sp_network.feature_value), 'feature_value'] = 'Unassigned'
    sp_network = sp_network.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    sp_network = sp_network[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    sp_dma = sp_dma[~(sp_dma.feature_value=='Total')]
    sp_dma['feature_value'] = sp_dma['feature_value'].str.lower()
    sp_dma=sp_dma.merge(dma_map, how='left', on=['feature_value'])
    sp_unmatched_dmas = sp_dma[pd.isnull(sp_dma.corrected_dma)]
    sp_dma = sp_dma[['corrected_dma', 'brand', 'Impressions', 'mvpd', 'data_type']]
    sp_dma.columns = ['feature_value','brand', 'Impressions', 'mvpd', 'data_type']
    sp_dma.loc[pd.isnull(sp_dma.feature_value), 'feature_value'] = 'Unassigned'
    sp_dma = sp_dma.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    sp_dma = sp_dma[['feature_value','brand', 'Impressions', 'mvpd', 'data_type']]
    if len(sp_unmatched_dmas)>0 or len(sp_unmatched_networks)>0:
        print('Warning! Unmapped Networks or Markets! Exported as unmapped_Spectrum_'+brand+'.xlsx')
        writer = pd.ExcelWriter('unmapped_Spectrum_'+brand+'.xlsx', engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        sp_unmatched_dmas.to_excel(writer, sheet_name='DMAs', index=False)
        sp_unmatched_networks.to_excel(writer, sheet_name='Networks', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


# # Cox

# In[ ]:


if len(cox_files) > 0:
    ###LOOP 1: Read Cox Files into a dictionary
    # Create a dictionary for all files, and then for each file, a dictionary for the sheets. This will allow us to use one file to then create a set of dataframes for each of the types of data sets we need (Summary, Network, Week, Hour (for daypart), Creative and DMA.
    cx_files = {}  # create dictionary for all cx_files
    fi = 0
    while fi < len(cox_files):
        cx_xl = pd.ExcelFile(cx_path+'/'+cox_files[fi])
        cx_sheets = cx_xl.sheet_names
        # read in all sheets for the file
        dfs = {}
        si = 0
        while si < len(cx_sheets):
            # read in data
            data = pd.read_excel(cx_xl, sheet_name=cx_sheets[si])
            # transform to numeric where possible
            for c in data.columns:
                if c.lower().find('date')==-1: 
                    data[c] = pd.to_numeric(data[c], errors='ignore')
            # drop NaN rows, as well as remove the total row
            data = data[~(data.iloc[:,0].astype(str).str.contains("Total", na=False))].dropna(axis=1, how='all')
            data = data.dropna()
            #replace impressions column names with "Impressions" so that they'll be found
            cols = data.columns
            if len([c for c in cols if (c.lower().find('impression')!=-1)|(c.lower().find('imp')!=-1)])==1:
                txt_replace = [c for c in cols if (c.lower().find('impression')!=-1)|(c.lower().find('imp')!=-1)][0]
                new_cols = ['Impressions' if x==txt_replace else x for x in cols]
                data.columns = new_cols
            elif len([c for c in cols if (c.lower().find('net counted ads')!=-1)|(c.lower().find('imp')!=-1)])==1:
                txt_replace = [c for c in cols if (c.lower().find('net counted ads')!=-1)|(c.lower().find('imp')!=-1)][0]
                new_cols = ['Impressions' if x==txt_replace else x for x in cols]
                data.columns = new_cols
            elif len([c for c in cols if ((c.lower().find('impression')!=-1)|(c.lower().find('imp')!=-1)) & (c.lower().find('deliver')!=-1)])==1:
                txt_replace = [c for c in cols if ((c.lower().find('impression')!=-1)|(c.lower().find('imp')!=-1)) & (c.lower().find('deliver')!=-1)][0]
                new_cols = ['Impressions' if x==txt_replace else x for x in cols]
                data.columns = new_cols
            elif cx_sheets[si].lower().find('reach')!=-1: 
                pass
            else: 
                print('Missing impressions in tab: '+cx_sheets[si]+' for file '+ cox_files[fi])
            data['file'] = cox_files[fi]
            dfs.update({cx_sheets[si]: data})

            si += 1
        cx_files.update({cox_files[fi]: dfs})
        fi += 1
    ###LOOP 2:Create dataframes for Cox files from Dictionary
    # Creates dataframes for Summary, Network, week, hour (for Daypart creation), Creative and DMAs
    cx_summary = pd.DataFrame()
    cx_network = pd.DataFrame()
    cx_week = pd.DataFrame()
    cx_dp_hour = pd.DataFrame()
    cx_creative = pd.DataFrame()
    cx_dma = pd.DataFrame()
    cx_weekhour = pd.DataFrame()
    cx_daily = pd.DataFrame()
    cx_rf = pd.DataFrame()
    so = pd.DataFrame(cx_sheets, columns=['sheet_name']).reset_index()
    try:
        daily_num = so[(so['sheet_name'].str.lower().str.contains('daily|date'))]['index'].values[0]
    except:
        print('Cox Daily tab missing or mis-named. Must include the words "date" or "daily".')
        pass
    try:
        dma_num = so[so['sheet_name'].str.lower().str.contains('geo|market|dma')]['index'].values[0]
    except:
        print('Cox Markets tab missing or mis-named - must include the phrase "geo"')
        pass
    try:
        nets_num = so[so['sheet_name'].str.lower().str.contains('net')]['index'].values[0]
    except:
        print('Cox Networks tab missing or mis-named - must include the phrase "net"')
        pass
    try:
        hour_num = so[so['sheet_name'].str.lower().str.contains('hour')]['index'].values[0]
    except: 
        print('Cox Hourly tab missing or mis-named - must include the phrase "hour"')
    try:
        creative_num = so[so['sheet_name'].str.lower().str.contains('creative')]['index'].values[0]
    except:
        pass
    try:
        rf_num = so[so['sheet_name'].str.lower().str.contains('reach')]['index'].values[0]
    except:
        print('Cox Reach tab missing or mis-named - must include the phrase "reach"')
    fi = 0
    # CHECK SHEET NAMES / ORDER above; use that list to update numbers for each df. ex: if Geo is second, use the number 1 for creating cx_dma.
    # ['Delivery','Reach-Frequency', 'Daily', 'Hourly', 'Geo', 'VOD Network']
    #Reach-Frequency (2), Daily (3), Hourly (4), Geo (6), Network (7)
    while fi < len([*cx_files]):
        #cx_summary = cx_summary.append(cx_files[[*cx_files][fi]][cx_sheets[0], sort=False).reset_index(drop=True)
        cx_network = cx_network.append(cx_files[[*cx_files][fi]][cx_sheets[nets_num]], sort=False).reset_index(drop=True)
        cx_dma = cx_dma.append(cx_files[[*cx_files][fi]][cx_sheets[dma_num]], sort=False).reset_index(drop=True)
        #cx_week = cx_week.append(cx_files[[*cx_files][fi]][cx_sheets[2]], sort=False).reset_index(drop=True)
        cx_dp_hour = cx_dp_hour.append(cx_files[[*cx_files][fi]][cx_sheets[hour_num]], sort=False).reset_index(drop=True)
        # cx_weekhour = cx_weekhour.append(cx_files[[*cx_files][fi]][cx_sheetss[5]], sort=False).reset_index(drop=True)
        cx_daily = cx_daily.append(cx_files[[*cx_files][fi]][cx_sheets[daily_num]], sort=False).reset_index(drop=True)
        #cx_creative = cx_creative.append(cx_files[[*cx_files][fi]][cx_sheets[4]], sort=False).reset_index(drop=True)
        cx_rf = cx_rf.append(cx_files[[*cx_files][fi]][cx_sheets[rf_num]], sort=False).reset_index(drop=True)
        fi += 1
        ##create daypart from hours
    cx_dp_hour[cx_dp_hour.columns[0]] = cx_dp_hour[cx_dp_hour.columns[0]].astype(str).str[-8:]  # save the hours as a string
    cx_dp_hour[cx_dp_hour.columns[0]] = pd.to_datetime('2020-01-01 ' + (cx_dp_hour[cx_dp_hour.columns[0]]),format='%Y-%m-%d %H/%M',errors='ignore')  # create date with time string
    cx_dp_hour.index = pd.DatetimeIndex(cx_dp_hour[cx_dp_hour.columns[0]])  # set time as index
    # set the times by hours
    if mk=='1': 
        cx_dp_hour.loc['2020-01-01 02:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight – 2AM-6AM'
        cx_dp_hour.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        cx_dp_hour.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        cx_dp_hour.loc['2020-01-01 16:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Early Fringe - 4PM-8PM'
        cx_dp_hour.loc['2020-01-01 20:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Prime - 8PM-12AM'
        cx_dp_hour.loc['2020-01-01 00:00:00': '2020-01-01 01:00:00', 'Daypart'] = 'Late Fringe - 12AM-2AM'
    else:
        cx_dp_hour.loc['2020-01-01 00:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight - Midnight-6AM'
        cx_dp_hour.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        cx_dp_hour.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        cx_dp_hour.loc['2020-01-01 16:00:00': '2020-01-01 17:00:00', 'Daypart'] = 'Early Fringe - 4PM-6PM'
        cx_dp_hour.loc['2020-01-01 18:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Prime Access - 6PM-8PM'
        cx_dp_hour.loc['2020-01-01 20:00:00': '2020-01-01 22:00:00', 'Daypart'] = 'Prime - 8PM-11PM'
        cx_dp_hour.loc['2020-01-01 23:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Late Fringe - 11PM-12AM'
    cx_dp_hour = cx_dp_hour.reset_index(drop=True)  # remove the unnessary index
    new = cx_dp_hour['file'].str.split(" ", n=2, expand=True)  # split the file to get the brand name
    # cx_dp_hour['brand'] = new[0]+' '+new[1] #set the new brand name
    cx_dp_hour['brand'] = brand
    cx_daypart = cx_dp_hour[['Daypart', 'Impressions', 'brand']].groupby(['Daypart', 'brand']).sum().reset_index()  # create final dayparts dataframe
    cx_daypart.columns = ['feature_value', 'brand', 'Impressions']
    cx_daypart['mvpd'] = 'Cox'
    cx_daypart['data_type'] = 'Daypart'
    cx_network['brand'] = brand
    cx_network = cx_network[['Network', 'Impressions', 'brand']].groupby(['Network', 'brand']).sum().reset_index()  # create final dayparts dataframe
    cx_network.columns = ['feature_value', 'brand', 'Impressions']
    cx_network['mvpd'] = 'Cox'
    cx_network['data_type'] = 'Network'

    # Cox  Cleanup - map to network and dma files
    cx_network = cx_network.merge(net_map, how='left', on=['feature_value'])
    cx_unmatched_networks = cx_network[pd.isnull(cx_network.corrected_network)]
    cx_network = cx_network[['corrected_network', 'brand', 'Impressions', 'mvpd', 'data_type']]
    cx_network.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    cx_network.loc[pd.isnull(cx_network.feature_value), 'feature_value'] = 'Unassigned'
    cx_network = cx_network.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    cx_network = cx_network[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    cx_dma['brand'] = brand
    if 'Delivered DMA' in list(cx_dma.columns): 
        cx_dma = cx_dma[['Delivered DMA', 'Impressions', 'brand']].groupby( ['Delivered DMA', 'brand']).sum().reset_index()  # create final dayparts dataframe
    if 'Delivered DMA Name' in list(cx_dma.columns): 
        cx_dma = cx_dma[['Delivered DMA Name', 'Impressions', 'brand']].groupby( ['Delivered DMA Name', 'brand']).sum().reset_index()  # create final dayparts dataframe
    if 'DMA' in list(cx_dma.columns): 
        cx_dma = cx_dma[['DMA', 'Impressions', 'brand']].groupby( ['DMA', 'brand']).sum().reset_index()  # create final dayparts dataframe
    cx_dma.columns = ['feature_value', 'brand', 'Impressions']
    cx_dma['mvpd'] = 'Cox'
    cx_dma['data_type'] = 'Market'
    cx_dma['feature_value'] = cx_dma['feature_value'].str.lower()
    cx_dma = cx_dma.merge(dma_map, how='left', on=['feature_value'])
    cx_unmatched_dmas = cx_dma[pd.isnull(cx_dma.corrected_dma)]
    cx_dma = cx_dma[['corrected_dma', 'brand', 'Impressions', 'mvpd', 'data_type']]
    cx_dma.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    cx_dma.loc[pd.isnull(cx_dma.feature_value), 'feature_value'] = 'Unassigned'
    cx_dma = cx_dma.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    cx_dma = cx_dma[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
# daily
    cx_day = cx_daily
    cx_day['brand'] = brand
    cx_day = cx_day[['Event Date', 'Impressions', 'brand']].groupby(['Event Date', 'brand']).sum().reset_index()  # create final dayparts dataframe
    cx_day.columns = ['Event Date', 'brand', 'Impressions']
    cx_day['day_num'] = cx_day['Event Date'].apply(lambda x: x.weekday())
    cx_day = cx_day.merge(dow_map, on=['day_num'], how='left')
    cx_day = cx_day[['feature_value', 'brand', 'Impressions']].groupby(['feature_value', 'brand']).sum().reset_index()
    cx_day['mvpd'] = 'Cox'
    cx_day['data_type'] = 'Day'
    cx_day = cx_day[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    ###LOOP 3: Clean up and group, limiting to Impressions
    # This part creates a list of DataFrames, then loops through them and creates a cleaner version of the brand, then creates a grouped version of the data by all values except the Impressions (brand and also network, week, creative and DMA). Then it puts those cleaned dataframes back into the original names
    cox_dfs = [cx_daypart,
               cx_network,
               cx_dma,
               cx_day]
    df_name = ['Daypart','Network','Market','Day']
    ci = 0
    while ci < len(cox_dfs):
        cox_dfs[ci]['brand'] = brand
        if ci == 0:
            # daypart
            cox_dfs[ci] = cox_dfs[ci].iloc[:, [0,1,2]]
        else:
            cox_dfs[ci] = cox_dfs[ci].iloc[:, [0,1,2]]
        cols = cox_dfs[ci].columns
        cols = list(map(lambda x: x.strip(), cols))  # remove excess spaces
        cox_dfs[ci].columns = cols
        cols = [c for c in cols if c != 'Impressions']
        cox_dfs[ci]['Impressions'] = cox_dfs[ci]['Impressions'].apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        cox_dfs[ci] = cox_dfs[ci].groupby(cols).sum().reset_index()
        cols = cox_dfs[ci].columns.values
        cols = list(map(lambda x: x if x != 'Impressions' else 'Impressions', cols))
        cols = cols[1:]
        cols.insert(0, 'feature_value')
        cox_dfs[ci].columns = cols
        cox_dfs[ci]['mvpd'] = 'Cox'
        cox_dfs[ci]['data_type'] = df_name[ci]
        ci += 1
    #cx_summary = cox_dfs[0]
    cx_network = cox_dfs[1]
    cx_dma = cox_dfs[2]
    cx_day = cox_dfs[3]
    #cx_week = cox_dfs[4]
    cx_daypart = cox_dfs[0]
        # vz_daypart = vz_daypart.replace('Prime Time','Prime') #change name from "Prime Time" to "Prime"
        # create summary from daily data
    if ('brand' in cx_daily) == False:
        cx_daily.insert(1, 'brand', brand)
    cx_day = cx_day[cx_day.feature_value != 'Total']
    #cx_day = cx_day[cx_day.feature_value != 'Total']
    cx_summary = cx_daily
    cx_summary = cx_daily[['brand', 'Impressions']].groupby(['brand']).sum().reset_index()
    cx_summary.columns = ['brand', 'Impressions']
    cx_summary['mvpd'] = 'Cox'
    cx_summary['feature_value'] = 'Total'
    cx_summary['data_type'] = 'Summary'
    cx_summary = cx_summary[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    # Spectrum Cleanup - map to network and dma files
    cx_network = cx_network[~(cx_network.feature_value == 'Total')]
    cx_network = cx_network.merge(net_map, how='left', on=['feature_value'])
    cx_unmatched_networks = cx_network[pd.isnull(cx_network.corrected_network)]
    cx_network = cx_network[['corrected_network', 'brand', 'Impressions', 'mvpd', 'data_type']]
    cx_network.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    cx_network.loc[pd.isnull(cx_network.feature_value), 'feature_value'] = 'Unassigned'
    cx_network = cx_network.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    cx_network = cx_network[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    cx_dma = cx_dma[~(cx_dma.feature_value == 'Total')]
    if len(cx_unmatched_dmas) > 0 or len(cx_unmatched_networks) > 0:
        print('Warning! Unmapped Networks or Markets! Exported as unmapped_Verizon_' + brand + '.xlsx')
        writer = pd.ExcelWriter('unmapped_Cox_' + brand + '.xlsx', engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        cx_unmatched_dmas.to_excel(writer, sheet_name='DMAs', index=False)
        cx_unmatched_networks.to_excel(writer, sheet_name='Networks', index=False)
            # Close the Pandas Excel writer and output the Excel file.
        writer.save()


# ## Verizon

# In[ ]:


if len(verizon_files) > 0:
    ###LOOP 1: Read Verizon Files into a dictionary
    # Create a dictionary for all files, and then for each file, a dictionary for the sheets. This will allow us to use one file to then create a set of dataframes for each of the types of data sets we need (Summary, Network, Week, Hour (for daypart), Creative and DMA.
    vz_files = {}  # create dictionary for all vz_files
    fi = 0
    while fi < len(verizon_files):
        vz_xl = pd.ExcelFile(vz_path+'/' + verizon_files[fi])
        vz_sheets = vz_xl.sheet_names
        # read in all sheets for the file
        dfs = {}
        si = 0
        while si < len(vz_sheets):
            # read in data
            raw_data = pd.read_excel(vz_xl, sheet_name=vz_sheets[si])
            # transform to numeric where possible
            data = raw_data
            # for c in data.columns:
            #    data[c] = pd.to_numeric(data[c], errors='ignore')
            # drop NaN rows, as well as remove the total row
            data = data[~(data.iloc[:,0].astype(str).str.contains("Total", na=False))].dropna(axis=1, how='all')
            data = data.dropna()
            data['file'] = verizon_files[fi]
            dfs.update({vz_sheets[si]: data})
            si += 1
        vz_files.update({verizon_files[fi]: dfs})
        fi += 1
    ###LOOP 2:Create dataframes for Cox files from Dictionary
    # Creates dataframes for Summary, Network, week, hour (for Daypart creation), Creative and DMAs
    vz_summary = pd.DataFrame()
    vz_network = pd.DataFrame()
    vz_week = pd.DataFrame()
    vz_daypart = pd.DataFrame()
    vz_creative = pd.DataFrame()
    vz_dma = pd.DataFrame()
    vz_weekhour = pd.DataFrame()
    vz_daily = pd.DataFrame()
    vz_day = pd.DataFrame()
    vz_rf = pd.DataFrame()
    #find numbers for each tab
    so = pd.DataFrame(vz_sheets, columns=['sheet_name']).reset_index()
    so.loc[(so['sheet_name'].str.lower().str.contains('daily|date')),'data_type'] = 'Date'
    so.loc[(so['sheet_name'].str.lower().str.contains('geo|market|dma')),'data_type'] = 'Market'
    so.loc[(so['sheet_name'].str.lower().str.contains('net')),'data_type'] = 'Network'
    so.loc[(so['sheet_name'].str.lower().str.contains('hour')),'data_type'] = 'Hour'
    so.loc[(so['sheet_name'].str.lower().str.contains('day|dow')),'data_type'] = 'Day of Week'
    so.loc[(so['sheet_name'].str.lower().str.contains('creative')),'data_type'] = 'Creative'
    so.loc[(so['sheet_name'].str.lower().str.contains('summary')),'data_type'] = 'Reach and Frequency'
    try:
        daily_num = so[(so['sheet_name'].str.lower().str.contains('daily|date'))]['index'].values[0]
    except:
        print('Verizon Daily tab missing or mis-named. Must include the words "date" or "daily".')
        pass
    try:
        dow_num = so[(so['sheet_name'].str.lower().str.contains('day|dow'))]['index'].values[0]
    except:
        print('Verizon Daily tab missing or mis-named. Must include the words "day".')
        pass
    try:
        dma_num = so[(so['sheet_name'].str.lower().str.contains('geo|market|dma'))]['index'].values[0]
    except:
        print('Verizon Markets/Geographies tab missing or mis-named - must include the phrase "DMA", "Geo" or "Market".')
        pass
    try:
        nets_num = so[so['sheet_name'].str.lower().str.contains('net')]['index'].values[0]
    except:
        print('Verizon Networks tab missing or mis-named - must include the phrase "net"')
        pass
    try:
        hour_num = so[so['sheet_name'].str.lower().str.contains('hour')]['index'].values[0]
    except: 
        print('Verizon Hourly tab missing or mis-named - must include the phrase "hour"')
    try:
        creative_num = so[so['sheet_name'].str.lower().str.contains('creative')]['index'].values[0]
    except:
        pass
    try:
        rf_num = so[so['sheet_name'].str.lower().str.contains('summary')]['index'].values[0]
    except:
        print('Verizon Reach tab missing or mis-named - must include the phrase "summary"')
    fi = 0
    while fi < len([*vz_files]):
        vz_summary = vz_summary.append(vz_files[[*vz_files][fi]][vz_sheets[rf_num]], sort=False).reset_index(drop=True)
        vz_network = vz_network.append(vz_files[[*vz_files][fi]][so[so.data_type=='Network']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        vz_dma = vz_dma.append(vz_files[[*vz_files][fi]][so[so.data_type=='Market']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        vz_daypart = vz_daypart.append(vz_files[[*vz_files][fi]][so[so.data_type=='Hour']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        vz_daily = vz_daily.append(vz_files[[*vz_files][fi]][so[so.data_type=='Date']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        vz_day = vz_day.append(vz_files[[*vz_files][fi]][so[so.data_type=='Day of Week']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        vz_rf = vz_rf.append(vz_files[[*vz_files][fi]][so[so.data_type=='Reach and Frequency']['sheet_name'].values[0]], sort=False).reset_index(drop=True)
        fi += 1
    ##create daypart from hours
    vz_daypart['Matched Daypart'] = vz_daypart.insert(2, 'Matched Daypart', 'ABC')
    vz_daypart['Matched Daypart'] = vz_daypart.iloc[:,0]
    vz_daypart = vz_daypart[vz_daypart[vz_daypart.columns[0]]!='Total']
    vz_daypart['Matched Daypart'] = vz_daypart['Matched Daypart'].astype(str).str[-8:]  # save the hours as a string
    vz_daypart['Matched Daypart'] = pd.to_datetime('2020-01-01 ' + (vz_daypart['Matched Daypart']),
                                                   format='%Y-%m-%d %H/%M', errors='ignore')  # create date with time strin
    vz_daypart.index = pd.DatetimeIndex(vz_daypart['Matched Daypart'])  # set time as index
    # set the times by hours
    if mk=='1': 
        vz_daypart.loc['2020-01-01 02:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight – 2AM-6AM'
        vz_daypart.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        vz_daypart.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        vz_daypart.loc['2020-01-01 16:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Early Fringe - 4PM-8PM'
        vz_daypart.loc['2020-01-01 20:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Prime - 8PM-12AM'
        vz_daypart.loc['2020-01-01 00:00:00': '2020-01-01 01:00:00', 'Daypart'] = 'Late Fringe - 12AM-2AM'
    else:
        vz_daypart.loc['2020-01-01 00:00:00': '2020-01-01 05:00:00', 'Daypart'] = 'Overnight - Midnight-6AM'
        vz_daypart.loc['2020-01-01 06:00:00': '2020-01-01 08:00:00', 'Daypart'] = 'Early Morning - 6AM-9AM'
        vz_daypart.loc['2020-01-01 09:00:00': '2020-01-01 15:00:00', 'Daypart'] = 'Daytime - 9AM-4PM'
        vz_daypart.loc['2020-01-01 16:00:00': '2020-01-01 17:00:00', 'Daypart'] = 'Early Fringe - 4PM-6PM'
        vz_daypart.loc['2020-01-01 18:00:00': '2020-01-01 19:00:00', 'Daypart'] = 'Prime Access - 6PM-8PM'
        vz_daypart.loc['2020-01-01 20:00:00': '2020-01-01 22:00:00', 'Daypart'] = 'Prime - 8PM-11PM'
        vz_daypart.loc['2020-01-01 23:00:00': '2020-01-01 23:59:59', 'Daypart'] = 'Late Fringe - 11PM-12AM'
    vz_daypart = vz_daypart.reset_index(drop=True)  # remove the unnessary index
    new = vz_daypart['file'].str.split(" ", n=2, expand=True)  # split the file to get the brand name
    # cx_dp_hour['brand'] = new[0]+' '+new[1] #set the new brand name
    vz_daypart['brand'] = brand
    imps_name = [c for c in vz_daypart.columns if c.lower().find('impressions')!=-1]
    vz_daypart['Impressions'] = vz_daypart[imps_name].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    vz_daypart = vz_daypart[['Daypart', 'Impressions', 'brand']].groupby(
        ['Daypart', 'brand']).sum().reset_index()  # create final dayparts dataframe
    vz_daypart.columns = ['feature_value', 'brand', 'Impressions']
    vz_daypart['mvpd'] = 'Verizon'
    vz_daypart['data_type'] = 'Daypart'
    ### LOOP 3: Clean up and group, limiting to Impressions
    # This part creates a list of DataFrames, then loops through them and creates a cleaner version of the brand, then creates a grouped version of the data by all values except the Impressions (brand and also network, week, creative and DMA). Then it puts those cleaned dataframes back into the original names
    verizon_dfs = [vz_network,
                   # vz_week,
                   vz_dma,
                   vz_day,
                   vz_daily,
                   vz_daypart]
    df_name = ['Network', 'Market', 'Day', 'Week', 'Daypart']
    ci = 0
    while ci < len(verizon_dfs):
        verizon_dfs[ci]['brand'] = brand
        if ci == 4:
            verizon_dfs[ci] = verizon_dfs[ci].iloc[:, [0, 1, 2]]
        # others
        else:
            verizon_dfs[ci] = verizon_dfs[ci].iloc[:, [0, 1, 3]]
        cols = verizon_dfs[ci].columns
        cols = list(map(lambda x: x.strip(), cols))  # remove excess spaces
        verizon_dfs[ci].columns = cols
        imps_name = [c for c in cols if c.lower().find('impressions')!=-1]
        cols = [c for c in cols if c != imps_name[0]]
        verizon_dfs[ci]['Impressions'] = verizon_dfs[ci][imps_name].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        verizon_dfs[ci] = verizon_dfs[ci].groupby(cols).sum().reset_index()
        cols = verizon_dfs[ci].columns.values
        cols = list(map(lambda x: x if x != 'Impressions' else 'Impressions', cols))
        cols = cols[1:]
        cols.insert(0, 'feature_value')
        verizon_dfs[ci].columns = cols
        verizon_dfs[ci]['mvpd'] = 'Verizon'
        verizon_dfs[ci]['data_type'] = df_name[ci]
        if imps_name[0] != 'Impressions': 
            verizon_dfs[ci] = verizon_dfs[ci].drop(columns=[imps_name[0]])
        ci += 1
    #vz_summary = verizon_dfs[0]
    vz_network = verizon_dfs[0]
    vz_dma = verizon_dfs[1]
    vz_day = verizon_dfs[2]
    vz_daily = verizon_dfs[3]
    vz_daypart = verizon_dfs[4]
    # create summary from daily data
    if ('brand' in vz_daily) == False:
        vz_daily.insert(1, 'brand', brand)
    #dow map when shortened
    vz_day = vz_day[vz_day.feature_value != 'Total']
    if 'Fri' in list(vz_day['feature_value'].values): 
        vz_day.columns = ['day_abc', 'brand', 'Impressions', 'mvpd', 'data_type']
        vz_day = vz_day.merge(da_map, how='left', on=['day_abc'])
        vz_day = vz_day[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    vz_daily = vz_daily[vz_daily.feature_value != 'Total']
    vz_summary = vz_daily[['brand', 'Impressions']].groupby(['brand']).sum().reset_index()
    vz_summary.columns = ['brand', 'Impressions']
    vz_summary['mvpd'] = 'Verizon'
    vz_summary['feature_value'] = 'Total'
    vz_summary['data_type'] = 'Summary'
    vz_summary = vz_summary[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    # Spectrum Cleanup - map to network and dma files
    vz_network = vz_network[~(vz_network.feature_value == 'Total')]
    vz_network = vz_network.merge(net_map, how='left', on=['feature_value'])
    vz_unmatched_networks = vz_network[pd.isnull(vz_network.corrected_network)]
    vz_network = vz_network[['corrected_network', 'brand', 'Impressions', 'mvpd', 'data_type']]
    vz_network.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    vz_network.loc[pd.isnull(vz_network.feature_value), 'feature_value'] = 'Unassigned'
    vz_network = vz_network.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    vz_network = vz_network[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    vz_dma = vz_dma[~(vz_dma.feature_value == 'Total')]
    vz_dma['feature_value'] = vz_dma['feature_value'].str.lower()
    vz_dma = vz_dma.merge(dma_map, how='left', on=['feature_value'])
    vz_unmatched_dmas = vz_dma[pd.isnull(vz_dma.corrected_dma)]
    vz_dma = vz_dma[['corrected_dma', 'brand', 'Impressions', 'mvpd', 'data_type']]
    vz_dma.columns = ['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']
    vz_dma.loc[pd.isnull(vz_dma.feature_value), 'feature_value'] = 'Unassigned'
    vz_dma = vz_dma.groupby(['feature_value', 'brand', 'mvpd', 'data_type']).sum().reset_index()
    vz_dma = vz_dma[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
    if len(vz_unmatched_dmas) > 0 or len(vz_unmatched_networks) > 0:
        print('Warning! Unmapped Networks or Markets! Exported as unmapped_Verizon_' + brand + '.xlsx')
        writer = pd.ExcelWriter('unmapped_Verizon_' + brand + '.xlsx', engine='xlsxwriter')
        # Write each dataframe to a different worksheet.
        vz_unmatched_dmas.to_excel(writer, sheet_name='DMAs', index=False)
        vz_unmatched_networks.to_excel(writer, sheet_name='Networks', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


# # Clean, Standardize and merge

# ### Create Weeks dataframes for all MVPDs

# In[ ]:


# ### Create weeks from Daily: Cox, Spectrum and Verizon
#create dataframe with 1/1/1900 date for each mvpd. Fill will updated start/end date for each mvpd after checking to confirm that each is actually a date. 
#then drop any values that still have original value.
dd = datetime.strptime('1900-01-01', '%Y-%m-%d')
mvpds = ['Comcast', 'Spectrum','Cox','Verizon']
dates = pd.DataFrame(data={'start': [dd, dd, dd, dd] , 'end': [dd, dd,dd,dd], 'mvpd': mvpds})
# create dates as values
if len(spectrum_files) > 0:
    try:
        min(sp_daily.iloc[:, 0]).year
    except: 
        sp_year = input('Year missing for Spectrum. define year: ')
        sp_daily.iloc[:,0] = sp_daily.iloc[:,0].apply(lambda x: datetime.strptime(x+'-'+sp_year, '%d-%b-%Y'))
    dates.loc[dates.mvpd=='Spectrum','start'] = min(sp_daily.iloc[:, 0])
    dates.loc[dates.mvpd=='Spectrum','end'] = max(sp_daily.iloc[:, 0])
if len(cox_files) > 0:
    try:
        min(cx_daily.iloc[:, 0]).year
    except: 
        cx_year = input('Year missing for Cox. define year: ')
        cx_daily.iloc[:,0] = cx_daily.iloc[:,0].apply(lambda x: datetime.strptime(x+'-'+cx_year, '%d-%b-%Y'))
    dates.loc[dates.mvpd=='Cox','start'] = min(cx_daily.iloc[:, 0])
    dates.loc[dates.mvpd=='Cox','end'] = max(cx_daily.iloc[:, 0])
if len(verizon_files) > 0:
    try:
        min(vz_daily.iloc[:, 0]).year
    except: 
        vz_year = input('Year missing for Verizon. define year: ')
        vz_daily.iloc[:,0] = vz_daily.iloc[:,0].apply(lambda x: datetime.strptime(x+'-'+vz_year, '%d-%b-%Y'))
    dates.loc[dates.mvpd=='Verizon','start'] = min(vz_daily.iloc[:, 0])
    dates.loc[dates.mvpd=='Verizon','end'] = max(vz_daily.iloc[:, 0])
if len(comcast_files) > 0:
    try:
        min(com_week.iloc[:, 0]).year
    except: 
        com_year = input('Year missing for Comcast. define year: ')
        com_week.iloc[:,0] = com_week.iloc[:,0].apply(lambda x: datetime.strptime(x+'-'+com_year, '%d-%b-%Y'))
    dates.loc[dates.mvpd=='Comcast','start'] = min(com_week.iloc[:, 0])
    dates.loc[dates.mvpd=='Comcast','end'] = max(com_week.iloc[:, 0])+pd.DateOffset(days=6)
dates = dates[~(dates.start=='1900-01-01')] #remove rows that still have original value
#for each MVPD, create weekly dataframe based on the beginning and ending dates across all mvpds
if len(comcast_files) > 0 :
    if min(dates.iloc[:, 0]) < min(dates[dates.mvpd == 'Comcast'].iloc[:, 0]):
        min_dt = min(dates.iloc[:,0])
        delta = min_dt-min(dates[dates.mvpd == 'Comcast'].iloc[:, 0])
        weeks_ahead = -(-abs(delta.days)//7) #round up to nearest number of weeks
        days_ahead = weeks_ahead*7
        fd = min(dates[dates.mvpd == 'Comcast'].iloc[:, 0]) - pd.DateOffset(days=days_ahead)
        #fd = min(dates[dates.mvpd == 'Comcast'].iloc[:, 0]) - pd.DateOffset(days=7)
    elif min(dates.iloc[:, 0]) == min(dates[dates.mvpd == 'Comcast'].iloc[:, 0]):
        fd = min(dates.iloc[:, 0])
    else:
        fd = min(dates[dates.mvpd == 'Comcast'].iloc[:, 0]) + pd.DateOffset(days=7)
else: 
    fd = min(dates.iloc[:,0])
ild = fd + pd.DateOffset(days=6)
ld = max(dates.iloc[:, 1])
wni = 1
weeks = pd.DataFrame()
while fd < ld:
    newdate = pd.DataFrame({'first_date': [fd], 'last_date': [ild]})
    newdate['week_num'] = 'Week ' + str(wni)
    weeks = weeks.append(newdate, sort=False).reset_index(drop=True)
    wni += 1
    fd = fd + pd.DateOffset(days=7)
    ild = ild + pd.DateOffset(days=7)
# Creates weekly files for Spectrum or Cox or Verizon if they exist.
if len(spectrum_files) > 0:
    # create weekly df 
    sp_week = sp_daily
    di = 0
    while di < len(weeks.iloc[:, 0]):
        sp_week.loc[(sp_week.iloc[:, 0] >= weeks.iloc[:, 0][di]) & (sp_week.iloc[:, 0] <= weeks.iloc[:, 1][di]), 'feature_value'] = weeks.iloc[:, 0][di]
        di += 1
    sp_week = sp_week[['feature_value', 'brand', 'Impressions']].groupby(['feature_value', 'brand']).sum().reset_index()
    sp_week['mvpd'] = 'Spectrum'
    sp_week['data_type'] = 'Week'
if len(cox_files) > 0:
    # create weekly files
    cx_week = cx_daily
    #cx_week.columns = ['Event Date', 'Placement Name','Impressions', 'Video Ads 100% Complete', 'file', 'brand']
    di = 0
    while di < len(weeks.iloc[:, 0]):
        cx_week.loc[(cx_week.iloc[:, 0] >= weeks.iloc[:, 0][di]) & (cx_week.iloc[:, 0] <= weeks.iloc[:, 1][di]), 'feature_value'] = weeks.iloc[:, 0][di]
        di += 1
    cx_week = cx_week[['feature_value', 'brand', 'Impressions']].groupby(['feature_value', 'brand']).sum().reset_index()
    cx_week['mvpd'] = 'Cox'
    cx_week['data_type'] = 'Week'
    cx_week = cx_week[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]
if len(verizon_files) > 0:
    # create weekly files
    vz_week = vz_daily
    di = 0
    while di < len(weeks.iloc[:, 0]):
        vz_week.loc[(vz_week.iloc[:, 0] >= weeks.iloc[:, 0][di]) & (vz_week.iloc[:, 0] <= weeks.iloc[:, 1][di]), 'feature_value'] = weeks.iloc[:, 0][di]
        di += 1
    vz_week = vz_week[['feature_value', 'brand', 'Impressions']].groupby(['feature_value', 'brand']).sum().reset_index()
    vz_week['mvpd'] = 'Verizon'
    vz_week['data_type'] = 'Week'
    vz_week = vz_week[['feature_value', 'brand', 'Impressions', 'mvpd', 'data_type']]


# ### Create Reach & Frequency dataframes for each MVPD
#  Read in reach & frequency files for Comcast and Spectrum. May need updating depending on structure; assumed inputs are:
# - Spectrum has "Spectrum" in the file name, is an excel file
# - Comcast has "parallel" in the file name, is a csv that is UTF-16 delimited by tabs

# In[ ]:


rf_df = pd.DataFrame()
#read files for Spectrum & Comcast
if len(rf_files) > 0:
    fi = 0
    while fi < len(rf_files):
        if rf_files[fi].find('xlsx') != -1:
            rf_df_new = pd.read_excel(rf_path+'/' + rf_files[fi])
            if len(rf_df_new) > 1:
                rf_df_new = rf_df_new[(rf_df_new.Segment.str.contains('Total', na=False))]
        #if rf_files[fi].find('csv') != -1:
           # rf_df_new = pd.read_csv('files/reach_freq/' + rf_files[fi])
        #if (rf_files[fi].lower().find('parallel') != -1) or (rf_files[fi].lower().find('parrallel') != -1) or (
        #rf_files[fi].lower().find('parralell') != -1):
            mvpd = 'Comcast'
        if rf_files[fi].lower().find('spectrum') != -1:
            mvpd = 'Spectrum'
        cols = rf_df_new.columns
        hhcols = [c for c in cols if (c.lower().find('hh') != -1) & (c.lower().find('book') == -1)]
        rf_df_new[hhcols] = rf_df_new[hhcols].apply(
            lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce'))
        total_hhs = [sum(x) for x in rf_df_new[hhcols].values][0]
        icols = [c for c in cols if
                 ((c.lower().find('impression') != -1) & (c.lower().find('deliver') != -1)) | (c.lower() == 'imp')]
        rf_df_new[icols] = rf_df_new[icols].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce'))
        total_imps = [sum(x) for x in rf_df_new[icols].values][0]
        
        # total_freq = round(total_imps/total_hhs,2)
        rf_df_new = pd.DataFrame(data=[[mvpd, total_imps, total_hhs]],
                                 columns=['MVPD', 'Total Impressions', 'Total HHs'])
        rf_df = rf_df.append(rf_df_new, sort=False).reset_index(drop=True)
        fi += 1
# clean up Cox reach-frequency when present
if len(cox_files) > 0:
    if 'Reach-Frequency' in cx_sheets:
        if len(cx_rf.columns)>3:
            cx_rf = cx_rf.iloc[:, 1:3]
            cx_rf.columns = ['Total HHs', 'Total Frequency']
            cx_rf['Total Impressions'] = cx_rf['Total HHs'] * cx_rf['Total Frequency']
            cx_rf['MVPD'] = 'Cox'
        cx_rf = cx_rf[['MVPD', 'Total Impressions', 'Total HHs']]
        rf_df = rf_df.append(cx_rf).reset_index(drop=True)
# clean up Verizon reach-frequency when present
if len(verizon_files) > 0:
    if 'Order Summary' in vz_sheets:
        if len(vz_rf.columns)>3: 
            vz_rf = vz_rf.iloc[:, [6,7,8]]
            vz_rf.columns = ['Total Impressions', 'Total HHs', 'Total Frequency']
            vz_rf['MVPD'] = 'Verizon'
        # cx_rf['Total Impressions'] = cx_rf['Total HHs'] * cx_rf['Total Frequency']
        vz_rf = vz_rf[['MVPD', 'Total Impressions', 'Total HHs']]
        rf_df = rf_df.append(vz_rf).reset_index(drop=True)
# create total row and frequency - only run for rf_df
if len(rf_df) > 0:
    rf_df = rf_df.append(newcom_campaign).reset_index(drop=True)
    total_imps = rf_df['Total Impressions'].sum()
    total_reach = (rf_df['Total HHs'].sum())
    rf_df = rf_df.append(pd.DataFrame(data=[['Total', total_imps, total_reach]], columns=rf_df.columns),
                         sort=False).reset_index(drop=True)
    rf_df.columns = ['mvpd', 'Impressions', 'Reach']
    rf_df['Frequency'] = round(rf_df['Impressions'] / rf_df['Reach'], 2)
#unpivot impressions reach and frequency
if len(rf_df)>0:
    rf_df = rf_df.set_index('mvpd')
    cols = rf_df.columns
    i = 0 
    rf=pd.DataFrame()
    while i<len(cols): 
        n = pd.DataFrame(rf_df.iloc[:,i])
        n.columns = ['metric']
        n['metric_type'] = cols[i]
        n = n.reset_index()
        rf = rf.append(n)
        i+=1
    rf['data_type'] = 'Reach and Frequency'
    rf['feature_value'] = 'Total'
    rf['brand'] = brand


# ### Create Full dataframe by appending all cuts for all available MVPDs and create totals by column and cut.

# In[ ]:


#Append files
#Combine Spectrum, Comcast and Cox and Verizon
# append applicable dataframes
df = pd.DataFrame()
if len(comcast_files) > 0:
    # Comcast
    df = df.append(com_summary, sort=False).append(com_network, sort=False).append(com_daypart, sort=False).append(com_dma, sort=False).append(com_day, sort=False).append(com_week, sort=False)
if len(spectrum_files) > 0:
    # Spectrum
    df = df.append(sp_summary, sort=False).append(sp_network, sort=False).append(sp_daypart, sort=False).append(sp_dma,sort=False).append(sp_day, sort=False).append(sp_week, sort=False)
if len(cox_files) > 0:
    # Cox
    df = df.append(cx_summary, sort=False).append(cx_network, sort=False).append(cx_daypart, sort=False).append(cx_dma,sort=False).append(cx_day, sort=False).append(cx_week, sort=False)
if len(verizon_files) > 0:
    # Verizon
    df = df.append(vz_summary, sort=False).append(vz_network, sort=False).append(vz_daypart, sort=False).append(vz_dma, sort=False).append(vz_day, sort=False).append(vz_week, sort=False)
if len(new_com_files) > 0:
    # New Comcast ADW added
    df = df.append(newcom_summary, sort=False).append(newcom_network, sort=False).append(newcom_daypart, sort=False).append(newcom_market, sort=False).append(newcom_day, sort=False).append(newcom_week, sort=False)
df.loc[df.data_type=='Week', 'feature_value'] = pd.to_datetime(df[df.data_type=='Week']["feature_value"]).dt.strftime("%Y-%m-%d")
# creates total ROWS for each feature_value; works regardless of # of MVPDs
types = list(df.data_type.unique())
ti = 0
while ti < len(types): #total per feature (all mvpds)
    fi = 0
    features = list(df[df.data_type == types[ti]]['feature_value'].unique())
    while fi < len(features): #total per feature, per feature type
        total = df[(df.data_type == types[ti]) & (df.feature_value == features[fi])]['Impressions'].fillna(0).sum()
        nd = [features[fi], brand, total, 'Total', types[ti]]
        ndf = pd.DataFrame(data=[nd], columns=df.columns)
        df = df.append(ndf).reset_index(drop=True)
        fi += 1
    ti += 1
# create totals for each feature and append
feature_totals = df[['Impressions','mvpd', 'data_type']].groupby(['mvpd','data_type']).sum().reset_index()
feature_totals['brand'] = brand
feature_totals['feature_value'] = 'Total'
feature_totals = feature_totals[['feature_value','brand', 'Impressions', 'mvpd','data_type']]
df = df.append(feature_totals).reset_index(drop=True)
#update names to be able to pivot on type of metric
df['metric_type'] = 'Impressions'
df.columns = ['feature_value', 'brand', 'metric', 'mvpd', 'data_type', 'metric_type']
if len(rf_df)>0:
    df = df.append(rf)
df['column_name'] = df['mvpd']+' '+df['metric_type']


# In[ ]:


# appending new MVPD artemas did
#df = df.append(art_df)


# # Prepare for Export

# ### Create separate dataframes for Day, Summary, Daypart, Network, Market and Week cuts. 

# In[ ]:


#create separate dataframes
#Re-order Days of Week
#use the day-of-week mapping to re-order the days of the week. Starts with Monday.
day = df[(df.data_type=='Day') & (df.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df[(df.data_type=='Day') & (df.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
# re-order days of week
day = day.merge(dow_map, on='feature_value', how='left').sort_values('day_num')# merge on day
day = day.iloc[:, 0:-1]  # remove order number
day.columns = np.where(day.columns=='feature_value', 'Day', day.columns) 
summary = df[(df.feature_value=='Total') & (df.data_type!='Reach and Frequency')].groupby(['brand','data_type', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df[(df.feature_value=='Total') & (df.data_type=='Reach and Frequency')].groupby(['brand','data_type', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
summary = summary.fillna(0)
summary = summary[summary.data_type!='Summary']
summary.columns = np.where(summary.columns=='data_type', 'cut', summary.columns) 
daypart = df[(df.data_type=='Daypart')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()
daypart.columns = np.where(daypart.columns=='feature_value', 'Daypart', daypart.columns) 
network = df[(df.data_type=='Network') & (df.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df[(df.data_type=='Network') & (df.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
network.columns = np.where(network.columns=='feature_value', 'Network', network.columns) 
dma = df[(df.data_type=='Market') & (df.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df[(df.data_type=='Market') & (df.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
dma.columns = np.where(dma.columns=='feature_value', 'Market', dma.columns) 
week = df[(df.data_type=='Week') & (df.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df[(df.data_type=='Week') & (df.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
week.columns = np.where(week.columns=='feature_value', 'Week', week.columns) 

dma = pd.merge(dma, full_mapping, left_on = 'Market', right_on = 'corrected_dma', how = 'left')
dma = dma.drop_duplicates() #drop duplicates
dma = dma.drop(columns=['corrected_dma'])
#dma.rename(columns = {'dma_code':'DMA'}, inplace = True)


# ### Export to internal file, including totals by MVPD

# In[ ]:


#Output everything into an Excel file
#Includes output of dataframes, and formatting for each tab, and placement of logo.
#output to Excel in correct format without creative
brand_list = [brand]
#output for Comcast+Spectrum+Cox+verizon with formatting UPDATED FOR SUMMARY with reach & frequency, no creative
bi = 0
dfs = [summary,
daypart, network, #creative,
dma, day,week]
if len(comcast_files)>0:
    dfs = [summary, daypart, network, dma, day, week]
while bi<len(brand_list):
    writer = pd.ExcelWriter(brand+'/'+brand_list[bi]+'_internal_output.xlsx', engine='xlsxwriter')
    # Write each dataframe to a different worksheet.
    summary[summary['brand']==brand_list[bi]].to_excel(writer, sheet_name='Summary', index=False, startrow=4, header=False)
    daypart[daypart['brand']==brand_list[bi]].to_excel(writer, sheet_name='Dayparts', index=False, startrow=1, header=False)
    network[network['brand']==brand_list[bi]].to_excel(writer, sheet_name='Networks', index=False, startrow=1, header=False)
    #creative[creative['brand']==brand_list[bi]].to_excel(writer, sheet_name='Creatives', index=False, startrow=1, header=False)
    dma[dma['brand']==brand_list[bi]].to_excel(writer, sheet_name='Market', index=False, startrow=1, header=False)
    day[day['brand']==brand_list[bi]].to_excel(writer, sheet_name='Day of Week', index=False, startrow=1, header=False)
    week[week['brand']==brand_list[bi]].to_excel(writer, sheet_name='Week', index=False, startrow=1, header=False)
    #if len(comcast_files)>0:
     #   day[day['brand']==brand_list[bi]].to_excel(writer, sheet_name='Day of Week', index=False, startrow=1, header=False)
      #  week[week['brand']==brand_list[bi]].to_excel(writer, sheet_name='Week', index=False, startrow=1, header=False)
    #get the workbook worksheet objects
    workbook  = writer.book
    sheet_names = list(writer.sheets.keys())
    si = 0
    while si<len(sheet_names):
        worksheet = writer.sheets[sheet_names[si]]
        #set formatting
        date_format = workbook.add_format({'num_format': 'dd/mm/yy', 'font_name': 'Brown', 'align': 'vjustify'})
        format_num = workbook.add_format({'num_format': '#,##0', 'font_name': 'Brown', 'align':'vjustify'})
        text_format = workbook.add_format({'font_name': 'Brown', 'align':'vjustify'})
        border_format = workbook.add_format({'border': 1})
        title_format = workbook.add_format({'font_name': 'Brown', 'font_size': 14, 'font_color':'#150038','valign': 'vjustify'})
        subtitle_format = workbook.add_format({'font_name': 'Brown', 'font_size': 11, 'font_color':'#150038'})
        #Set header formats
        header_format = workbook.add_format({'bold': True, 'valign': 'vjustify', 'align': 'center_across', 'bg_color': '#150038', 'font_color': '#FFFFFF', 'font_name': 'Brown','font_size':12})
        worksheet.hide_gridlines(2)
        #determine final col
        col_len = len(dfs[si][dfs[si]==brand_list[bi]].columns.values)
        col_len_map = pd.DataFrame(data={'col_len': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 'letter': ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']})
        col_letter=col_len_map['letter'][col_len_map.col_len==col_len].values[0]
        #determine final row
        row_final = len(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values)+1
        #different rules for the first tab, which is a summary
        if si==0:
            worksheet.set_column('A:A', 35, text_format) #set first column to text format, width 12
            worksheet.set_column('B:B', 24, text_format) #set second column to text format, width 12
            worksheet.set_column('C:Q', 24, format_num) #set columns C:N to 12 width with number format
            worksheet.set_column('K:Q', 24, text_format)
            #add column header with formatting
            for col_num, value in enumerate(dfs[si][dfs[si]==brand_list[bi]].columns.values):
                worksheet.write(3, col_num, value.title(), header_format)
            worksheet.set_row(3,28) #set column header row height to 28
            #set all rows in the table to 25 height
            for row_num,value in enumerate(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values):
                worksheet.set_row(row_num+4,25)
            #add borders for table based on size of table as defined above. Adding 4 to account for title.
            worksheet.conditional_format('A5:'+col_letter+str(row_final+4) , { 'type' : 'no_blanks' , 'format' : border_format} )
            #add Ampersand logo to report
            worksheet.insert_image('E1', '/Users/artemasw/Ampersand_20-21/ampersand_logo_purple_blue.png', {'x_scale': 0.12, 'y_scale': 0.12})
            #add Title
            worksheet.write('A1', brand_list[bi], title_format)
            worksheet.set_row(0,28)
            worksheet.write('A2', 'End of Campaign Report', subtitle_format)
        else:
            worksheet.set_column('A:A', 25, text_format)
            worksheet.set_column('B:B', 30, text_format)
            worksheet.set_column('C:G', 24, format_num)
            worksheet.conditional_format('A'+str(row_final)+':'+col_letter+str(row_final), {'type' : 'no_blanks','format':header_format})
            for col_num, value in enumerate(dfs[si][dfs[si]==brand_list[bi]].columns.values):
                worksheet.write(0, col_num, value.title(), header_format)
            for row_num,value in enumerate(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values):
                worksheet.set_row(row_num+1,25)
            worksheet.conditional_format('A1:'+col_letter+str(row_final) , { 'type' : 'no_errors' , 'format' : border_format} )
            worksheet.set_row(0,28)
        si+=1
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    bi+=1


# ### Limit to totals, removing values by individual MVPD

# In[ ]:


df_total = df[df.mvpd=='Total'] #limit to only totals
#create all dataframes
#Re-order Days of Week
#use the day-of-week mapping to re-order the days of the week. Starts with Monday.
day = df_total[(df_total.data_type=='Day') & (df_total.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df_total[(df_total.data_type=='Day') & (df_total.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
# re-order days of week
day = day.merge(dow_map, on='feature_value', how='left').sort_values('day_num')# merge on day
day = day.iloc[:, 0:-1]  # remove order number
day.columns = np.where(day.columns=='feature_value', 'Day', day.columns) 
summary = df_total[(df_total.feature_value=='Total') & (df_total.data_type!='Reach and Frequency')].groupby(['brand','data_type', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df_total[(df_total.feature_value=='Total') & (df_total.data_type=='Reach and Frequency')].groupby(['brand','data_type', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
summary = summary.fillna(0)
summary = summary[summary.data_type!='Summary']
summary.columns = np.where(summary.columns=='data_type', 'cut', summary.columns) 
daypart = df_total[(df_total.data_type=='Daypart')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()
daypart.columns = np.where(daypart.columns=='feature_value', 'Daypart', daypart.columns) 
network = df_total[(df_total.data_type=='Network') & (df_total.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df_total[(df_total.data_type=='Network') & (df_total.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
network.columns = np.where(network.columns=='feature_value', 'Network', network.columns) 
dma = df_total[(df_total.data_type=='Market') & (df_total.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df_total[(df_total.data_type=='Market') & (df_total.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
dma.columns = np.where(dma.columns=='feature_value', 'Market', dma.columns) 
week = df_total[(df_total.data_type=='Week') & (df_total.feature_value!='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index().append(df_total[(df_total.data_type=='Week') & (df_total.feature_value=='Total')].groupby(['feature_value','brand', 'column_name'])['metric'].sum().unstack('column_name').reset_index()).reset_index(drop=True)
week.columns = np.where(week.columns=='feature_value', 'Week', week.columns) 

dma = pd.merge(dma, full_mapping, left_on = 'Market', right_on = 'corrected_dma', how = 'left')
dma = dma.drop_duplicates()
dma = dma.drop(columns=['corrected_dma'])

# # Export totals file

# In[ ]:


#Output everything into an Excel file
#Includes output of dataframes, and formatting for each tab, and placement of logo.
#output to Excel in correct format without creative
brand_list = [brand]
#output for Comcast+Spectrum+Cox+verizon with formatting UPDATED FOR SUMMARY with reach & frequency, no creative
bi = 0
dfs = [summary,
daypart, network, #creative,
dma, day,week]
if len(comcast_files)>0:
    dfs = [summary, daypart, network, dma, day, week]
while bi<len(brand_list):
    writer = pd.ExcelWriter(brand+'/'+brand_list[bi]+'_totals_only_output.xlsx', engine='xlsxwriter')
    # Write each dataframe to a different worksheet.
    summary[summary['brand']==brand_list[bi]].to_excel(writer, sheet_name='Summary', index=False, startrow=4, header=False)
    daypart[daypart['brand']==brand_list[bi]].to_excel(writer, sheet_name='Dayparts', index=False, startrow=1, header=False)
    network[network['brand']==brand_list[bi]].to_excel(writer, sheet_name='Networks', index=False, startrow=1, header=False)
    #creative[creative['brand']==brand_list[bi]].to_excel(writer, sheet_name='Creatives', index=False, startrow=1, header=False)
    dma[dma['brand']==brand_list[bi]].to_excel(writer, sheet_name='Market', index=False, startrow=1, header=False)
    #if len(comcast_files)>0:
    day[day['brand']==brand_list[bi]].to_excel(writer, sheet_name='Day of Week', index=False, startrow=1, header=False)
    week[week['brand']==brand_list[bi]].to_excel(writer, sheet_name='Week', index=False, startrow=1, header=False)
    #get the workbook worksheet objects
    workbook  = writer.book
    sheet_names = list(writer.sheets.keys())
    si = 0
    while si<len(sheet_names):
        worksheet = writer.sheets[sheet_names[si]]
        #set formatting
        date_format = workbook.add_format({'num_format': 'dd/mm/yy', 'font_name': 'Brown', 'align': 'vjustify'})
        format_num = workbook.add_format({'num_format': '#,##0', 'font_name': 'Brown', 'align':'vjustify'})
        text_format = workbook.add_format({'font_name': 'Brown', 'align':'vjustify'})
        border_format = workbook.add_format({'border': 1})
        title_format = workbook.add_format({'font_name': 'Brown', 'font_size': 14, 'font_color':'#150038','valign': 'vjustify'})
        subtitle_format = workbook.add_format({'font_name': 'Brown', 'font_size': 11, 'font_color':'#150038'})
        #Set header formats
        header_format = workbook.add_format({'bold': True, 'valign': 'vjustify', 'align': 'center_across', 'bg_color': '#150038', 'font_color': '#FFFFFF', 'font_name': 'Brown','font_size':12})
        worksheet.hide_gridlines(2)
        #determine final col
        col_len = len(dfs[si][dfs[si]==brand_list[bi]].columns.values)
        col_len_map = pd.DataFrame(data={'col_len': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], 'letter': ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q']})
        col_letter=col_len_map['letter'][col_len_map.col_len==col_len].values[0]
        #determine final row
        row_final = len(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values)+1
        #different rules for the first tab, which is a summary
        if si==0:
            worksheet.set_column('A:A', 35, text_format) #set first column to text format, width 12
            worksheet.set_column('B:B', 24, text_format) #set second column to text format, width 12
            worksheet.set_column('C:Q', 24, format_num) #set columns C:N to 12 width with number format
            worksheet.set_column('K:Q', 24, text_format)
            #add column header with formatting
            for col_num, value in enumerate(dfs[si][dfs[si]==brand_list[bi]].columns.values):
                worksheet.write(3, col_num, value.title(), header_format)
            worksheet.set_row(3,28) #set column header row height to 28
            #set all rows in the table to 25 height
            for row_num,value in enumerate(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values):
                worksheet.set_row(row_num+4,25)
            #add borders for table based on size of table as defined above. Adding 4 to account for title.
            worksheet.conditional_format('A5:'+col_letter+str(row_final+4) , { 'type' : 'no_blanks' , 'format' : border_format} )
            #add Ampersand logo to report
            worksheet.insert_image('E1', '/Users/artemasw/Ampersand_20-21/ampersand_logo_purple_blue.png', {'x_scale': 0.12, 'y_scale': 0.12})
            #add Title
            worksheet.write('A1', brand_list[bi], title_format)
            worksheet.set_row(0,28)
            worksheet.write('A2', 'End of Campaign Report', subtitle_format)
        else:
            worksheet.set_column('A:A', 25, text_format)
            worksheet.set_column('B:B', 30, text_format)
            worksheet.set_column('C:G', 24, format_num)
            worksheet.conditional_format('A'+str(row_final)+':'+col_letter+str(row_final), {'type' : 'no_blanks','format':header_format})
            for col_num, value in enumerate(dfs[si][dfs[si]==brand_list[bi]].columns.values):
                worksheet.write(0, col_num, value.title(), header_format)
            for row_num,value in enumerate(dfs[si][dfs[si]['brand']==brand_list[bi]]['brand'].values):
                worksheet.set_row(row_num+1,25)
            worksheet.conditional_format('A1:'+col_letter+str(row_final) , { 'type' : 'no_errors' , 'format' : border_format} )
            worksheet.set_row(0,28)
        si+=1
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    bi+=1



