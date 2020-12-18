import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn plotting style
sns.set_style("darkgrid")

def ecoRead(filename, usecols, names, sep='\t', skiprows=None, splitDate='False') :

    # Read in data frame
    # df = pd.read_csv(filename, usecols=usecols, sep=sep, skiprows=skiprows, header=None)
    df = pd.read_csv(filename, usecols=usecols, names=names, sep=sep, skiprows=skiprows, header=0)
  
    # convert the date/time column(s) (it's a string) to a datetime type
    if splitDate == 'ctd' :        
        # 27 Sep 2020 02:21:04
        datetime_series = pd.to_datetime(df['date'])
    elif splitDate == 'boussole2020' :
        # Dec + 9  + 2020 + 11:38:04  (month is mislabeled as day_name in file header)
        datetime_series = pd.to_datetime(df['month'] + ' ' + df['day'].astype(str) + ' ' + df['year'].astype(str) + ' ' + df['time'])
    else :
        if splitDate :
            #[Thu Jun 18 09:09:53.952 2020] ECOV2-CREE00530	06/18/20	09:09:34
            datetime_series = pd.to_datetime(df['date'] + ' ' + df['time'])
        else :
            #[Thu Jun 18 09:10:13.760 2020] MCOMS-037  (This code will break if day-of-month digits %d are variable width.)
            datetime_series = pd.to_datetime(df['date'].str[:30], format='[%a %b %d %X.%f %Y]')

    # create datetime index passing the datetime series
    df2 = df.set_index(pd.DatetimeIndex(datetime_series.values, name='datetime'))
    
    # we might not need the string date/time column anymore
    # df2.drop(['date', 'time'], axis=1, inplace=True, errors='ignore')
    return df2

def ecoSubtractBlank(df, channel, blank) :
    df[channel+'b'] = df[channel] - blank
    return df

def ecoCombineHLGain(df, hiGain, lowGain, combined) :
    df[combined] = df[hiGain]
    df[combined].where(df[hiGain] < 65200, df[lowGain]*10, inplace=True) # cond == True, keep HG1; otherwise replace with LG1*10
    return df

def ecoFilter(df, channel, filterWidth=7) :
    df[channel+'m'] = df[channel].rolling(filterWidth).median().rolling(filterWidth).mean()
    df[channel+'s'] = df[channel].rolling(filterWidth).median().rolling(filterWidth).std()
    df[channel+'cov'] = df[channel+'s']/df[channel+'m']
    return df

def ecoClusterData(df, channel, eps=0.01, min_samples=10) :
    # Extract filtered + selectged channel data from data frame for use with DBSCAN
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    X = df[channel].to_numpy()
    XX = np.concatenate([X.reshape(-1,1)], axis=1)
    XXs = StandardScaler().fit_transform(XX)

    # Cluster data using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(XXs)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Number of clusters:', n_clusters_)
    print('Number of points as noise:', n_noise_, 'of', XX.shape[0], 'points total')
    df['labels'] = labels
    df['core_samples_mask'] = core_samples_mask

    return df

def ecoPlotTgtRef(dftgt, channel_tgt, dfref, channel_ref, legend_labels) :
    hf = plt.figure()
    sns.lineplot(data=dftgt[channel_tgt],    marker='o', markersize=2, markeredgecolor='none', linewidth=0)
    sns.lineplot(data=dfref[channel_ref], marker='o', markersize=2, markeredgecolor='none', linewidth=0)
    plt.legend(legend_labels)
    plt.show()
    return hf

def ecoPlotClusters(dfc, channel, original_df, original_channel) :
    hf = plt.figure()
    print(channel)

    # Depends on masks and label from ecoClusterData()
    core_samples_mask = dfc['core_samples_mask']
    labels = dfc.labels
    unique_labels = set(labels)

    # Map set of cluster labels to colors in color map 
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]

    sns.lineplot(data=original_df[original_channel], linewidth=1)  # Original channel data

    # Loop through labels and plot each cluster in separate color
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        # Plot class members in core samples as small dots, outside core samples as larter dots
        sns.lineplot(data=dfc[channel].where(class_member_mask & core_samples_mask), 
            color=col, marker='o', markersize=3, 
            markerfacecolor=tuple(col), markeredgecolor='none', linewidth=0)

        sns.lineplot(data=dfc[channel].where(class_member_mask & ~core_samples_mask), 
            color=col, marker='o', markersize=6, 
            markerfacecolor=tuple(col), markeredgecolor='none', linewidth=0)
    return hf

def ecoExtractFromCluster(dfmatch, df, channel, device, sample_to_extract) :
    labels = df['labels']
    core_samples_mask = df['core_samples_mask']
    unique_labels = set(labels)

    for k in unique_labels :
        if k >= 0 :
            class_member_mask = (labels == k)
            
            data = df[channel].where(class_member_mask & core_samples_mask).dropna().values

            if data.shape[0] >= sample_to_extract :
                measurement = data[sample_to_extract - 1]
                # print(k, measurement)
                dfmatch.loc[k, device] = measurement

            else :
                print('cluster', k, 'has less than', sample_to_extract, 'values')
    return dfmatch