# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:29:37 2022

@author: ddimoudis
"""
from more_itertools import sliced
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def anomaly_score(dataframe, column, init_window, update_frequency = 100, ct = 10):
    """
    This function is responsible for attributing an outlier score to every
    observation in the time series based on the median and the standard 
    deviation of a rolling window with dynamic size.
    ---------------------------------------------------------------------------
    Arguments
    ---------------------------------------------------------------------------
    dataframe: The dataframe of the particular metric.
    column:    The particular measurement.
    init_window:    The initial window to be used in order to indentify outliers
    update_frequency: The size of the slices upon which to perform the update 
                of the window size.
    ct:  The size of change when an update happens to window. 
    """
    
    # Check input and highlight the available choices
    if (column not in dataframe.columns):
        print ("Incorrect column please choose from : " + " ".join(i for i in dataframe.columns))
        return 
    
    #Clean from NA and reset the index of the copied dataframe
    clean_dataframe = dataframe.copy()
    df = pd.DataFrame(clean_dataframe[column].dropna()).reset_index()
    
    #Initialize the values and structures to be used
    df['median'] = np.nan
    df['std'] = np.nan
    labels_pred = []
    
    # Calculating update frequences slices, dividing into chunks according to the choosen update frequency
    index_slices = list(sliced(range(len(df)), update_frequency))
    prior_slice = index_slices[0]
    priors = 0
    # For every slice calculate the outliers and re-calculate the initial window
    for index_slice in index_slices:
        
        # Calculate the median and std of the slice using the window that is selected
        df.loc[index_slice,'median'] = df[column][int(index_slice.start-priors):index_slice.stop].rolling(init_window).median()[init_window:]
        df.loc[index_slice,'std'] = df[column][int(index_slice.start-priors):index_slice.stop].rolling(init_window).std()[init_window:]
    
        # Compare medians and std to identify outliers according to the tolerance that has been selected
        diffs = np.asarray(df[column][index_slice].astype('float') - df['median'][index_slice])
        # Relu-like transformation to penalize only the positive values
        df_outliers = np.where(diffs<0, 0 ,diffs)/ (df['std'][index_slice] + 1)# | 
                           # (df[column][index_slice].astype('float') < df['median'][index_slice] - tolerance * df['std'][index_slice])]
        #Populate the labels list.
        labels_pred.extend(df_outliers)
        # Update the window according to the selected method.
        init_window = int(t_update_window(df[column][prior_slice].astype('float'),df[column][index_slice].astype('float'),init_window,ct))
        prior_slice = index_slice 
        priors = init_window
    scaler = MinMaxScaler()
    labels_pred = scaler.fit_transform(np.asarray(labels_pred).reshape(-1,1))
    return labels_pred

def t_update_window(prev_vals,current_vals,window,ct,bounds = [1,700]):
    """
    This function is responsible for conducting the T-Test and determining the
    appropriate changes fow the window.
   ----------------------------------------------------------------------------
   Arguments
   ----------------------------------------------------------------------------
        prev_vals: The previouses chuncks values.
        current_vals: The current chunk values.
        window: The window size at the moment
        ct: The size of the change to apply to the window if needed.
        boundes: The minimum-maximum value that the window can have.
    *******
    Output:
        The new updated window size
    """
    import scipy.stats as stats
    inc = 0
    t_val,p_val = stats.ttest_ind(prev_vals, current_vals)
    
    if (p_val>=0.05):
        inc = 0
    elif (p_val<0.05):
        if (t_val>0):
            inc = - ct
        else:
            inc = ct         
    window = window_adjustment(window,inc,bounds)
    return window

def window_adjustment(window,inc,bounds):
    """
    This is a universal function to apply window changes in respect to the 
    existing boundaries.
    ---------------------------------------------------------------------------
    Arguments
    ---------------------------------------------------------------------------
        window: The window size at the moment.
        inc: The sighted change to apply to the original window.
        bounds: The minimum-maximum value that the window can have.
    *******
    Output:
        The new window with the appropriate changes in respect to the bounds.
    """
    if (window + inc) in range(min(bounds),max(bounds)):
        return window + inc
    elif (window + inc) > max(bounds):
        return np.round(window/2)
    else:
        return min(bounds)
