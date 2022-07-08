from more_itertools import sliced
import pandas as pd
import numpy as np 
from sklearn.metrics import f1_score

def rolling_median(dataframe, column, init_window, tolerance, thresholds = None, labels = None, use_F1 = True, update_frequency = 100, ct = 10):
    """
    This function is responsible for indentifing outliers in the time series based on 
    the median and the standard deviation of a rolling window with dynamic size.
    --------------------------------------------------------------------------
    Argsuments
    ---------------------------------------------------------------------------
    dataframe: The dataframe of the particular metric
    column:    The particular sensor
    window:    The window to be used in order to indentify outliers
    tolerance: The times std is added to mean in order to flag an outlier, when 
               large little points are identified as outliers.
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
    prior_f1 = -1
    
    # Calculating update frequences slices, dividing into chunks according to the choosen update frequency
    index_slices = list(sliced(range(len(df)), update_frequency))
    prior_window = index_slices[0]
    
    # Identify normal and abnormal values based on pre-set thresholds or pass the actual labels
    if use_F1 :
        if labels is None:
            actuals = actual_labels(df,column,thresholds)
        else:
            actuals = labels
    
    # For every slice calculate the outliers and re-calculate the initial window
    for index_slice in index_slices:
        
        # Calculate the median and std of the slice using the window that is selected
        df.loc[index_slice,'median'] = df[column][int(index_slice.start-init_window):index_slice.stop].rolling(init_window).median()[init_window:]
        df.loc[index_slice,'std'] = df[column][int(index_slice.start-init_window):index_slice.stop].rolling(init_window).std()[init_window:]
    
        # Compare medians and std to identify outliers according to the tolerance that has been selected
        df_outliers = df.loc[index_slice][(df[column][index_slice].astype('float') > df['median'][index_slice] + tolerance * df['std'][index_slice])]# | 
                           # (df[column][index_slice].astype('float') < df['median'][index_slice] - tolerance * df['std'][index_slice])]
        
        #Populate the labels list.
        labels_pred.extend([1 if i in df_outliers.index else 0 for i in index_slice])
        
        # Update the window according to the selected method.
        if use_F1:
            init_window = int(update_window(prior_f1,f1_score(actuals[index_slice.start:index_slice.stop],
                                                             labels_pred[index_slice.start:index_slice.stop],zero_division=1),init_window,ct))
            prior_f1 = f1_score(actuals[index_slice.start:index_slice.stop],labels_pred[index_slice.start:index_slice.stop],zero_division=1)
        else:
            init_window = int(t_update_window(df[column][prior_window].astype('float'),df[column][index_slice].astype('float'),init_window,ct))
            prior_window = index_slice 
            
    return labels_pred

def update_window(prior_f1,current_f1,window,ct,bounds = [100,650]):
    """Adapts the window based on the changes in f1 score.
       Arguments:
       prior_f1: The f1 score of the previous window
       current_f1: The f1 score of the current window
       window: The value of the window at the current iteration
       ct: The static value to increase or decrease the value of
           the window 
    """
    inc = 0
    if (prior_f1 == current_f1):
        inc = - ct
    elif (prior_f1 > current_f1):
        inc = ct
    else:
        inc = 0
        
    window = window_adjustment(window,inc,bounds)
    return window
    
def actual_labels(df,column,thresholds):
    """Outputs the actual labels based on the provided thresholds
        Arguments:
        df: the dataframe to perform the label extraction upon
        column: the specific column to use
        thresholds: a list of the uper and lower bounds on a 
                    percieved as normal value
    """
    actuals = df[column].dropna().astype('float') > max(thresholds)
    actuals = [1 if i is True else 0 for i in actuals]
    return actuals

def t_update_window(prev_vals,current_vals,window,ct,bounds = [50,700]):
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
    
def multivariate_median(df, init_window, tolerance, thresholds = None, labels = None, use_F1 = True, update_frequency = 100, ct = 10):
    """
    TODO : make it work for multivariate series with the f1 functionality (probably can not)
    """
    
    """
    Function to apply rolling median to multivariate time series. This function runs individually
    the RM algorithm in each time series and then outputs a list with all indexes of the dataframe
    that identified as anomalies.
    """
    labels = np.zeros(len(df))
    for column in df.columns:
        y = rolling_median(df, column, init_window, tolerance, thresholds, labels, use_F1, update_frequency, ct)
        labels = np.add(labels, y)
    mv = [1 if i > 0 else 0 for i in labels]
    return mv

def window_adjustment(window,inc,bounds):
    """
    Universal function to apply window changes in respect to the existing boundaries.
    """
    if (window + inc) in range(min(bounds),max(bounds)):
        return window + inc
    elif (window + inc) > max(bounds):
        return np.round(window/2)
    else:
        return min(bounds)