import numpy as np


######## BIN_SPIKES ########
def bin_spikes(spike_times,dt,wdw_start,wdw_end):
    """
    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    num_neurons=spike_times.shape[0] #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data



######## BIN_OUTPUT #######
def bin_output(outputs,output_times,dt,wdw_start,wdw_end,downsample_factor=1):
    """
    Function that puts outputs into bins

    Parameters
    ----------
    outputs: matrix of size "number of times the output was recorded" x "number of features in the output"
        each entry in the matrix is the value of the output feature
    output_times: a vector of size "number of times the output was recorded"
        each entry has the time the output was recorded
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for binning the outputs
    wdw_end: number (any format)
        the end time for binning the outputs
    downsample_factor: integer, optional, default=1
        how much to downsample the outputs prior to binning
        larger values will increase speed, but decrease precision

    Returns
    -------
    outputs_binned: matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    ###Downsample output###
    #We just take 1 out of every "downsample_factor" values#
    if downsample_factor!=1: #Don't downsample if downsample_factor=1
        downsample_idxs=np.arange(0,output_times.shape[0],downsample_factor) #Get the idxs of values we are going to include after downsampling
        outputs=outputs[downsample_idxs,:] #Get the downsampled outputs
        output_times=output_times[downsample_idxs] #Get the downsampled output times

    ###Put outputs into bins###
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1 #Number of bins
    output_dim=outputs.shape[1] #Number of output features
    outputs_binned=np.empty([num_bins,output_dim]) #Initialize matrix of binned outputs
    #Loop through bins, and get the mean outputs in those bins
    for i in range(num_bins): #Loop through bins
        idxs=np.where((np.squeeze(output_times)>=edges[i]) & (np.squeeze(output_times)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
        for j in range(output_dim): #Loop through output features
            outputs_binned[i,j]=np.mean(outputs[idxs,j])

    return outputs_binned


###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X

def get_spikes_flexible_bins(neural_data, bin_array):
    """
    Function that creates the covariate matrix of neural activity.
    Based on 'get_spikes_with_history', but with flexible bin selection: instead of a continuous bin selection, can
    skip bins, use only bins in the past or future, etc. Useful for sanity checks and also to see if information about
    behavior is reliably contained in past or future spiking

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_array: array of integers for the bins ot be used relative to the timepoint being predicted.
        Current bin only:                                           [0]
        One two bins to the past plus the current and one future:   [-2, -1, 0, 1] or np.arange(-2,1+1)
        Only 5 bins in the past:                                    [-5]

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples = neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons = neural_data.shape[1] #Number of neurons
    bin_array = np.array(bin_array)
    num_bins = bin_array.shape[0]
    X=np.empty([num_examples,num_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.NaN

    bins_before = np.abs(np.min((0, bin_array.min())))
    bins_after  = np.abs(np.max((0, bin_array.max())))

    bins_current = 0 in bin_array
    bins_span = bins_before + bins_after + bins_current


    for ii in range(num_examples):

        if ii < bins_before:
            continue
        if ii >= num_examples-bins_after:
            break

        X[ii,:,:] = neural_data[bin_array+ii,:]

    return X

def train_test_validation_splits(X, y, train_valid = (0.2, 0.2), shuffle=True, random_state=None):
    '''
    X_train, X_test, X_valid, y_train, y_test, y_valid =
    train_test_validation_splits(X, y, train_valid = (0.2, 0.2), shuffle=True, random_state=None)

    Splits data and output into training, testing and validation sets

    Parameters
    ----------
    X:              data / matrix of predictors, [observations by features]
    y               output variable(s) [observations by variables]
    train_valid     [optional] training and validation splits [tuple, default is (0.2, 0.2)]
    shuffle         [optional] randomly shuffle the data. default is True
    random_state    [optional] integer for the randomization. Specify for reproducibility. default is None

    Returns
    -------
    X_train, X_test, X_valid, y_train, y_test, y_valid

    '''

    train_pct = 1 - train_valid[0]
    adjusted_valid_pct = train_valid[1] / train_pct

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_valid[0],
                                                        random_state = random_state, shuffle = shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=adjusted_valid_pct,
                                                      random_state = random_state, shuffle = shuffle)

    return X_train, X_test, X_valid, y_train, y_test, y_valid
def train_test_validation_splits_no_overlap(X, y, train_valid = (0.2, 0.2), bins_before=0, bins_after=0):
    '''

    X_train, X_test, X_valid, y_train, y_test, y_valid =
    train_test_validation_splits_no_overlap(X, y, train_valid = (0.2, 0.2), bins_before=0, bins_after=0)

    Splits data and output into training, testing and validation sets. If X comes from 'get_spikes_with_history' or
    if X contains partially overlapping data, past and future overlaps should be of 'bins_before' and 'bins_after',
    respectively. This function splits the training, testing and validation sets avoiding overlapping
    IMPORTANT: doesn't support randomization due to the temporal nature of X

    Parameters
    ----------
    X               data / matrix of predictors, [observations by features]
    y               output variable(s) [observations by variables]
    train_valid     [optional] training and validation splits [tuple, default is (0.2, 0.2)]
    bins_before     [optional] number of bins / timepoints to the past
    bins_after      [optional] number of bins / timepoints to the future

    Returns
    -------
    X_train, X_test, X_valid, y_train, y_test, y_valid
    '''

    # adapted from the notebook
    # https://github.com/KordingLab/Neural_Decoding/blob/master/Examples_all_decoders.ipynb

    training_range =    [0,                                     1-(train_valid[0]+train_valid[1])]
    testing_range =     [1-(train_valid[0]+train_valid[1]),     1-train_valid[1]]
    valid_range =       [1-train_valid[1],                      1]

    num_examples = X.shape[0]
    training_set = np.arange(int(np.round(training_range[0] * num_examples)) + bins_before,
                             int(np.round(training_range[1] * num_examples)) - bins_after)
    testing_set = np.arange(int(np.round(testing_range[0] * num_examples)) + bins_before,
                            int(np.round(testing_range[1] * num_examples)) - bins_after)
    valid_set = np.arange(int(np.round(valid_range[0] * num_examples)) + bins_before,
                          int(np.round(valid_range[1] * num_examples)) - bins_after)

    X_train = X[training_set]
    X_test  = X[testing_set]
    X_valid = X[valid_set]

    y_train = y[training_set]
    y_test = y[testing_set]
    y_valid = y[valid_set]

    return X_train, X_test, X_valid, y_train, y_test, y_valid