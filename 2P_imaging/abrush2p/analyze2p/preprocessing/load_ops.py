import suite2p

def load_ops(exp = "bruker_jRGECO1a_GRABDA3m", denoise = True):

    """
    Load the options for Suite2P preprocessing.
    You may find the descriptions for each parameters in the options [here]: 
    (https://suite2p.readthedocs.io/en/latest/settings.html)
    
    
    Returns:
        dict: A dictionary containing the options for Suite2P.
    """
    
    ops = suite2p.default_ops()
    ops['denoise'] = denoise

    # ---- the following options are for jRGECO1a and GRABDA3m dual channel imaging ----
    if exp == "bruker_jRGECO1a_GRABDA3m":
        ops['filetype']           = 'tif'
        ops['nchannels']          = 2 # Number of channels (1 for jRGECO1a, 1 for GRABDA3m)
        ops['functional_chan']    = 1 # Channel to use for functional imaging (1 for jRGECO1a)
        ops['tau']                = 0.7 # tau for jRGECO1a
        ops['fs']                 = 7.5  # Sampling frequency for imaging acquisition: interval: 0.135
        ops['move_bin']           = True  # Bin size for motion correction
        ops['nplanes']            = 1  # Number of imaging planes
        ops['align_by_chan']      = 2  # Align by the second channel (GRABDA3m)
        ops['spatial_scale']      = 0  # Diameter of the region of interest (ROI) for cell detection
        ops['input_format']       = 'bruker'  # Input format for the imaging data
        ops['sparse_mode']        = True  # Use sparse mode for faster processing
        ops['look_one_level_down']= True  # Look for data in subdirectories
    return ops