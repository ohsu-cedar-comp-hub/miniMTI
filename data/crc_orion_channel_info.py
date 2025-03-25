def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    channels = [
        'DAPI',
        'AF1',
        'CD31',
        'CD45',
        'CD68',
        'Argo550',
        'CD4',
        'FOXP3',
        'CD8a',
        'CD45RO',
        'CD20',
        'PD-L1',
        'CD3e', #Cd3?
        'CD163',
        'E-cadherin',
        'PD-1',
        'Ki67',
        'PanCK',
        'aSMA']

    keep_channels = [ch for ch in channels if ch != "AF1" and ch != 'Argo550']
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}
    ch2idx['HE'] = 17
    #ch2idx['HE2'] = 18

    return keep_channels, keep_channels_idx, ch2idx