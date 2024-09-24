def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    channels = ['DAPI_R1','aSMA','Tryp','Ki67', 'CD68','AR','CD20','ChromA','CK5','HLADRB1','CD3',
                'CD11b','CD4','CD45','CD163','CD66b','PD1','GZMB','NKX31','CK8','AMACR','FOXP3','CD8',
                'EPCAM','CD56','NCR1','ERG','CK14','ECAD','VIM','FOSB','CD31','Tbr2','CD45RA','p53',
                'CD45RO','FOXA1','CDX2','HOXB13','NOTCH1']
    keep_channels = channels
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx
