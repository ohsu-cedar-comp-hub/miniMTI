def get_channel_info():
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    channels = ['DAPI','TRITC','Cy5','TOMM20','CD90','CD45','ERG','HLA-DR','CD11b','CD3','AR','TUBB3','GZMB',
                'Ecad','CK5','CD68','TH','aSMA','AMACR','CD56','NFKB','HIF-1','CD4','FOXA1','ADAM10','DCX',
                'CD11c','CD20','Ki67','CD8','CD31','VIM','NRXN1','NLGN4X','ChromA','TRYP','CD44','NLGN1','CK8',
                'B-catenin','H3K4','H3K27ac','CD163']
    keep_channels = channels
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx
