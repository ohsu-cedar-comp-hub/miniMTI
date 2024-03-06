def get_channel_info(include_he=False):
    """returns lists of channel names and indices that are going to be kept, as well as a dictionary mapping maker names to indices"""
    RoundsCyclesTable = '../data/RoundsCyclesTable.txt'
    with open(RoundsCyclesTable) as f:
        channels = []
        for l in f.readlines():
            ch_name = l.split(' ')[0]
            if l.split(' ')[2] == 'c2': channels.extend([f"DAPI_{l.split(' ')[1]}", ch_name])
            else: channels.append(ch_name)

    keep_channels = [ch for ch in channels if ('DAPI' not in ch) or (('DAPI' in ch) and (ch == 'DAPI_R1'))]
    keep_channels_idx = [i for i,ch in enumerate(channels) if ch in keep_channels]
    ch2idx = {ch:i for i,ch in enumerate(keep_channels)}

    return keep_channels, keep_channels_idx, ch2idx