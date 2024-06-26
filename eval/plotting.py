import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def plot_scatter(mints, pmints, corrs_per_marker, masked_ch_idx, ch2stain, save_id):
    '''generates a scatter plot of mean intensities vs. predicted mean intensities for each predicted marker'''
    #cls iteratively selected panel
    fig, ax = plt.subplots(1, len(masked_ch_idx), figsize=(12 * len(masked_ch_idx), 12), layout='tight')
    for i,a in enumerate(fig.axes):
        a.set_title(f'{ch2stain[masked_ch_idx[i]]}\n(⍴={round(corrs_per_marker[i].item(),2)})', fontsize=120)
        x,y = mints[:20000,i].cpu(), pmints[:20000,i].cpu()
        # Calculate the point density
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        a.scatter(x, y, c=z, s=10)
        a.set_xticks([])
        a.set_yticks([])
        a.plot(np.arange(255), np.arange(255), linestyle='dashed', c='black')
    plt.savefig(f"plots/{save_id}/scatter_{len(masked_ch_idx)}_masked.png")

    
def plot_heatmap(corrs_per_panel_size, panel_order, NUM_CHANNELS, ch2stain, save_id):
    masked_chs = []
    for panel_size in range(1, NUM_CHANNELS):
        msk_chs = [i for i in range(NUM_CHANNELS) if i not in panel_order[:panel_size]]
        masked_chs.append(msk_chs)

    corr_array = np.ones((NUM_CHANNELS, NUM_CHANNELS))
    for i,(panel,panel_i) in enumerate(zip(corrs_per_panel_size, masked_chs)):
        if i == NUM_CHANNELS - 2:
            panel = [panel.item()]
            panel_i = list(panel_i)
        for j,(ch, ch_i) in enumerate(zip(panel,panel_i)):
            corr_array[i][ch_i] = ch

    last_panel = panel_order
    still_masked = [i for i in range(NUM_CHANNELS) if i not in last_panel]
    sorted_channels = last_panel + still_masked

    corr_array_sorted = np.zeros(corr_array.shape)
    ch2stain_sorted = {}
    for i,ch in  enumerate(sorted_channels):
        corr_array_sorted[:,i] = corr_array[:,ch]
        ch2stain_sorted[i] = ch2stain[ch]

    fig, ax = plt.subplots(figsize=(NUM_CHANNELS, NUM_CHANNELS))
    img = corr_array_sorted[:NUM_CHANNELS]
    img = img.transpose()
    matrix = np.triu(img)
    im = sns.heatmap(img, cmap='inferno', vmin=0, vmax=1, mask=matrix)
    im.collections[0].colorbar.ax.tick_params(labelsize=24)
    ax.set_xticks(np.arange(NUM_CHANNELS) + 0.5,list(ch2stain_sorted.values()), fontsize=32)
    ax.set_yticks(np.arange(NUM_CHANNELS) + 0.5,list(ch2stain_sorted.values()), fontsize=32)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.xlabel('Next Marker Added to Panel', fontsize=36)
    plt.ylabel('Predicted Marker', fontsize=36)
    plt.savefig(f'plots/{save_id}/heatmap.png')
    
    
def plot_hist(mints, pmints, masked_ch_idx, ch2stain, save_id):
    fig, ax = plt.subplots(2, len(masked_ch_idx), figsize=(64,8))
    markers = [ch2stain[i] for i in masked_ch_idx]
    if len(markers) == 1:
        ax[0].hist(mints[:].cpu(), bins=1000, range=(0,255))
        ax[0].set_title(f'real {marker}')
        ax[1].hist(pmints[:].cpu(), bins=1000, range=(0,255))
        ax[1].set_title(f'predicted {marker}')
    else:    
        for i,marker in enumerate(markers):
            ax[0,i].hist(mints[:,i].cpu(), bins=1000, range=(0,255))
            ax[0,i].set_title(f'real {marker}')
            ax[1,i].hist(pmints[:,i].cpu(), bins=1000, range=(0,255))
            ax[1,i].set_title(f'predicted {marker}')
    plt.savefig(f"plots/{save_id}/histograms_{len(masked_ch_idx)}_masked.png")
    
    
def plot_violin(corrs_per_panel_size, NUM_CHANNELS, save_id):
    fig, ax = plt.subplots()
    bplot = plt.violinplot(corrs_per_panel_size, positions = np.arange(NUM_CHANNELS - 1), showmeans=True)
    ax.set_xticks(np.arange(NUM_CHANNELS - 1), np.arange(NUM_CHANNELS - 1)+1, fontsize=7)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    plt.xlabel('Reduced Panel Size')
    plt.ylabel('Mean Spearman Correlation of Witheld Marker Intensities')
    plt.savefig(f'plots/{save_id}/barplots.png')