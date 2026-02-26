import numpy as np
from scipy import stats
from config import get_args
from utils import makedirs, make_model_path
from visualize import *

# load the saved feature values
def load_and_plot_histogram(c):
    load_dir = os.path.join(c.model_dir, c.class_name, 'feat_hist')
    
    feat_numpy = np.load(os.path.join(load_dir, 'feat_maps.npz'), allow_pickle=True)
    train_feats = feat_numpy['train_feat_maps']
    test_feats = feat_numpy['test_feat_maps']
    gt = feat_numpy['gt']

    if len(c.att_type)==0:
        feat_num = int(len(train_feats)/len(c.pool_layers))

        f_init = 0 
        for l in range(len(c.pool_layers)):
            f_last = f_init+feat_num
            save_plots(train_feats[f_init:f_last], test_feats[f_init:f_last], gt, load_dir, l)
            f_init = f_last
    else:
        normal_layers = list(set(c.pool_layers)-set(c.att_layers)) 
        total_feat_num = len(train_feats) 
        normal_feat_num = 1 
        att_feat_num = int((total_feat_num-len(normal_layers)*normal_feat_num)/len(c.att_layers))

        f_init = 0 
        for l in range(len(c.pool_layers)):
            if c.pool_layers[l] in c.att_layers:
                f_last = f_init+att_feat_num
            else:
                f_last = f_init+normal_feat_num
            save_plots(train_feats[f_init:f_last], test_feats[f_init:f_last], gt, load_dir, l)
            f_init = f_last

# save histogram plots of feature values for train set and test set 
def save_plots(train_feat, test_feat, gt, save_dir, l_idx):
    plt.rcParams.update({'font.size': 4})
#    fig, axes = plt.subplots(2, len(train_feat), sharex='all', sharey='all', figsize=(8*len(train_feat)*cm, 4*cm))
    fig, axes = plt.subplots(2, len(train_feat), sharex='all', figsize=(8*len(train_feat)*cm, 4*cm))

    col_labels = ['min','max','median', 'mean', 'std']
    row_labels = ['TRAIN', 'TYP', 'ANO']
    row_colors=['b', 'g', 'r']

    titles = ['diff', 'SpCo_diff','transformed_diff', 'transformed']
    image_file = os.path.join(save_dir, f'hist_feat{l_idx}_pix_with_trainset.png')
    gt_array = np.array(np.mean(np.uint8(gt>0), axis=1), dtype=np.uint8)
    for f_idx in range(len(train_feat)):
        if len(train_feat)==1:
            ax = axes[0]
            ax_tab = axes[1]
        else:
            ax = axes[0, f_idx]
            ax_tab = axes[1, f_idx]
        train_feat_ = np.array(train_feat[f_idx])
        test_feat_ = np.array(test_feat[f_idx])

        Y = test_feat_.flatten()
        Y_label = gt_array.flatten()
#        defect_num = np.sum(Y_label==1)
        Y_train = train_feat_.flatten() 

        table_vals = [
                [np.min(Y_train), np.max(Y_train), np.median(Y_train), np.mean(Y_train), stats.tstd(Y_train)], 
                [np.min(Y[Y_label==0]), np.max(Y[Y_label==0]), np.median(Y[Y_label==0]), np.mean(Y[Y_label==0]), stats.tstd(Y[Y_label==0])],
                [np.min(Y[Y_label==1]), np.max(Y[Y_label==1]), np.median(Y[Y_label==1]), np.mean(Y[Y_label==1]), stats.tstd(Y[Y_label==1])], 
                ]
        table_texts = []
        for row in table_vals:
            table_texts.append([f'{val:0.4e}' for val in row])

        ax_tab.axis("off")
        tab = ax_tab.table(cellText=table_texts,
                             rowLabels=row_labels,
                             colLabels=col_labels,
                             rowColours=row_colors,
                             loc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(3)
        ax.hist([Y_train, Y[Y_label==0], Y[Y_label==1]], 500, density=True, color=row_colors, alpha=0.3, histtype='stepfilled')
        ax.set_title(titles[f_idx])
    plt.subplots_adjust(left=0.2, bottom = 0.1)
    fig.savefig(image_file, dpi=dpi*len(train_feat), format='png', bbox_inches = 'tight')
    plt.close()

if __name__ == '__main__':
    c = get_args()
    c.model_dir = make_model_path(c)
    load_and_plot_histogram(c)
