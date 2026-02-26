from __future__ import print_function
import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser(description='PCB Inspection')
    parser.add_argument('-s', '--seed', default=None, type=int,
                        help='the random seed number (default: the integer part of timestamp)')
    parser.add_argument('-re', '--repeatability', default=None, type=int,
                        help='the random seed number (default: the integer part of timestamp)')
    parser.add_argument('-run', '--run-name', default=0, type=int, 
                        help='name of the run (default: 0)')
    parser.add_argument("--gpu", default='0', type=str, 
                        help='GPU device number')
    parser.add_argument('--no_cuda', type=str2bool, default='no', 
                        help='disables CUDA training (default: no)')
    parser.add_argument('--workers', default=4, type=int, 
                        help='number of data loading workers (default: 4)')


    # configure settings
    parser.add_argument('--data_path', default='/home/dspl/nyla/Anomaly_with_NF/Dataset/plain/ATI/PCB/cropped_data/overlapped_30', type=str, 
                        help='path of dataset (default: ./datasets/processed_data)')
    parser.add_argument('--aug_data_path', default='/home/dspl/nyla/Anomaly_with_NF/Dataset/aug_data/ATI/PCB/cropped_data/overlapped_30', type=str, 
                        help='path of dataset (default: ./datasets/processed_data)')
    parser.add_argument('--data_cfg_dir', default='./configs/data', type=str, 
                        help='directory for configuraion files for constructing dataset (default: ./configs/data)')
    parser.add_argument('-ds', '--data_settings', default='2CP_2CPv1', type=str, 
                        help='dataset settings (train/test): {trian_product}_{test_product}v{version} (default: 2CP_2CPv1)')
    parser.add_argument('--model_cfg_dir', default='./configs/model', type=str, 
                        help='directory for configuraion files for constructing network (default: ./configs/model)')
    parser.add_argument('-arch', '--network_arch', default='visformer_tiny-skip123-dec8_glow', type=str, 
                        help='network architecture: resnet18-skip123-dec8_glow (default: resnet18-skip123-dec8_glow)')


    # data settings 
    parser.add_argument('-inp', '--input_size', default=256, type=int, 
                        help='image resize dimensions (default: 256)')
    parser.add_argument('--norm_mean', type=float, default=[0.485, 0.456, 0.406], nargs='+')
    parser.add_argument('--norm_std', type=float, default=[0.229, 0.224, 0.225], nargs='+')

    parser.add_argument('--aug_ratio_train', type=float, default=1.0, 
                        help='the ratio of synthetic defect data for training (default: 1.0)')
    parser.add_argument('--nf_aug_ratio_train', type=float, default=0, 
                        help='the ratio of synthetic defect data for training (default: 0.2)')
    parser.add_argument('--use_in_domain_data', type=float, default=0.97, 
                        help='use the in-domain dataset for generating synthetic defect data with dtd data together (50% selection, default: yes)')
    parser.add_argument('--anomaly_size', type=float, default=0.83, 
                        help='')
    parser.add_argument('--anomaly_confidence', type=float, default=0.5, 
                        help='')
    parser.add_argument('--repeat_num', type=int, default=5, 
                        help='period to generate new synthetic defect dataset (default: 5)')
    parser.add_argument('--drop_last', type=str2bool, default='no', 
                        help='do not train the last batch if the size of last minibatch is smaller than batch size (default: no)')


    # network settings 
    parser.add_argument('--model_path', default='./../models', type=str, 
                        help='root directory to save or load models (default: ./../models)')
    parser.add_argument('-nf_inp', '--nf_input', default=['diff'], type=str, nargs = '+', 
                        help='input features of normalizing flow model (default: ["diff", "transformed_diff"])')
    parser.add_argument('-nl', '--nf_layers', type=int, default=[1,2,3], nargs='+',
                        help='number of layers used in NF model (default: [1,2,3])')
    parser.add_argument('-al', '--att_layers', type=int, default=[], nargs='+',
                        help='number of attention layers (default: [])')
    parser.add_argument('--att_type', default='', type=str, 
                        help='the type of attention (default:)')
    parser.add_argument('--pretrained', type=str2bool, default='yes', 
                        help='initialize the weight of feature extractor by that of pretrained network on ImageNet (default: yes)')
    parser.add_argument('--finetuning', type=str2bool, default='yes', 
                        help='initialize the weight of feature extractor by that of pre-trained network on ImageNet (default: yes)')
    parser.add_argument('--eval_bn', type=str2bool, default='yes', 
                        help='determine whether updating the sample mean and standard deviztion for batch normalization layers of a feature extractor when fine-tuning it. (default: yes)')
    parser.add_argument('--freeze_bn', type=str2bool, default='yes', 
                        help='determine whether training batch normalization layers of a feature extractor when fine-tuning it. (default: yes)')
    parser.add_argument('--r_ch', type=int, default=8, 
                        help='channel reduction for shallow decoder (default: 8)')


    # program settings 
    parser.add_argument('--train_type', type=str, default='twin_fe_only', choices = ['single_fe_only', 'twin_fe_only', 'single_nf_only', 'twin_nf_only'], 
                        help='set which networks do you want to train. default: twin_fe_only')
    parser.add_argument('--infer_type', type=str, default='twin_fe_only', choices = ['single_fe_only', 'twin_fe_only', 'single_nf_only', 'twin_nf_only', 'single_joint', 'twin_joint'], 
                        help='set which networks do you want to test. default: twin_fe_only')
    parser.add_argument('--test_data_type', type=str, default='real', choices = ['aug', 'real'], 
                        help = 'set which dataset do you want to evaluate. (aug: synthetic defect dataset for evaluation, real: real test datastet provided from MVTecAD dataset, default: real)')
    parser.add_argument('--loss_type', type=str, default='smooth_cls', choices = ['cls', 'reg', 'smooth_cls'], 
                        help='set which task do you want to fine-tune the encoder. (cls: pixelwise classification with hard labels, reg: pixelwise regression network, smooth_cls: pixelwise classification network with soft labels, default: smooth_cls)')


    # hyper parameters 
    parser.add_argument('-bs', '--batch-size', default=32, type=int,
                        help='train batch size (default: 32)')
    parser.add_argument('--meta_epochs', type=int, default=25,
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=8,
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('--eval_epoch', type=int, default=25,
                        help='')
    parser.add_argument('--freeze_enc_epochs', type=int, default=5, 
                        help='number of epochs not to train an encoder (default: 5)')
    parser.add_argument("--is_train", default='yes', type=str2bool,
                        help=' yes-train/no-inference (default: yes)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_nf_list', type=float, nargs='+', default=None,
                    help='LR per NF')
    parser.add_argument('--lr_decay_epochs_percentage', type=float, default=[0.9], nargs='+', 
                        help='epochs to decay learning rate for the StepLR decay scheduling.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help = 'multiplicative factor of learning rate decay. (default: 0.1)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, 
                        help='epochs to increase learning rate up to the initial learning rate in early training steps. (default: 2)')
    parser.add_argument('--lr_warm', type=str2bool, default='yes', 
                        help= 'determine whether using warmup learning rate scheduling. (default: yes)')
    parser.add_argument('--lr_cosine', type=str2bool, default='yes', 
                        help='yes-CosineAnnealingLR scheduling, no-StepLR scheduling. (default: yes)')
    parser.add_argument('--w_defect', type=float, default=0.9,
                        help='the weight of defect class for weighted cross-entropy. (default:0.9)')
    parser.add_argument('--w_decay', type=float, default=1e-4,
                        help='the hyperparameter for the weight decay regularization. (default:1e-4)')


    # inference settings
    parser.add_argument('--get_best_w_fe', type=str2bool, default='no', 
                        help='find the best weight to balance the anomaly scores of the pixelwise classification and CNF networks. (default: no)')
    parser.add_argument('--w_fe', type=float, default=0, 
                        help='the weight to balance the anomaly scores of the pixelwise classification and CNF networks.')
    parser.add_argument('--feat_avg_topk', type=float, default=1.0, 
                        help='select feature maps with the k larges averaged values and average them to visualize the features.')
    parser.add_argument('--add_fe_anomaly', type = str2bool, default='no',
                        help='determine whether combining the score maps of a pixelwise classification network and CNF networks.')
    parser.add_argument('--init_chip', default=0, type=int, 
                        help='the initial chip index to inference. (default: 0)')
    parser.add_argument('--fin_chip', default=-1, type=int,
                        help='the final chip index to inference. (default: -1)')
    parser.add_argument('--th_manual', type = float, default=0,
                        help='manual threshold value to make predictions from anomaly scores. (default: 0)')
    parser.add_argument('--is_full', type = str2bool, default='no',
                        help='yes- test full chip, no-test input image of a network (default: no)')
    parser.add_argument('--save_all_tn_sample', type = str2bool, default='no',
                        help='yes- save true negative samples, no- do not save true negative samples (default: no)')
    parser.add_argument('--save_features', type = str2bool, default='no',
                        help='determine whether saving values of total feature maps in .npz file (default: no)')
    parser.add_argument('--viz_total_features', type = str2bool, default='no',
                        help='determine whether visualizing total feature maps (default: no)')
    parser.add_argument('--viz_diff_features', type = str2bool, default='no',
                        help='determine whether visualizing feature map differences (default: no)')

    parser.add_argument('--is_close', type = str2bool, default='no',
                        help='determine whether applying the morphological closing operation to the prediction map. (default: no)')
    parser.add_argument('--is_open', type = str2bool, default='yes',
                        help='determine whether applying the morphological opening operation to the prediction map. (default: yes)')
    parser.add_argument('--is_k_disk', type = str2bool, default='yes',
                        help='determine whether using the disk type mask for the morphological operations. (default: yes)')
    parser.add_argument('--k_size', default=1, type=int, 
                        help='the hyperparameter for the size of the mask for the morphological operations. (default: 1)')
    parser.add_argument('--th_pix', type = int, default=30, 
                        help='the pixel threshold to make predictions as defects')
    parser.add_argument('--fn', type = int, default=-1, 
                        help='the number of false negatives to determine a threshold value for anomaly scores')
    parser.add_argument('--pro',  type = str2bool, default='no',
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz',  type = str2bool, default='no',
                        help='saves test data visualizations')
    parser.add_argument('--com_num', default=23, type=int)


    # output settings
    parser.add_argument('--verbose', type=str2bool, default='yes')
    parser.add_argument('--hide_tqdm_bar', type=str2bool, default='yes')
    parser.add_argument('--save_results', type=str2bool, default='yes')

    args = parser.parse_args()
    return args


def str2bool(v):
   if isinstance(v, bool):
       return v
   if v.lower() in ('yes', 'True', 't', 'y'):
       return True
   elif v.lower() in ('no', 'false', 'f', 'n'):
       return False
   else:
       raise argparse.ArgumentTypeError('Boolean value expected.')
