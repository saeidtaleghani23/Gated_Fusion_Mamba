import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed, train_2_patch_label2, show_calaError
import argparse
import os
import time
from CALC_Saeid import train_network

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("CALC")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=16, help='number of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
parser.add_argument('--train_samples', type=int, default=100, help='number of training samples per class')
parser.add_argument('--fold', type=int, default=1, help='cross-validation fold to use')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def train_1times():
    # -------------------------------------------------------------------------------
    # prepare data based on dataset
    if args.dataset == 'Houston':
        base_path = '/beluga/Hackathon15/dataset/Houston2013'
        data_file = os.path.join(base_path, f'train_test_{args.train_samples}', 'houston2013_data.mat')
        split_file = os.path.join(base_path, f'train_test_{args.train_samples}', f'train_test_gt_{args.fold}.mat')
        
        # Load data
        data_mat = scio.loadmat(data_file)
        Data1 = data_mat['hsi_data'].astype(np.float32)  # HSI data
        Data2 = data_mat['lidar_data'].astype(np.float32)  # LiDAR data
        
        # Load train/test split
        split_mat = scio.loadmat(split_file)
        TrLabel = split_mat['train_data']  # Training mask
        TsLabel = split_mat['test_data']   # Testing mask
        
        # Set number of classes for Houston
        args.num_classes = 15
        dataset_name = "Houston2013"
        
    elif args.dataset == 'Trento':
        base_path = '/beluga/Hackathon15/dataset/Trento'
        data_file = os.path.join(base_path, f'train_test_{args.train_samples}', 'trento_data.mat')
        split_file = os.path.join(base_path, f'train_test_{args.train_samples}', f'trento_train_test_gt_{args.fold}.mat')
        
        # Load data
        data_mat = scio.loadmat(data_file)
        Data1 = data_mat['hsi_data'].astype(np.float32)  # HSI data
        Data2 = data_mat['lidar_data'].astype(np.float32)  # LiDAR data
        
        # Load train/test split
        split_mat = scio.loadmat(split_file)
        TrLabel = split_mat['train_data']  # Training mask
        TsLabel = split_mat['test_data']   # Testing mask
        
        # Set number of classes for Trento
        args.num_classes = 6
        dataset_name = "Trento"
    
    # Ensure Data2 has proper shape (important for LiDAR with 1 band)
    if len(Data2.shape) == 2:
        Data2 = Data2.reshape([Data2.shape[0], Data2.shape[1], -1])
    
    [m1, n1, l1] = np.shape(Data1)
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    
    # data size
    print("Dataset: {}".format(dataset_name))
    print("HSI shape: height={}, width={}, bands={}".format(height1, width1, band1))
    print("LiDAR shape: height={}, width={}, bands={}".format(height2, width2, band2))
    print("Number of classes: {}".format(args.num_classes))
    
    # Print label statistics
    train_pixels = np.sum(TrLabel > 0)
    test_pixels = np.sum(TsLabel > 0)
    print("Training samples: {}, Testing samples: {}".format(train_pixels, test_pixels))
    
    # Create model save directory
    model_save_dir = os.path.join('trained_models', dataset_name)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'CALC_{dataset_name}.pkl')
    
    # sample generation
    patchsize = args.patch_size
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    
    TrainPatch1, TrainPatch2, TrainLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TrLabel)
    TestPatch1, TestPatch2, TestLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TsLabel)
    
    train_dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print('HSI Training size and testing size:', TrainPatch1.shape, TestPatch1.shape)
    print('LiDAR Training size and testing size:', TrainPatch2.shape, TestPatch2.shape)
    print('Label Training size and testing size:', TrainLabel.shape, TestLabel.shape)
    print('Model will be saved to:', model_save_path)
    
    # -------------------------------------------------------------------------------
    # train and test
    tic1 = time.time()
    
    # Pass model_save_path to train_network function
    pred_y, val_acc = train_network(train_loader, TrainPatch1, TrainPatch2, TrainLabel,
                                    TestPatch1, TestPatch2, TestLabel, LR=args.learning_rate,
                                    EPOCH=args.epoches, patchsize=args.patch_size, l1=band1, l2=band2,
                                    Classes=args.num_classes, model_save_path=model_save_path)
    
    pred_y = pred_y.type(torch.FloatTensor)
    TestLabel = TestLabel.type(torch.FloatTensor)
    
    print("Maximal Accuracy: {:.4f}, at epoch: {}".format(max(val_acc), val_acc.index(max(val_acc))))
    toc1 = time.time()
    time_1 = toc1 - tic1
    print('Training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    
    OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))


if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()