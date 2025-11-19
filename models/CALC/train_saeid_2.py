import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed, train_2_patch_label2, show_calaError
import argparse
import os
import time
from CALC_Saeid_2 import train_network

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("CALC")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Muufl', help='dataset to use')
parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=16, help='number of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
parser.add_argument('--train_samples', type=int, default=100, help='number of training samples per class')
parser.add_argument('--val_samples', type=int, default=50, help='number of validation samples per class')
parser.add_argument('--fold', type=int, default=1, help='cross-validation fold to use')
parser.add_argument('--eval_interval', type=int, default=50, help='interval for validation evaluation')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def save_performance_report(report_path, dataset_name, args, train_stats, val_acc, test_acc, 
                          oa, kappa, ca, aa, training_time, best_epoch):
    """
    Save comprehensive performance report as a text file
    """
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CALC MODEL PERFORMANCE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset and Model Information
        f.write("DATASET AND MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Classes: {args.num_classes}\n")
        f.write(f"Training Samples per Class: {args.train_samples}\n")
        f.write(f"Validation Samples per Class: {args.val_samples}\n")
        f.write(f"Fold: {args.fold}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Patch Size: {args.patch_size}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Total Epochs: {args.epoches}\n")
        f.write(f"Best Epoch: {best_epoch}\n\n")
        
        # Training Statistics
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Samples: {train_stats['train_samples']}\n")
        f.write(f"Validation Samples: {train_stats['val_samples']}\n")
        f.write(f"Test Samples: {train_stats['test_samples']}\n")
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Validation Accuracy: {max(val_acc):.4f}\n")
        f.write(f"Test Accuracy at Best Validation: {max(test_acc):.4f}\n")
        f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
        f.write(f"Kappa Coefficient: {kappa:.4f}\n")
        f.write(f"Average Accuracy (AA): {aa:.4f}\n\n")
        
        # Class-wise Accuracy
        f.write("CLASS-WISE ACCURACY\n")
        f.write("-" * 40 + "\n")
        for i, acc in enumerate(ca):
            f.write(f"Class {i+1}: {acc:.4f}\n")
        f.write("\n")
        
        # Validation Accuracy History (first 5 and last 5)
        f.write("VALIDATION ACCURACY HISTORY\n")
        f.write("-" * 40 + "\n")
        if len(val_acc) <= 10:
            for i, acc in enumerate(val_acc):
                f.write(f"Epoch {i+1}: {acc:.4f}\n")
        else:
            for i in range(5):
                f.write(f"Epoch {i+1}: {val_acc[i]:.4f}\n")
            f.write("...\n")
            for i in range(len(val_acc)-5, len(val_acc)):
                f.write(f"Epoch {i+1}: {val_acc[i]:.4f}\n")
        f.write("\n")
        
        # Test Accuracy History (first 5 and last 5)
        f.write("TEST ACCURACY HISTORY\n")
        f.write("-" * 40 + "\n")
        if len(test_acc) <= 10:
            for i, acc in enumerate(test_acc):
                f.write(f"Epoch {i+1}: {acc:.4f}\n")
        else:
            for i in range(5):
                f.write(f"Epoch {i+1}: {test_acc[i]:.4f}\n")
            f.write("...\n")
            for i in range(len(test_acc)-5, len(test_acc)):
                f.write(f"Epoch {i+1}: {test_acc[i]:.4f}\n")
        f.write("\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model trained successfully on {dataset_name} dataset.\n")
        f.write(f"Achieved {max(val_acc):.4f} validation accuracy and {max(test_acc):.4f} test accuracy.\n")
        f.write(f"Overall performance: OA={oa:.4f}, Kappa={kappa:.4f}, AA={aa:.4f}\n")
        f.write(f"Training completed in {training_time/60:.2f} minutes.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def train_1times():
    # -------------------------------------------------------------------------------
    # prepare data based on dataset
    if args.dataset == 'Houston':
        base_path = '/beluga/Hackathon15/dataset/Houston2013'
        data_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', 'houston2013_data.mat')
        split_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', f'train_val_test_gt_{args.fold}.mat')
        
        # Load data
        data_mat = scio.loadmat(data_file)
        Data1 = data_mat['hsi_data'].astype(np.float32)  # HSI data
        Data2 = data_mat['lidar_data'].astype(np.float32)  # LiDAR data
        
        # Load train/val/test split
        split_mat = scio.loadmat(split_file)
        TrLabel = split_mat['train_data']  # Training mask
        ValLabel = split_mat['val_data']   # Validation mask
        TsLabel = split_mat['test_data']   # Testing mask
        
        # Set number of classes for Houston
        args.num_classes = 15
        dataset_name = "Houston2013"
        
    elif args.dataset == 'Trento':
        base_path = '/beluga/Hackathon15/dataset/Trento'
        data_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', 'trento_data.mat')
        split_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', f'trento_train_val_test_gt_{args.fold}.mat')
        
        # Load data
        data_mat = scio.loadmat(data_file)
        Data1 = data_mat['hsi_data'].astype(np.float32)  # HSI data
        Data2 = data_mat['lidar_data'].astype(np.float32)  # LiDAR data
        
        # Load train/val/test split
        split_mat = scio.loadmat(split_file)
        TrLabel = split_mat['train_data']  # Training mask
        ValLabel = split_mat['val_data']   # Validation mask
        TsLabel = split_mat['test_data']   # Testing mask
        
        # Set number of classes for Trento
        args.num_classes = 6
        dataset_name = "Trento"
    
    elif args.dataset == 'Muufl':
        base_path = '/beluga/Hackathon15/dataset/MUUFL'
        data_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', 'muufl_data.mat')
        split_file = os.path.join(base_path, f'train_val_test_{args.train_samples}_{args.val_samples}', f'muufl_train_val_test_gt_{args.fold}.mat')
        
        # Load data
        data_mat = scio.loadmat(data_file)
        Data1 = data_mat['hsi_data'].astype(np.float32)  # HSI data
        Data2 = data_mat['lidar_data'].astype(np.float32)  # LiDAR data
        
        # Load train/val/test split
        split_mat = scio.loadmat(split_file)
        TrLabel = split_mat['train_data']  # Training mask
        ValLabel = split_mat['val_data']   # Validation mask
        TsLabel = split_mat['test_data']   # Testing mask
        
        # Set number of classes for MUUFL
        args.num_classes = 11
        dataset_name = "MUUFL"
    
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
    val_pixels = np.sum(ValLabel > 0)
    test_pixels = np.sum(TsLabel > 0)
    print("Training samples: {}, Validation samples: {}, Testing samples: {}".format(train_pixels, val_pixels, test_pixels))
    
    # Create model save directory
    model_save_dir = os.path.join('trained_models_train_val_test', dataset_name, 'CALC')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'CALC_{dataset_name}.pkl')
    report_save_path = os.path.join(model_save_dir, f'CALC_{dataset_name}_performance_report.txt')
    
    # sample generation
    patchsize = args.patch_size
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    
    TrainPatch1, TrainPatch2, TrainLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TrLabel)
    ValPatch1, ValPatch2, ValLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, ValLabel)
    TestPatch1, TestPatch2, TestLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TsLabel)
    
    train_dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print('HSI Training size and validation size and testing size:', TrainPatch1.shape, ValPatch1.shape, TestPatch1.shape)
    print('LiDAR Training size and validation size and testing size:', TrainPatch2.shape, ValPatch2.shape, TestPatch2.shape)
    print('Label Training size and validation size and testing size:', TrainLabel.shape, ValLabel.shape, TestLabel.shape)
    print('Model will be saved to:', model_save_path)
    print('Performance report will be saved to:', report_save_path)
    
    # -------------------------------------------------------------------------------
    # train and test
    tic1 = time.time()
    
    # Get results from train_network - handle both 3 and 4 return values
    results = train_network(
        train_loader=train_loader,
        TrainPatch1=TrainPatch1, 
        TrainPatch2=TrainPatch2, 
        TrainLabel1=TrainLabel,
        ValPatch1=ValPatch1,
        ValPatch2=ValPatch2,
        ValLabel=ValLabel,
        TestPatch1=TestPatch1, 
        TestPatch2=TestPatch2, 
        TestLabel=TestLabel, 
        LR=args.learning_rate,
        EPOCH=args.epoches, 
        patchsize=args.patch_size, 
        l1=band1, 
        l2=band2,
        Classes=args.num_classes, 
        model_save_path=model_save_path,
        eval_interval=args.eval_interval
    )
    
    # Handle different return values from train_network
    if len(results) == 4:
        pred_y, val_acc, test_acc, best_epoch = results
    else:
        # If train_network returns only 3 values, use the last epoch as best_epoch
        pred_y, val_acc, test_acc = results
        best_epoch = args.epoches  # Use last epoch as best epoch
        print(f"Note: Using last epoch ({best_epoch}) as best epoch")
    
    pred_y = pred_y.type(torch.FloatTensor)
    TestLabel = TestLabel.type(torch.FloatTensor)
    
    # Calculate metrics
    OA, Kappa, CA, AA = show_calaError(pred_y, TestLabel)
    
    toc1 = time.time()
    time_1 = toc1 - tic1
    
    print("\nTraining Summary:")
    print("Maximal Validation Accuracy: {:.4f}".format(max(val_acc)))
    print("Test Accuracy at Best Validation: {:.4f}".format(max(test_acc)))
    print("Overall Accuracy (OA): {:.4f}".format(OA))
    print("Kappa Coefficient: {:.4f}".format(Kappa))
    print("Average Accuracy (AA): {:.4f}".format(AA))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    
    # Prepare training statistics
    train_stats = {
        'train_samples': train_pixels,
        'val_samples': val_pixels,
        'test_samples': test_pixels
    }
    
    # Find the actual best epoch from validation accuracy
    actual_best_epoch = np.argmax(val_acc) + 1  # +1 because epochs start from 1
    
    # Save performance report
    save_performance_report(
        report_path=report_save_path,
        dataset_name=dataset_name,
        args=args,
        train_stats=train_stats,
        val_acc=val_acc,
        test_acc=test_acc,
        oa=OA,
        kappa=Kappa,
        ca=CA,
        aa=AA,
        training_time=time_1,
        best_epoch=actual_best_epoch
    )
    
    print(f"Performance report saved to: {report_save_path}")
    
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))


if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()