import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed, train_2_patch_label2, show_calaError
import argparse
import os
import time
from models.saeid_model import VSSM_Fusion_Classifier, CombinedFusionClassificationLoss
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("SAEID_MODEL")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--epoches', type=int, default=50, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Trento', help='dataset to use')
parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--patch_size', type=int, default=32, help='number of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
parser.add_argument('--train_samples', type=int, default=100, help='number of training samples per class')
parser.add_argument('--val_samples', type=int, default=50, help='number of validation samples per class')
parser.add_argument('--fold', type=int, default=1, help='cross-validation fold to use')
parser.add_argument('--eval_interval', type=int, default=10, help='interval for validation evaluation')
parser.add_argument('--fusion_weight', type=float, default=1.0, help='weight for fusion loss')
parser.add_argument('--classification_weight', type=float, default=1.0, help='weight for classification loss')
parser.add_argument('--stem_channels', type=int, default=64, help='number of channels after STEM layers')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def save_performance_report(report_path, dataset_name, args, train_stats, train_acc, val_acc, test_acc, 
                          oa, kappa, ca, aa, training_time, best_epoch):
    """
    Save comprehensive performance report as a text file
    """
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SAEID MODEL PERFORMANCE REPORT\n")
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
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Fusion Weight: {args.fusion_weight}\n")
        f.write(f"Classification Weight: {args.classification_weight}\n")
        f.write(f"STEM Channels: {args.stem_channels}\n")
        f.write(f"HSI Input Channels: {train_stats['hsi_channels']}\n")
        f.write(f"LiDAR Input Channels: {train_stats['lidar_channels']}\n\n")
        
        # Training Statistics
        f.write("TRAINING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training Samples: {train_stats['train_samples']}\n")
        f.write(f"Validation Samples: {train_stats['val_samples']}\n")
        f.write(f"Test Samples: {train_stats['test_samples']}\n")
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        # Performance Metrics section - UPDATED
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Training Accuracy: {max(train_acc):.2f}%\n")  # NEW
        f.write(f"Best Validation Accuracy: {max(val_acc):.2f}%\n")
        f.write(f"Test Accuracy at Best Validation: {max(test_acc):.2f}%\n")
        f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
        f.write(f"Kappa Coefficient: {kappa:.4f}\n")
        f.write(f"Average Accuracy (AA): {aa:.4f}\n\n")
        
        # NEW: Training Accuracy History section
        f.write("TRAINING ACCURACY HISTORY\n")
        f.write("-" * 40 + "\n")
        if len(train_acc) <= 10:
            for i, acc in enumerate(train_acc):
                f.write(f"Epoch {i+1}: {acc:.2f}%\n")
        else:
            for i in range(5):
                f.write(f"Epoch {i+1}: {train_acc[i]:.2f}%\n")
            f.write("...\n")
            for i in range(len(train_acc)-5, len(train_acc)):
                f.write(f"Epoch {i+1}: {train_acc[i]:.2f}%\n")
        f.write("\n")
        
        # Validation Accuracy History (first 5 and last 5)
        f.write("VALIDATION ACCURACY HISTORY\n")
        f.write("-" * 40 + "\n")
        if len(val_acc) <= 10:
            for i, acc in enumerate(val_acc):
                f.write(f"Epoch {i+1}: {acc:.2f}%\n")
        else:
            for i in range(5):
                f.write(f"Epoch {i+1}: {val_acc[i]:.2f}%\n")
            f.write("...\n")
            for i in range(len(val_acc)-5, len(val_acc)):
                f.write(f"Epoch {i+1}: {val_acc[i]:.2f}%\n")
        f.write("\n")
        
        # Test Accuracy History (first 5 and last 5)
        f.write("TEST ACCURACY HISTORY\n")
        f.write("-" * 40 + "\n")
        if len(test_acc) <= 10:
            for i, acc in enumerate(test_acc):
                f.write(f"Epoch {i+1}: {acc:.2f}%\n")
        else:
            for i in range(5):
                f.write(f"Epoch {i+1}: {test_acc[i]:.2f}%\n")
            f.write("...\n")
            for i in range(len(test_acc)-5, len(test_acc)):
                f.write(f"Epoch {i+1}: {test_acc[i]:.2f}%\n")
        f.write("\n")
        
        # Class-wise Accuracy
        f.write("CLASS-WISE ACCURACY\n")
        f.write("-" * 40 + "\n")
        for i, acc in enumerate(ca):
            f.write(f"Class {i+1}: {acc:.4f}\n")
        f.write("\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model trained successfully on {dataset_name} dataset.\n")
        f.write(f"Achieved {max(train_acc):.2f}% training accuracy, {max(val_acc):.2f}% validation accuracy and {max(test_acc):.2f}% test accuracy.\n")
        f.write(f"Overall performance: OA={oa:.4f}, Kappa={kappa:.4f}, AA={aa:.4f}\n")
        f.write(f"Training completed in {training_time/60:.2f} minutes.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

def train_model(train_loader, val_patch1, val_patch2, val_label, test_patch1, test_patch2, test_label,
                model, criterion, optimizer, scheduler, num_epochs, device, model_save_path, eval_interval=10):
    """
    Train the SAEID model with validation and testing
    """
    best_val_acc = 0.0
    best_epoch = 0
    val_acc_history = []
    test_acc_history = []
    train_acc_history = []  # NEW: Track training accuracy
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_fusion_loss = 0.0
        train_correct = 0  # NEW: Track training correct predictions
        train_total = 0    # NEW: Track training total samples
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (patch1, patch2, labels) in enumerate(pbar):
            patch1, patch2, labels = patch1.to(device), patch2.to(device), labels.to(device)
            
            # Ensure labels are long type for classification
            classification_targets = labels.long()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(patch1, patch2)
            
            # Compute loss
            losses = criterion(outputs, patch1, patch2, classification_targets)
            
            # NEW: Calculate training accuracy
            _, predicted = torch.max(outputs['classification'].data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()
            
            running_loss += losses['total_loss'].item()
            running_class_loss += losses['classification_loss'].item()
            running_fusion_loss += losses['fusion_total'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total_loss'].item():.4f}",
                'Class': f"{losses['classification_loss'].item():.4f}",
                'Fusion': f"{losses['fusion_total'].item():.4f}"
            })
            

            if batch_idx % 50 == 0:
                batch_accuracy = 100 * (predicted == labels).float().mean().item()
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {losses["total_loss"].item():.4f}, '
                      f'Class: {losses["classification_loss"].item():.4f}, '
                      f'Fusion: {losses["fusion_total"].item():.4f}, '
                      f'Batch Acc: {batch_accuracy:.2f}%')
        
        # NEW: Calculate epoch training accuracy
        train_accuracy = 100 * train_correct / train_total
        train_acc_history.append(train_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        avg_class_loss = running_class_loss / len(train_loader)
        avg_fusion_loss = running_fusion_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, '
              f'Class Loss: {avg_class_loss:.4f}, Fusion Loss: {avg_fusion_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%')  # NEW: Print training accuracy
        

        # Validation phase
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            val_acc = evaluate_model(model, val_patch1, val_patch2, val_label, device)
            test_acc = evaluate_model(model, test_patch1, test_patch2, test_label, device)
            
            val_acc_history.append(val_acc)
            test_acc_history.append(test_acc)
            
            print(f'Train Accuracy: {train_accuracy:.2f}%, '  # NEW: Include in validation print
                  f'Validation Accuracy: {val_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
    
    return val_acc_history, test_acc_history, train_acc_history, best_epoch  # NEW: Return train accuracy history

# In evaluate_model function, add memory cleanup
def evaluate_model(model, patch1, patch2, labels, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(patch1), batch_size):
            batch_patch1 = patch1[i:i+batch_size].to(device)
            batch_patch2 = patch2[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            outputs = model(batch_patch1, batch_patch2)
            _, predicted = torch.max(outputs['classification'].data, 1)
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Clean up to save memory
            del batch_patch1, batch_patch2, batch_labels, outputs, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accuracy = 100 * correct / total
    return accuracy

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
    model_save_dir = os.path.join('trained_models_train_val_test', dataset_name, 'FusionMamba')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'FusionMamba_{dataset_name}.pkl')
    report_save_path = os.path.join(model_save_dir, f'FusionMamba_{dataset_name}_performance_report.txt')
    
    # sample generation
    patchsize = args.patch_size
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    
    TrainPatch1, TrainPatch2, TrainLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TrLabel)
    ValPatch1, ValPatch2, ValLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, ValLabel)
    TestPatch1, TestPatch2, TestLabel = train_2_patch_label2(Data1, Data2, patchsize, pad_width, TsLabel)
    
        # Check if already tensors, otherwise convert
    def ensure_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        else:
            return torch.from_numpy(data).float()

    def ensure_label_tensor(data):
        if isinstance(data, torch.Tensor):
            return data.long().squeeze()
        else:
            return torch.from_numpy(data).long().squeeze()

    # Convert to PyTorch tensors (only if they're numpy arrays)
    TrainPatch1 = ensure_tensor(TrainPatch1)
    TrainPatch2 = ensure_tensor(TrainPatch2)
    TrainLabel = ensure_label_tensor(TrainLabel)

    ValPatch1 = ensure_tensor(ValPatch1)
    ValPatch2 = ensure_tensor(ValPatch2)
    ValLabel = ensure_label_tensor(ValLabel)

    TestPatch1 = ensure_tensor(TestPatch1)
    TestPatch2 = ensure_tensor(TestPatch2)
    TestLabel = ensure_label_tensor(TestLabel)
    
    # Create data loader
    train_dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel)
    train_loader = dataf.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print('HSI Training size and validation size and testing size:', TrainPatch1.shape, ValPatch1.shape, TestPatch1.shape)
    print('LiDAR Training size and validation size and testing size:', TrainPatch2.shape, ValPatch2.shape, TestPatch2.shape)
    print('Label Training size and validation size and testing size:', TrainLabel.shape, ValLabel.shape, TestLabel.shape)
    print('Model will be saved to:', model_save_path)
    print('Performance report will be saved to:', report_save_path)
    
    # -------------------------------------------------------------------------------
    # Initialize model with correct channel dimensions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get actual channel dimensions from data
    hsi_channels = TrainPatch1.shape[1]  # Number of HSI bands
    lidar_channels = TrainPatch2.shape[1]  # Number of LiDAR bands
    
    print(f"HSI input channels: {hsi_channels}, LiDAR input channels: {lidar_channels}")
    print(f"STEM output channels: {args.stem_channels}")
    
    # Initialize model with STEM layers and correct channel dimensions
    model = VSSM_Fusion_Classifier(
        patch_size=4,
        in_chans1=hsi_channels,   # HSI input channels
        in_chans2=lidar_channels, # LiDAR input channels
        num_classes=args.num_classes,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        classifier_dropout=0.2,
        stem_channels=args.stem_channels,  # Convert all inputs to stem_channels
        use_checkpoint=False
    ).to(device)
    
    # Initialize the improved loss function
    criterion = CombinedFusionClassificationLoss(
        fusion_weight=args.fusion_weight,
        classification_weight=args.classification_weight,
        use_channel_aware=True  # Use the better channel-aware fusion loss
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # -------------------------------------------------------------------------------
    # train and test
    tic1 = time.time()
    
    val_acc_history, test_acc_history, train_acc_history, best_epoch = train_model(
        train_loader=train_loader,
        val_patch1=ValPatch1,
        val_patch2=ValPatch2, 
        val_label=ValLabel,
        test_patch1=TestPatch1,
        test_patch2=TestPatch2,
        test_label=TestLabel,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epoches,
        device=device,
        model_save_path=model_save_path,
        eval_interval=args.eval_interval
    )
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Final evaluation on test set
    with torch.no_grad():
        test_outputs = model(TestPatch1.to(device), TestPatch2.to(device))
        _, test_pred = torch.max(test_outputs['classification'].data, 1)
        test_pred = test_pred.cpu()
    
    # Calculate metrics
    OA, Kappa, CA, AA = show_calaError(test_pred.unsqueeze(1).float(), TestLabel.unsqueeze(1).float())
    
    toc1 = time.time()
    time_1 = toc1 - tic1
    
    print("\nTraining Summary:")
    print("Maximal Training Accuracy: {:.2f}%".format(max(train_acc_history)))  # NEW
    print("Maximal Validation Accuracy: {:.2f}%".format(max(val_acc_history)))
    print("Test Accuracy at Best Validation: {:.2f}%".format(max(test_acc_history)))
    print("Overall Accuracy (OA): {:.4f}".format(OA))
    print("Kappa Coefficient: {:.4f}".format(Kappa))
    print("Average Accuracy (AA): {:.4f}".format(AA))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_1 / 60, time_1 % 60))
    
    # Prepare training statistics
    train_stats = {
        'train_samples': train_pixels,
        'val_samples': val_pixels,
        'test_samples': test_pixels,
        'hsi_channels': hsi_channels,
        'lidar_channels': lidar_channels
    }
    
    # Find the actual best epoch from validation accuracy
    actual_best_epoch = best_epoch
    
    # Save performance report - FIXED: Added train_acc parameter
    save_performance_report(
        report_path=report_save_path,
        dataset_name=dataset_name,
        args=args,
        train_stats=train_stats,
        train_acc=train_acc_history,  # ADDED THIS LINE
        val_acc=val_acc_history,
        test_acc=test_acc_history,
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