import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dataf
import argparse
import os
import sys
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import json
import math
import yaml  # Add this import

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
utils_path = '/beluga/Hackathon15/baseline/Fusion_Mamba'
sys.path.insert(0, utils_path)

try:
    from utils import setup_seed, train_2_patch_label2, show_calaError
    print(f"✅ Successfully imported utils from: {utils_path}")
except ImportError as e:
    print(f"❌ Failed to import utils from {utils_path}: {e}")
    sys.exit(1)

model_path = '/beluga/Hackathon15/saeid_model'
sys.path.insert(0, model_path)

try:
    from MultiLayerFusionClassifier import MultiLayerFusionClassifier
    print(f"✅ Successfully imported model from: {model_path}")
except ImportError as e:
    print(f"❌ Failed to import model from {model_path}: {e}")
    sys.exit(1)

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================
def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Successfully loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Failed to load config from {config_path}: {e}")
        sys.exit(1)
def update_args_from_config(args, config):
    """Update command line arguments with values from config file"""
    # Training parameters
    if 'training' in config:
        training_config = config['training']
        args.gpu_id = training_config.get('gpu_id', args.gpu_id)
        args.seed = training_config.get('seed', args.seed)
        args.epoches = training_config.get('epochs', args.epoches)
        args.learning_rate = training_config.get('learning_rate', args.learning_rate)
        args.dataset = training_config.get('dataset', args.dataset)
        args.num_classes = training_config.get('num_classes', args.num_classes)
        args.batch_size = training_config.get('batch_size', args.batch_size)
        args.patch_size = training_config.get('patch_size', args.patch_size)
        args.training_mode = training_config.get('training_mode', args.training_mode)
        args.train_samples = training_config.get('train_samples', args.train_samples)
        args.val_samples = training_config.get('val_samples', args.val_samples)
        args.fold = training_config.get('fold', args.fold)
        args.eval_interval = training_config.get('eval_interval', args.eval_interval)
    
    # Model architecture
    if 'model' in config:
        model_config = config['model']
        args.num_layers = model_config.get('num_layers', args.num_layers)
        args.use_downsampling = model_config.get('use_downsampling', args.use_downsampling)
        args.downsample_method = model_config.get('downsample_method', args.downsample_method)
        
        # Handle layer_clusters - convert list to string if needed
        layer_clusters = model_config.get('layer_clusters', args.layer_clusters)
        if isinstance(layer_clusters, list):
            args.layer_clusters = ','.join(map(str, layer_clusters))
        else:
            args.layer_clusters = layer_clusters
            
        # Handle feature_dims - convert list to string if needed
        feature_dims = model_config.get('feature_dims', args.feature_dims)
        if isinstance(feature_dims, list):
            args.feature_dims = ','.join(map(str, feature_dims))
        else:
            args.feature_dims = feature_dims
        
        # Mamba parameters
        if 'mamba' in model_config:
            mamba_config = model_config['mamba']
            args.d_state = mamba_config.get('d_state', args.d_state)
            args.d_conv = mamba_config.get('d_conv', args.d_conv)
            args.expand = mamba_config.get('expand', args.expand)
    
    # Optimizer configuration
    if 'optimizer' in config:
        optimizer_config = config['optimizer']
        args.optimizer_type = optimizer_config.get('type', args.optimizer_type)
        args.weight_decay = optimizer_config.get('weight_decay', args.weight_decay)
        args.momentum = optimizer_config.get('momentum', args.momentum)
        
    # Scheduler configuration
    if 'scheduler' in config:
        scheduler_config = config['scheduler']
        args.scheduler_type = scheduler_config.get('type', args.scheduler_type)
        args.step_size = scheduler_config.get('step_size', args.step_size)
        args.gamma = scheduler_config.get('gamma', args.gamma)
        args.warmup_epochs = scheduler_config.get('warmup_epochs', args.warmup_epochs)
    
    # Loss function configuration
    if 'loss' in config:
        loss_config = config['loss']
        args.loss_function = loss_config.get('type', args.loss_function)
        args.label_smoothing = loss_config.get('label_smoothing', args.label_smoothing)
    
    return args

# =============================================================================
# OPTIMIZER, SCHEDULER AND LOSS FUNCTION FACTORIES
# =============================================================================
def create_optimizer(model, args):
    """Create optimizer based on configuration"""
    if args.optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), 
                         lr=args.learning_rate, 
                         weight_decay=args.weight_decay)
    elif args.optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), 
                          lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    elif args.optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), 
                        lr=args.learning_rate, 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_type}")

def create_scheduler(optimizer, args):
    """Create learning rate scheduler based on configuration"""
    if args.scheduler_type.lower() == 'cosine':
        return get_cosine_with_warmup_scheduler(optimizer, 
                                              args.warmup_epochs, 
                                              args.epoches)
    elif args.scheduler_type.lower() == 'step':
        return StepLR(optimizer, 
                     step_size=args.step_size, 
                     gamma=args.gamma)
    elif args.scheduler_type.lower() == 'none':
        # Return a dummy scheduler that does nothing
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler_type}")

def create_loss_function(args, device=None):
    """Create loss function based on configuration"""
    if args.loss_function.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss_function.lower() == 'focal':
        # You would need to implement or import FocalLoss
        # For now, using CrossEntropy as fallback
        print("⚠️  Focal loss not implemented, using CrossEntropy instead")
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_function}")

# Add this to your training setup
def get_cosine_with_warmup_scheduler(optimizer, num_warmup_epochs, num_training_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =============================================================================
# MODEL CONFIGURATION - PROPER CONTROL
# =============================================================================
def create_model_config(args, hsi_channels, lidar_channels, num_classes):
    """Create model configuration dictionary"""
    
    # Parse layer clusters from string to list
    if isinstance(args.layer_clusters, str):
        layer_clusters = [int(x.strip()) for x in args.layer_clusters.split(',')]
    else:
        layer_clusters = args.layer_clusters
    
    # Parse feature dimensions from string to list
    if isinstance(args.feature_dims, str):
        feature_dims = [int(x.strip()) for x in args.feature_dims.split(',')]
    else:
        feature_dims = args.feature_dims
    
    # Validate that feature_dims length matches num_layers
    if len(feature_dims) != args.num_layers:
        raise ValueError(f"feature_dims length ({len(feature_dims)}) must match num_layers ({args.num_layers})")
    
    # Model configuration
    model_config = {
        'hsi_channels': hsi_channels,
        'lidar_channels': lidar_channels,
        'feature_dims': feature_dims,  # ← Now a list per layer
        'num_layers': args.num_layers,
        'num_classes': num_classes,
        'use_downsampling': args.use_downsampling,
        'downsample_method': args.downsample_method,
        'layer_clusters': layer_clusters,
        'patch_size': args.patch_size,
        'mamba_config': {
            'd_state': args.d_state,
            'd_conv': args.d_conv,
            'expand': args.expand
        }
    }
    
    return model_config

def validate_model_config(config):
    """Validate model configuration"""
    errors = []
    
    if len(config['layer_clusters']) != config['num_layers']:
        errors.append(f"layer_clusters length ({len(config['layer_clusters'])}) must match num_layers ({config['num_layers']})")
    
    if len(config['feature_dims']) != config['num_layers']:
        errors.append(f"feature_dims length ({len(config['feature_dims'])}) must match num_layers ({config['num_layers']})")
    
    if any(dim <= 0 for dim in config['feature_dims']):
        errors.append(f"All feature_dims must be positive, got {config['feature_dims']}")
    
    if config['num_layers'] <= 0:
        errors.append(f"num_layers must be positive, got {config['num_layers']}")
    
    if any(cluster <= 0 for cluster in config['layer_clusters']):
        errors.append(f"All layer_clusters must be positive, got {config['layer_clusters']}")
    
    if config['downsample_method'] not in ['conv', 'pool']:
        errors.append(f"downsample_method must be 'conv' or 'pool', got {config['downsample_method']}")
    
    return errors

# =============================================================================
# ARGUMENT PARSER WITH COMPLETE MODEL CONFIG
# =============================================================================
parser = argparse.ArgumentParser("MULTILAYER_FUSION_CLASSIFIER")

# Configuration file
parser.add_argument('--config', type=str, default='/beluga/Hackathon15/saeid_model/config_huston2013.yaml', help='Path to config YAML file')

# Training parameters
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='learning rate')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Muufl', help='dataset to use')
parser.add_argument('--num_classes', type=int, default=15, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patch_size', type=int, default=32, help='patch size')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times'], default='one_time', help='training times')
parser.add_argument('--train_samples', type=int, default=100, help='training samples per class')
parser.add_argument('--val_samples', type=int, default=50, help='validation samples per class')
parser.add_argument('--fold', type=int, default=1, help='cross-validation fold')
parser.add_argument('--eval_interval', type=int, default=10, help='validation evaluation interval')

# Model architecture parameters
parser.add_argument('--num_layers', type=int, default=3, help='number of fusion layers')
parser.add_argument('--use_downsampling', type=bool, default=True, help='use downsampling in fusion layers')
parser.add_argument('--downsample_method', choices=['conv', 'pool'], default='conv', help='downsampling method')
parser.add_argument('--layer_clusters', type=str, default='50,25,15', help='clusters for each layer as comma-separated list')
parser.add_argument('--feature_dims', type=str, default='64,128,256', help='feature dimensions for each layer as comma-separated list')

# Mamba-specific parameters
parser.add_argument('--d_state', type=int, default=16, help='Mamba d_state parameter')
parser.add_argument('--d_conv', type=int, default=4, help='Mamba d_conv parameter')
parser.add_argument('--expand', type=int, default=2, help='Mamba expand parameter')

# Optimizer and loss function parameters (new)
parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='optimizer type')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step', 'none'], help='scheduler type')
parser.add_argument('--step_size', type=int, default=30, help='step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma for StepLR scheduler')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs for cosine scheduler')
parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'focal'], help='loss function type')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for cross entropy loss')

args = parser.parse_args()

# Load configuration from YAML file and update args
config = load_config(args.config)
args = update_args_from_config(args, config)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# =============================================================================
# TRAINING FUNCTIONS (remain the same as before)
# =============================================================================

def save_performance_report(report_path, dataset_name, args, train_stats, train_acc, val_acc, test_acc, 
                          oa, kappa, ca, aa, training_time, best_epoch, model_config, val_loss_history=None):
    """
    Save comprehensive performance report as a text file
    """
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTILAYER FUSION CLASSIFIER PERFORMANCE REPORT\n")
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
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of Layers: {model_config['num_layers']}\n")
        f.write(f"Use Downsampling: {model_config['use_downsampling']}\n")
        f.write(f"Downsample Method: {model_config['downsample_method']}\n")
        f.write(f"Layer Clusters: {model_config['layer_clusters']}\n")
        f.write(f"Feature Dimensions: {model_config['feature_dims']}\n")
        f.write(f"HSI Input Channels: {model_config['hsi_channels']}\n")
        f.write(f"LiDAR Input Channels: {model_config['lidar_channels']}\n")
        f.write(f"Mamba d_state: {model_config['mamba_config']['d_state']}\n")
        f.write(f"Mamba d_conv: {model_config['mamba_config']['d_conv']}\n")
        f.write(f"Mamba expand: {model_config['mamba_config']['expand']}\n\n")
        
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
        f.write(f"Best Training Accuracy: {max(train_acc):.2f}%\n")
        f.write(f"Best Validation Accuracy: {max(val_acc):.2f}%\n")
        f.write(f"Test Accuracy at Best Validation: {max(test_acc):.2f}%\n")
        f.write(f"Overall Accuracy (OA): {oa:.4f}\n")
        f.write(f"Kappa Coefficient: {kappa:.4f}\n")
        f.write(f"Average Accuracy (AA): {aa:.4f}\n\n")
        
        # NEW: Validation Loss History
        if val_loss_history is not None:
            f.write("VALIDATION LOSS HISTORY\n")
            f.write("-" * 40 + "\n")
            if len(val_loss_history) <= 10:
                for i, loss in enumerate(val_loss_history):
                    f.write(f"Epoch {i+1}: {loss:.4f}\n")
            else:
                for i in range(5):
                    f.write(f"Epoch {i+1}: {val_loss_history[i]:.4f}\n")
                f.write("...\n")
                for i in range(len(val_loss_history)-5, len(val_loss_history)):
                    f.write(f"Epoch {i+1}: {val_loss_history[i]:.4f}\n")
            f.write("\n")
        
        # Training Accuracy History
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
        
        # Validation Accuracy History
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
        
        # Test Accuracy History
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
    Train the MultiLayerFusionClassifier model with validation and testing
    """
    best_val_acc = 0.0
    best_epoch = 0
    val_acc_history = []
    test_acc_history = []
    train_acc_history = []
    val_loss_history = []  # NEW: Track validation loss
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Add progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (patch1, patch2, labels) in enumerate(pbar):
            patch1, patch2, labels = patch1.to(device), patch2.to(device), labels.to(device)
            
            # Ensure labels are long type for classification
            classification_targets = labels.long()
            
            optimizer.zero_grad()
            
            # Forward pass - MultiLayerFusionClassifier returns (logits, layer_outputs)
            logits, _ = model(patch1, patch2)
            
            # Compute loss
            loss = criterion(logits, classification_targets)
            
            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Calculate gradient norms (more efficient)
            total_norm = 0.0
            num_params = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    num_params += 1
            total_norm = total_norm ** 0.5


            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            current_acc = 100 * train_correct / train_total
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{current_acc:.2f}%"
            })
            
            # if batch_idx % 50 == 0:
            #     batch_accuracy = 100 * (predicted == labels).float().mean().item()
            #     print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
            #           f'Loss: {loss.item():.4f}, '
            #           f'Batch Acc: {batch_accuracy:.2f}%')
        
        # Calculate epoch training accuracy
        train_accuracy = 100 * train_correct / train_total
        train_acc_history.append(train_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        avg_loss = running_loss / len(train_loader)
        
        # NEW: Calculate validation loss and accuracy EVERY EPOCH
        val_acc, val_loss = evaluate_model_with_loss(model, val_patch1, val_patch2, val_label, criterion, device)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        
        # NEW: Print both training and validation metrics every epoch
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training   - Loss: {avg_loss:.4f}, Acc: {train_accuracy:.2f}%')
        print(f'  Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # Test accuracy only during eval_interval or last epoch (to save time)
        if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
            test_acc = evaluate_model(model, test_patch1, test_patch2, test_label, device)
            test_acc_history.append(test_acc)
            print(f'  Test       - Acc: {test_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f'  ✅ New best model saved! (Val Acc: {best_val_acc:.2f}%)')
            # save model with .pkl format as well
            pkl_save_path = model_save_path.replace('.pth', '.pkl')
            torch.save(model.state_dict(), pkl_save_path)
        
        print('')  # Empty line for readability
    
    return val_acc_history, test_acc_history, train_acc_history, best_epoch, val_loss_history

def evaluate_model_with_loss(model, patch1, patch2, labels, criterion, device):
    """Evaluate model performance and return both accuracy and loss"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(patch1), batch_size):
            batch_patch1 = patch1[i:i+batch_size].to(device)
            batch_patch2 = patch2[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            # Forward pass
            logits, _ = model(batch_patch1, batch_patch2)
            loss = criterion(logits, batch_labels.long())
            
            _, predicted = torch.max(logits.data, 1)
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            running_loss += loss.item() * batch_labels.size(0)
            
            # Clean up to save memory
            del batch_patch1, batch_patch2, batch_labels, logits, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / total
    return accuracy, avg_loss

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
            
            # Forward pass
            logits, _ = model(batch_patch1, batch_patch2)
            _, predicted = torch.max(logits.data, 1)
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            # Clean up to save memory
            del batch_patch1, batch_patch2, batch_labels, logits, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    accuracy = 100 * correct / total
    return accuracy

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def test_in_batches(model, test_data1, test_data2, batch_size=32):
    """Test in smaller batches to avoid OOM"""
    all_logits = []
    
    # Get device from model parameters
    device = next(model.parameters()).device
    print(f"🔍 Using device: {device}")
    
    # MOVE ENTIRE DATASET TO GPU ONCE (more efficient)
    test_data1 = test_data1.to(device)
    test_data2 = test_data2.to(device)
    #print(f"🔍 Moved entire dataset to: {device}")
    
    total_batches = (len(test_data1) + batch_size - 1) // batch_size
    #print(f"🔍 Testing {len(test_data1)} samples in {total_batches} batches of {batch_size}")
    
    
    for i in tqdm(range(0, len(test_data1), batch_size), 
                  total=total_batches, 
                  desc="Testing batches", 
                  unit="batch"):
        end_idx = min(i + batch_size, len(test_data1))
        batch1 = test_data1[i:end_idx]
        batch2 = test_data2[i:end_idx]
        
        # No need for .to(device) here - batches are already on GPU
        with torch.no_grad():
            batch_logits, _ = model(batch1, batch2)  # Direct pass, no device transfer
            all_logits.append(batch_logits.cpu())  # Move to CPU immediately
            
        # Clear GPU memory between batches
        torch.cuda.empty_cache()
        
    return torch.cat(all_logits)

# =============================================================================
# MAIN TRAINING FUNCTION (with updated optimizer/loss creation)
# =============================================================================
def train_1times():
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
    
    # Create model save directory - UPDATED PATH
    model_save_dir = os.path.join('/beluga/Hackathon15/trained_models_train_val_test', dataset_name, 'MultiLayerFusion')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'MultiLayerFusion_{dataset_name}_fold{args.fold}.pth')
    report_save_path = os.path.join(model_save_dir, f'MultiLayerFusion_{dataset_name}_fold{args.fold}_performance_report.txt')
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get actual channel dimensions from data
    hsi_channels = TrainPatch1.shape[1]  # Number of HSI bands
    lidar_channels = TrainPatch2.shape[1]  # Number of LiDAR bands
    
    # CREATE AND VALIDATE MODEL CONFIG
    model_config = create_model_config(args, hsi_channels, lidar_channels, args.num_classes)
    
    # Validate configuration
    config_errors = validate_model_config(model_config)
    if config_errors:
        print("❌ Model configuration errors:")
        for error in config_errors:
            print(f"   - {error}")
        sys.exit(1)
    print("✅ Model Configuration:")
    print(f"   - HSI channels: {model_config['hsi_channels']}")
    print(f"   - LiDAR channels: {model_config['lidar_channels']}")
    print(f"   - Feature dimensions per layer: {model_config['feature_dims']}")
    print(f"   - Number of layers: {model_config['num_layers']}")
    print(f"   - Layer clusters: {model_config['layer_clusters']}")
    print(f"   - Downsampling: {model_config['use_downsampling']} ({model_config['downsample_method']})")
    print(f"   - Mamba config: d_state={model_config['mamba_config']['d_state']}, "
        f"d_conv={model_config['mamba_config']['d_conv']}, expand={model_config['mamba_config']['expand']}")
    
    # Initialize model WITH CONFIG
    model = MultiLayerFusionClassifier(**model_config).to(device)
    
    # Save model configuration
    config_save_path = os.path.join(model_save_dir, f'MultiLayerFusion_{dataset_name}_fold{args.fold}_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✅ Model configuration saved to: {config_save_path}")
    
    # USE CONFIG-BASED OPTIMIZER, LOSS AND SCHEDULER
    print("✅ Creating components from configuration:")
    print(f"   - Optimizer: {args.optimizer_type}")
    print(f"   - Scheduler: {args.scheduler_type}")
    print(f"   - Loss function: {args.loss_function}")
    
    criterion = create_loss_function(args, device)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Print model layer information
    layer_info = model.get_layer_info()
    print("\nModel Layer Configuration:")
    for info in layer_info:
        print(f"Layer {info['layer']}: {info['clusters']} clusters, "
              f"Downsampling: {info['downsampling']}, "
              f"Input: HSI{info['input_channels_hsi']}/LiDAR{info['input_channels_lidar']}")
    
    # train and test
    tic1 = time.time()
    
    val_acc_history, test_acc_history, train_acc_history, best_epoch, val_loss_history = train_model(
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
        print('test part')
        test_logits = test_in_batches(model, TestPatch1, TestPatch2, batch_size=32)
        _, test_pred = torch.max(test_logits.data, 1)
        test_pred = test_pred.cpu()
    
    # Calculate metrics
    OA, Kappa, CA, AA = show_calaError(test_pred.unsqueeze(1).float(), TestLabel.unsqueeze(1).float())
    
    toc1 = time.time()
    time_1 = toc1 - tic1
    
    print("\nTraining Summary:")
    print("Maximal Training Accuracy: {:.2f}%".format(max(train_acc_history)))
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
    
    # Add val_loss_history to the save_performance_report call
    save_performance_report(
        report_path=report_save_path,
        dataset_name=dataset_name,
        args=args,
        train_stats=train_stats,
        train_acc=train_acc_history,
        val_acc=val_acc_history,
        test_acc=test_acc_history,
        oa=OA,
        kappa=Kappa,
        ca=CA,
        aa=AA,
        training_time=time_1,
        best_epoch=actual_best_epoch,
        model_config=model_config,
        val_loss_history=val_loss_history  # NEW: Add validation loss history
    )
    
    print(f"Performance report saved to: {report_save_path}")
    
    toc = time.time()
    time_all = toc - tic1
    print('All process complete in {:.0f}m {:.0f}s'.format(time_all / 60, time_all % 60))

if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()