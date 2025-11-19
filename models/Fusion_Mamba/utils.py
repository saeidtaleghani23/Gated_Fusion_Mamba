# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 15:10
# @Author  : DingKexin
# @FileName: utils.py
# @Software: PyCharm
import torch
import numpy as np

def CalAccuracy(predict, label):
    n = label.shape[0]
    OA = torch.sum(predict == label) * 1.0 / n
    
    # Get the actual number of classes from the data
    num_classes = len(torch.unique(label))
    
    correct_sum = torch.zeros(num_classes)
    reali = torch.zeros(num_classes)
    predicti = torch.zeros(num_classes)
    CA = torch.zeros(num_classes)
    
    for i in range(0, num_classes):
        correct_sum[i] = torch.sum(label[torch.where(predict == i)] == i)
        reali[i] = torch.sum(label == i)
        predicti[i] = torch.sum(predict == i)
        CA[i] = correct_sum[i] / reali[i]

    Kappa = (n * torch.sum(correct_sum) - torch.sum(reali * predicti)) * 1.0 / (n * n - torch.sum(reali * predicti))
    AA = torch.mean(CA)
    return OA, Kappa, CA, AA


def show_calaError(val_predict_labels, val_true_labels):
    """
    Display classification accuracy metrics in a formatted output.
    
    This function serves as a wrapper for CalAccuracy that formats and prints
    the results for easy interpretation during model evaluation.
    
    Args:
        val_predict_labels (torch.Tensor): Predicted labels from validation/test set
        val_true_labels (torch.Tensor): True labels from validation/test set
        
    Returns:
        tuple: Same as CalAccuracy - (OA, Kappa, CA, AA)
        
    Example:
        >>> OA, Kappa, CA, AA = show_calaError(predictions, true_labels)
        OA: 0.950000, Kappa: 0.930000, AA: 0.940000
        CA: tensor([0.92, 0.96, 0.94, ...])
    """
    val_predict_labels = torch.squeeze(val_predict_labels)
    val_true_labels = torch.squeeze(val_true_labels)
    OA, Kappa, CA, AA = CalAccuracy(val_predict_labels, val_true_labels)
    # ic(OA, Kappa, CA, AA)
    print("OA: %f, Kappa: %f,  AA: %f" % (OA, Kappa, AA))
    print("CA: ", )
    print(CA)
    return OA, Kappa, CA, AA


def setup_seed(seed):
    """
    Set random seeds for reproducibility across all random number generators.
    
    This function ensures that experiments are reproducible by fixing the random
    seed for PyTorch (CPU and GPU), NumPy, and CUDA backends. Essential for
    obtaining consistent results in remote sensing classification experiments.
    
    Args:
        seed (int): Random seed value to use for all generators
        
    Note:
        Setting cudnn.deterministic = True may reduce performance but ensures
        reproducible results on GPUs.
    """
    torch.manual_seed(seed)  # Sets seed for PyTorch CPU
    torch.cuda.manual_seed_all(seed) # Sets seed for PyTorch GPU(s)
    np.random.seed(seed)  # Sets seed for NumPy
    torch.backends.cudnn.deterministic = True # Makes GPU operations deterministic (slower but reproducible)


def train_2_patch_label2(Data1, Data2, patchsize, pad_width, Label):
    """
    Preprocess hyperspectral and LiDAR data into patches for 2D-CNN training.
    
    This function performs the complete data preprocessing pipeline for multi-modal
    remote sensing data, including normalization, padding, patch extraction, and
    format conversion for PyTorch models.
    
    Args:
        Data1 (numpy.ndarray): Hyperspectral image data of shape (height, width, bands)
        Data2 (numpy.ndarray): LiDAR data of shape (height, width, bands) 
        patchsize (int): Size of the square patches to extract (e.g., 16 for 16x16 patches)
        pad_width (int): Padding width, typically patchsize//2 for centered patches
        Label (numpy.ndarray): Training/Testing label map of shape (height, width)
        
    Returns:
        tuple: A tuple containing:
            - TrainPatch1 (torch.Tensor): Hyperspectral patches of shape (n_samples, bands, patchsize, patchsize)
            - TrainPatch2 (torch.Tensor): LiDAR patches of shape (n_samples, bands, patchsize, patchsize) 
            - TrainLabel (torch.Tensor): Class labels of shape (n_samples,) with 0-based indexing
            
    Processing Steps:
        1. Normalize each band of both datasets to [0, 1] range
        2. Apply symmetric padding to handle border pixels
        3. Extract patches centered on labeled pixels
        4. Convert from HWC to CHW format for PyTorch
        5. Convert labels to 0-based indexing and PyTorch tensors
        
    Note:
        - Labels are converted from 1-based to 0-based indexing (subtract 1)
        - Output format is compatible with PyTorch DataLoader for batch processing
        - Supports datasets like Houston (349x1905), Muufl, and Trento
    """
    [m1, n1, l1] = np.shape(Data1)
    [m2, n2, l2] = np.shape(Data2)
    # Normalize hyperspectral data (Data1)
    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1  # 349*1905*144
    
    # Normalize LiDAR data (Data2)  
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2  # 349*1905*21

    # Create padded arrays for patch extraction
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')  # 365*1921*144
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')  # 365*1921*21 -> Saeid: for LiDAR data, 365x1921x1 is correct
    
    # Apply symmetric padding to handle border pixels
    for i in range(l1):
        temp = x1[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x1_pad[:, :, i] = temp2  # 365*1921*144
    for i in range(l2):
        temp = x2[:, :, i]  # 349*1905
        temp2 = np.pad(temp, pad_width, 'symmetric')  # 365*1921
        x2_pad[:, :, i] = temp2  # 365*1921*21

    # Extract patches from labeled locations
    [ind1, ind2] = np.where(Label > 0)  # Find coordinates of labeled pixels
    TrainNum = len(ind1)  # Total number of training samples
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')  # 300*144*16*16
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')  # 300*21*16*16
    TrainLabel = np.empty(TrainNum)  # 300
    
    # Convert coordinates to padded image space
    ind3 = ind1 + pad_width  # 300
    ind4 = ind2 + pad_width  # 300
    
    # Extract patches for each labeled pixel
    for i in range(len(ind1)):
        # Extract hyperspectral patch
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*144
        patch1 = np.transpose(patch1, (2, 0, 1))  # 144*16*16 (HWC to CHW)
        TrainPatch1[i, :, :, :] = patch1  # 300*144*16*16
        
        # Extract LiDAR patch  
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]  # 16*16*21
        patch2 = np.transpose(patch2, (2, 0, 1))  # 21*16*16 (HWC to CHW)
        TrainPatch2[i, :, :, :] = patch2  # 300*21*16*16
        
        # Get corresponding label
        patchlabel = Label[ind1[i], ind2[i]]  # 1
        TrainLabel[i] = patchlabel  # 300

    # Convert to PyTorch tensors with proper data types
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1  # Convert to 0-based indexing
    TrainLabel = TrainLabel.long()  # Convert to long tensor for classification
    
    return TrainPatch1, TrainPatch2, TrainLabel