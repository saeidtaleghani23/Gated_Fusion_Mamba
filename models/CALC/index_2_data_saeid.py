# This code only generate training and test samples.
# To get validation samples as well, use index_2_data_saeid_2.py
import numpy as np
import scipy.io as scio
from osgeo import gdal
import os

def read_raster_data(file_path):
    """
    Read raster data using GDAL (supports TIFF, IMG, and other GDAL-supported formats).
    
    Args:
        file_path (str): Path to the raster file
        
    Returns:
        tuple: (data, geotransform, projection) 
               - data: numpy array of the raster data
               - geotransform: GDAL geotransform parameters
               - projection: Coordinate system projection
    """
    try:
        # Open the dataset
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"GDAL could not open file: {file_path}")
        
        # Get basic information
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        
        print(f"Reading {file_path}:")
        print(f"  Size: {width} x {height} x {bands}")
        print(f"  Data type: {gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)}")
        
        # Read all bands
        if bands > 1:
            data = np.zeros((height, width, bands), dtype=np.float32)
            for i in range(bands):
                band = dataset.GetRasterBand(i + 1)
                data[:, :, i] = band.ReadAsArray(0, 0, width, height)
        else:
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray(0, 0, width, height)
            # Add channel dimension for consistency
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]
        
        # Close the dataset
        dataset = None
        
        return data, geotransform, projection
        
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        raise

def samplingFixedNum(sample_num, groundTruth, seed):
    """
    Divide dataset into train and test datasets with fixed number of samples per class.
    
    Args:
        sample_num (int): Number of training samples per class
        groundTruth (numpy.ndarray): Ground truth labels
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, test_indices) - indices for training and testing
    """
    labels_loc = {}
    train_ = {}
    test_ = {}
    np.random.seed(seed)
    
    # Get unique classes (excluding background/0)
    unique_classes = np.unique(groundTruth)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background
    
    print(f"Found {len(unique_classes)} classes: {unique_classes}")
    
    for class_id in unique_classes:
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == class_id]
        np.random.shuffle(indices)
        labels_loc[class_id] = indices
        
        # Ensure we don't take more samples than available
        actual_sample_num = min(sample_num, len(indices))
        train_[class_id] = indices[:actual_sample_num]
        test_[class_id] = indices[actual_sample_num:]
        
        print(f"Class {class_id}: {len(indices)} total samples, {len(train_[class_id])} train, {len(test_[class_id])} test")
    
    train_fix_indices = []
    test_fix_indices = []
    
    for class_id in unique_classes:
        train_fix_indices += train_[class_id]
        test_fix_indices += test_[class_id]
    
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    
    print(f"Total training samples: {len(train_fix_indices)}")
    print(f"Total testing samples: {len(test_fix_indices)}")
    
    return train_fix_indices, test_fix_indices

def generate_houston2013_data(train_num=20, num_folds=1):
    """
    Generate training and testing data for Houston2013 dataset using GDAL.
    
    Args:
        train_num (int): Number of training samples per class
        num_folds (int): Number of cross-validation folds to generate
    """
    base_path = '/beluga/Hackathon15/dataset/Houston2013'
    
    print("Loading Houston2013 dataset with GDAL...")
    
    # Load HSI data
    hsi_path = os.path.join(base_path, '2013_IEEE_GRSS_DF_Contest_CASI.tif')
    hsi_data, hsi_gt, hsi_proj = read_raster_data(hsi_path)
    print(f"HSI data shape: {hsi_data.shape}")
    
    # Load LiDAR data
    lidar_path = os.path.join(base_path, '2013_IEEE_GRSS_DF_Contest_LiDAR.tif')
    lidar_data, lidar_gt, lidar_proj = read_raster_data(lidar_path)
    print(f"LiDAR data shape: {lidar_data.shape}")
    
    # Load training labels
    train_label_path = os.path.join(base_path, '2013_IEEE_GRSS_DF_Contest_Samples_TR.tif')
    train_labels, train_gt, train_proj = read_raster_data(train_label_path)
    train_labels = train_labels.squeeze()  # Remove channel dimension if exists
    
    # Load test labels
    test_label_path = os.path.join(base_path, '2013_IEEE_GRSS_DF_Contest_Samples_VA.tif')
    test_labels, test_gt, test_proj = read_raster_data(test_label_path)
    test_labels = test_labels.squeeze()  # Remove channel dimension if exists
    
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Combine train and test labels for full ground truth
    # In Houston2013, train and test labels are provided separately and don't overlap
    full_gt = np.zeros_like(train_labels, dtype=np.int32)
    full_gt = np.where(train_labels > 0, train_labels, full_gt)
    full_gt = np.where(test_labels > 0, test_labels, full_gt)
    
    # Verify data shapes match
    assert hsi_data.shape[0] == full_gt.shape[0] and hsi_data.shape[1] == full_gt.shape[1], \
        f"HSI data shape {hsi_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    assert lidar_data.shape[0] == full_gt.shape[0] and lidar_data.shape[1] == full_gt.shape[1], \
        f"LiDAR data shape {lidar_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    print(f"Full ground truth shape: {full_gt.shape}")
    unique_classes = np.unique(full_gt)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background
    print(f"Unique classes in ground truth: {unique_classes}")
    
    # Create output directory
    output_dir = os.path.join(base_path, f'train_test_{train_num}')
    os.makedirs(output_dir, exist_ok=True)

    
    # Generate multiple splits
    for i in range(num_folds):
        seed = i + 1
        print(f"\nGenerating fold {i+1} with seed {seed}...")
        
        gt_flat = full_gt.reshape(np.prod(full_gt.shape[:2]), ).astype(np.int32)
        
        train_index, test_index = samplingFixedNum(train_num, gt_flat, seed)
        
        # Create train and test masks
        train_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        train_data[train_index] = gt_flat[train_index]
        
        test_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        test_data[test_index] = gt_flat[test_index]
        
        # Reshape back to original dimensions
        train_data = train_data.reshape(full_gt.shape[0], full_gt.shape[1])
        test_data = test_data.reshape(full_gt.shape[0], full_gt.shape[1])
        
        # Save the data
        output_path = os.path.join(output_dir, f'train_test_gt_{i+1}.mat')
        scio.savemat(output_path,
                    {'train_data': train_data, 'test_data': test_data,
                     'train_index': train_index, 'test_index': test_index,
                     'full_gt': full_gt})
        
        print(f"Saved fold {i+1} to {output_path}")
    
    # Save the HSI and LiDAR data for easy access
    data_output_path = os.path.join(output_dir, 'houston2013_data.mat')
    scio.savemat(data_output_path,
                {'hsi_data': hsi_data, 'lidar_data': lidar_data,
                 'full_gt': full_gt, 'train_labels': train_labels,
                 'test_labels': test_labels})
    
    print(f"\nHouston2013 data generation completed!")
    print(f"Data saved to: {output_dir}")


def generate_trento_data(train_num=20, num_folds=1):
    """
    Generate training and testing data for Trento dataset using GDAL.
    
    Args:
        train_num (int): Number of training samples per class
        num_folds (int): Number of cross-validation folds to generate
    """
    base_path = '/beluga/Hackathon15/dataset/Trento'
    
    print("Loading Trento dataset...")
    
    # Load HSI data
    mat_hsi_path = os.path.join(base_path, 'HSI.mat')
    if os.path.exists(mat_hsi_path):
        print("Loading HSI data from MAT file...")
        hsi_data_mat = scio.loadmat(mat_hsi_path)
        hsi_key = [key for key in hsi_data_mat.keys() if not key.startswith('__')][0]
        hsi_data = hsi_data_mat[hsi_key].astype(np.float32)
        print(f"HSI data shape from MAT: {hsi_data.shape}")
    else:
        raise FileNotFoundError("Could not find HSI.mat file")
    
    # Load LiDAR data
    mat_lidar_path = os.path.join(base_path, 'LiDAR.mat')
    if os.path.exists(mat_lidar_path):
        print("Loading LiDAR data from MAT file...")
        lidar_data_mat = scio.loadmat(mat_lidar_path)
        lidar_key = [key for key in lidar_data_mat.keys() if not key.startswith('__')][0]
        lidar_data = lidar_data_mat[lidar_key].astype(np.float32)
        # Ensure LiDAR has 3D shape
        if len(lidar_data.shape) == 2:
            lidar_data = lidar_data[:, :, np.newaxis]
        print(f"LiDAR data shape from MAT: {lidar_data.shape}")
    else:
        raise FileNotFoundError("Could not find LiDAR.mat file")
    
    # Load training labels
    train_label_path = os.path.join(base_path, 'TRLabel.mat')
    if os.path.exists(train_label_path):
        print("Loading training labels from TRLabel.mat...")
        train_labels_mat = scio.loadmat(train_label_path)
        train_labels_key = [key for key in train_labels_mat.keys() if not key.startswith('__')][0]
        train_labels = train_labels_mat[train_labels_key].astype(np.int32)
        print(f"Training labels shape: {train_labels.shape}")
    else:
        raise FileNotFoundError("Could not find TRLabel.mat file")
    
    # Load test labels
    test_label_path = os.path.join(base_path, 'TSLabel.mat')
    if os.path.exists(test_label_path):
        print("Loading test labels from TSLabel.mat...")
        test_labels_mat = scio.loadmat(test_label_path)
        test_labels_key = [key for key in test_labels_mat.keys() if not key.startswith('__')][0]
        test_labels = test_labels_mat[test_labels_key].astype(np.int32)
        print(f"Test labels shape: {test_labels.shape}")
    else:
        raise FileNotFoundError("Could not find TSLabel.mat file")
    
    # Combine train and test labels for full ground truth (similar to Houston2013)
    # In Trento, train and test labels are provided separately and don't overlap
    full_gt = np.zeros_like(train_labels, dtype=np.int32)
    full_gt = np.where(train_labels > 0, train_labels, full_gt)
    full_gt = np.where(test_labels > 0, test_labels, full_gt)
    
    print(f"HSI data shape: {hsi_data.shape}")
    print(f"LiDAR data shape: {lidar_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Full ground truth shape: {full_gt.shape}")
    
    # Verify data shapes match
    assert hsi_data.shape[0] == full_gt.shape[0] and hsi_data.shape[1] == full_gt.shape[1], \
        f"HSI data shape {hsi_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    assert lidar_data.shape[0] == full_gt.shape[0] and lidar_data.shape[1] == full_gt.shape[1], \
        f"LiDAR data shape {lidar_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    # Check class distribution
    unique_classes = np.unique(full_gt)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background
    print(f"Found {len(unique_classes)} classes in full ground truth: {unique_classes}")
    
    # Print class statistics
    train_flat = train_labels.ravel()
    test_flat = test_labels.ravel()
    full_flat = full_gt.ravel()
    
    print("\nClass distribution in original data:")
    print("Class\tTrain\tTest\tTotal")
    print("-" * 30)
    for class_id in unique_classes:
        train_count = np.sum(train_flat == class_id)
        test_count = np.sum(test_flat == class_id)
        total_count = np.sum(full_flat == class_id)
        print(f"{class_id}\t{train_count}\t{test_count}\t{total_count}")
    
    # Create output directory
    output_dir = os.path.join(base_path, f'train_test_{train_num}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate multiple splits
    for i in range(num_folds):
        seed = i + 1
        print(f"\nGenerating Trento fold {i+1} with seed {seed}...")
        
        gt_flat = full_gt.reshape(np.prod(full_gt.shape[:2]), ).astype(np.int32)
        
        train_index, test_index = samplingFixedNum(train_num, gt_flat, seed)
        
        # Create train and test masks
        train_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        train_data[train_index] = gt_flat[train_index]
        
        test_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        test_data[test_index] = gt_flat[test_index]
        
        # Reshape back to original dimensions
        train_data = train_data.reshape(full_gt.shape[0], full_gt.shape[1])
        test_data = test_data.reshape(full_gt.shape[0], full_gt.shape[1])
        
        # Save the data
        output_path = os.path.join(output_dir, f'trento_train_test_gt_{i+1}.mat')
        scio.savemat(output_path,
                    {'train_data': train_data, 'test_data': test_data,
                     'train_index': train_index, 'test_index': test_index,
                     'full_gt': full_gt, 'original_train_labels': train_labels,
                     'original_test_labels': test_labels})
        
        print(f"Saved Trento fold {i+1} to {output_path}")
        
        # Analyze the generated split
        analyze_dataset(f"Trento Fold {i+1}", full_gt, train_data, test_data)
    
    # Save the HSI and LiDAR data for easy access
    data_output_path = os.path.join(output_dir, 'trento_data.mat')
    scio.savemat(data_output_path,
                {'hsi_data': hsi_data, 'lidar_data': lidar_data,
                 'full_gt': full_gt, 'train_labels': train_labels,
                 'test_labels': test_labels})
    
    print(f"\nTrento data generation completed!")
    print(f"Data saved to: {output_dir}")

def analyze_dataset(dataset_name, full_gt, train_data, test_data):
    """
    Analyze dataset statistics.
    """
    unique_classes = np.unique(full_gt)
    unique_classes = unique_classes[unique_classes > 0]
    
    print(f"\n{dataset_name} Dataset Analysis:")
    print("=" * 50)
    print(f"Image size: {full_gt.shape}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Total pixels: {np.prod(full_gt.shape)}")
    print(f"Labeled pixels: {np.sum(full_gt > 0)}")
    print(f"Labeled percentage: {np.sum(full_gt > 0) / np.prod(full_gt.shape) * 100:.2f}%")
    
    print("\nClass Distribution:")
    print("Class\tTrain\tTest\tTotal")
    print("-" * 30)
    
    train_flat = train_data.ravel()
    test_flat = test_data.ravel()
    full_flat = full_gt.ravel()
    
    for class_id in unique_classes:
        train_count = np.sum(train_flat == class_id)
        test_count = np.sum(test_flat == class_id)
        total_count = np.sum(full_flat == class_id)
        print(f"{class_id}\t{train_count}\t{test_count}\t{total_count}")

# Main execution
if __name__ == '__main__':
    train_num = 100  # Number of training samples per class
    
    print("Starting data generation with GDAL...")
    print("=" * 60)
    
    # Generate Houston2013 data
    print("\n1. PROCESSING HOUSTON2013 DATASET")
    print("-" * 40)
    generate_houston2013_data(train_num=train_num, num_folds=1)
    
    print("\n" + "=" * 60)
    
    # Generate Trento data
    print("\n2. PROCESSING TRENTO DATASET")
    print("-" * 40)
    generate_trento_data(train_num=train_num, num_folds=1)
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETED SUCCESSFULLY!")