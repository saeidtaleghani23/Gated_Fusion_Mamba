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

def samplingFixedNum(train_num, val_num, groundTruth, seed):
    """
    Divide dataset into train, validation and test datasets with fixed number of samples per class.
    
    Args:
        train_num (int): Number of training samples per class
        val_num (int): Number of validation samples per class
        groundTruth (numpy.ndarray): Ground truth labels
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_indices, val_indices, test_indices) - indices for training, validation and testing
    """
    labels_loc = {}
    train_ = {}
    val_ = {}
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
        actual_train_num = min(train_num, len(indices))
        remaining_after_train = len(indices) - actual_train_num
        actual_val_num = min(val_num, remaining_after_train)
        
        train_[class_id] = indices[:actual_train_num]
        val_[class_id] = indices[actual_train_num:actual_train_num + actual_val_num]
        test_[class_id] = indices[actual_train_num + actual_val_num:]
        
        print(f"Class {class_id}: {len(indices)} total samples, {len(train_[class_id])} train, {len(val_[class_id])} val, {len(test_[class_id])} test")
    
    train_fix_indices = []
    val_fix_indices = []
    test_fix_indices = []
    
    for class_id in unique_classes:
        train_fix_indices += train_[class_id]
        val_fix_indices += val_[class_id]
        test_fix_indices += test_[class_id]
    
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(val_fix_indices)
    np.random.shuffle(test_fix_indices)
    
    print(f"Total training samples: {len(train_fix_indices)}")
    print(f"Total validation samples: {len(val_fix_indices)}")
    print(f"Total testing samples: {len(test_fix_indices)}")
    
    return train_fix_indices, val_fix_indices, test_fix_indices

def generate_houston2013_data(train_num=20, val_num=10, num_folds=1):
    """
    Generate training, validation and testing data for Houston2013 dataset using GDAL.
    
    Args:
        train_num (int): Number of training samples per class
        val_num (int): Number of validation samples per class
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
    output_dir = os.path.join(base_path, f'train_val_test_{train_num}_{val_num}')
    os.makedirs(output_dir, exist_ok=True)

    
    # Generate multiple splits
    for i in range(num_folds):
        seed = i + 1
        print(f"\nGenerating fold {i+1} with seed {seed}...")
        
        gt_flat = full_gt.reshape(np.prod(full_gt.shape[:2]), ).astype(np.int32)
        
        train_index, val_index, test_index = samplingFixedNum(train_num, val_num, gt_flat, seed)
        
        # Create train, val and test masks
        train_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        train_data[train_index] = gt_flat[train_index]
        
        val_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        val_data[val_index] = gt_flat[val_index]
        
        test_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        test_data[test_index] = gt_flat[test_index]
        
        # Reshape back to original dimensions
        train_data = train_data.reshape(full_gt.shape[0], full_gt.shape[1])
        val_data = val_data.reshape(full_gt.shape[0], full_gt.shape[1])
        test_data = test_data.reshape(full_gt.shape[0], full_gt.shape[1])
        
        # Save the data
        output_path = os.path.join(output_dir, f'train_val_test_gt_{i+1}.mat')
        scio.savemat(output_path,
                    {'train_data': train_data, 'val_data': val_data, 'test_data': test_data,
                     'train_index': train_index, 'val_index': val_index, 'test_index': test_index,
                     'full_gt': full_gt})
        
        print(f"Saved fold {i+1} to {output_path}")
        
        # Analyze the generated split
        analyze_dataset(f"Houston2013 Fold {i+1}", full_gt, train_data, val_data, test_data)
    
    # Save the HSI and LiDAR data for easy access
    data_output_path = os.path.join(output_dir, 'houston2013_data.mat')
    scio.savemat(data_output_path,
                {'hsi_data': hsi_data, 'lidar_data': lidar_data,
                 'full_gt': full_gt, 'train_labels': train_labels,
                 'test_labels': test_labels})
    
    print(f"\nHouston2013 data generation completed!")
    print(f"Data saved to: {output_dir}")


def generate_trento_data(train_num=20, val_num=10, num_folds=1):
    """
    Generate training, validation and testing data for Trento dataset using GDAL.
    
    Args:
        train_num (int): Number of training samples per class
        val_num (int): Number of validation samples per class
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
    output_dir = os.path.join(base_path, f'train_val_test_{train_num}_{val_num}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate multiple splits
    for i in range(num_folds):
        seed = i + 1
        print(f"\nGenerating Trento fold {i+1} with seed {seed}...")
        
        gt_flat = full_gt.reshape(np.prod(full_gt.shape[:2]), ).astype(np.int32)
        
        train_index, val_index, test_index = samplingFixedNum(train_num, val_num, gt_flat, seed)
        
        # Create train, val and test masks
        train_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        train_data[train_index] = gt_flat[train_index]
        
        val_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        val_data[val_index] = gt_flat[val_index]
        
        test_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        test_data[test_index] = gt_flat[test_index]
        
        # Reshape back to original dimensions
        train_data = train_data.reshape(full_gt.shape[0], full_gt.shape[1])
        val_data = val_data.reshape(full_gt.shape[0], full_gt.shape[1])
        test_data = test_data.reshape(full_gt.shape[0], full_gt.shape[1])
        
        # Save the data
        output_path = os.path.join(output_dir, f'trento_train_val_test_gt_{i+1}.mat')
        scio.savemat(output_path,
                    {'train_data': train_data, 'val_data': val_data, 'test_data': test_data,
                     'train_index': train_index, 'val_index': val_index, 'test_index': test_index,
                     'full_gt': full_gt, 'original_train_labels': train_labels,
                     'original_test_labels': test_labels})
        
        print(f"Saved Trento fold {i+1} to {output_path}")
        
        # Analyze the generated split
        analyze_dataset(f"Trento Fold {i+1}", full_gt, train_data, val_data, test_data)
    
    # Save the HSI and LiDAR data for easy access
    data_output_path = os.path.join(output_dir, 'trento_data.mat')
    scio.savemat(data_output_path,
                {'hsi_data': hsi_data, 'lidar_data': lidar_data,
                 'full_gt': full_gt, 'train_labels': train_labels,
                 'test_labels': test_labels})
    
    print(f"\nTrento data generation completed!")
    print(f"Data saved to: {output_dir}")
    

def generate_muufl_data(train_num=20, val_num=10, num_folds=1):
    """
    Generate training, validation and testing data for MUUFL dataset.
    
    Args:
        train_num (int): Number of training samples per class
        val_num (int): Number of validation samples per class
        num_folds (int): Number of cross-validation folds to generate
    """
    base_path = '/beluga/Hackathon15/dataset/MUUFL'
    mat_file_path = os.path.join(base_path, 'muufl_gulfport_campus_1_hsi_220_label.mat')
    
    print("Loading MUUFL dataset...")
    
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Could not find MUUFL dataset file: {mat_file_path}")
    
    # Load the MAT file
    mat_data = scio.loadmat(mat_file_path)
    
    # Extract HSI data - CORRECTED ACCESS
    hsi_data = mat_data['hsi'][0, 0]['Data']  # Direct access to the array
    print(f"HSI data shape: {hsi_data.shape}")
    
    # Extract LiDAR data - FIXED ACCESS
    # Based on the diagnostic output, LiDAR is in a structured array with fields
    lidar_struct = mat_data['hsi'][0, 0]['Lidar'][0, 0]
    print(f"LiDAR structure type: {type(lidar_struct)}")
    
    # The LiDAR data has fields: ('x', 'y', 'z', 'info')
    # We need to access the 'z' field which contains the elevation data
    if hasattr(lidar_struct, 'dtype') and lidar_struct.dtype.names is not None:
        print(f"LiDAR fields: {lidar_struct.dtype.names}")
        
        # Access the 'z' field which contains the elevation data
        if 'z' in lidar_struct.dtype.names:
            lidar_z_data = lidar_struct['z'][0, 0]
            print(f"LiDAR z data shape: {lidar_z_data.shape}")
            
            # The z data should be the elevation data with shape (325, 220)
            if len(lidar_z_data.shape) == 2:
                lidar_data = lidar_z_data[:, :, np.newaxis]  # Add channel dimension
            else:
                lidar_data = lidar_z_data
        else:
            # If 'z' field doesn't exist, try to use the first field
            first_field = lidar_struct.dtype.names[0]
            lidar_data = lidar_struct[first_field][0, 0]
            if len(lidar_data.shape) == 2:
                lidar_data = lidar_data[:, :, np.newaxis]
    else:
        # If it's not a structured array, use it directly
        lidar_data = lidar_struct
        if len(lidar_data.shape) == 2:
            lidar_data = lidar_data[:, :, np.newaxis]
    
    print(f"Final LiDAR data shape: {lidar_data.shape}")
    
    # Extract labels
    labels = mat_data['hsi'][0, 0]['sceneLabels'][0, 0]['labels']
    print(f"Labels shape: {labels.shape}")
    
    # Convert background labels (-1) to 0 for consistency with other datasets
    full_gt = labels.copy()
    full_gt[full_gt == -1] = 0
    
    # Verify data shapes match
    assert hsi_data.shape[0] == full_gt.shape[0] and hsi_data.shape[1] == full_gt.shape[1], \
        f"HSI data shape {hsi_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    assert lidar_data.shape[0] == full_gt.shape[0] and lidar_data.shape[1] == full_gt.shape[1], \
        f"LiDAR data shape {lidar_data.shape[:2]} doesn't match ground truth shape {full_gt.shape}"
    
    print(f"Full ground truth shape: {full_gt.shape}")
    
    # Check class distribution
    unique_classes = np.unique(full_gt)
    unique_classes = unique_classes[unique_classes > 0]  # Exclude background (0)
    print(f"Found {len(unique_classes)} classes in full ground truth: {unique_classes}")
    
    # Print class statistics
    full_flat = full_gt.ravel()
    
    print("\nClass distribution in MUUFL data:")
    print("Class\tTotal")
    print("-" * 20)
    for class_id in unique_classes:
        total_count = np.sum(full_flat == class_id)
        print(f"{class_id}\t{total_count}")
    
    # Create output directory
    output_dir = os.path.join(base_path, f'train_val_test_{train_num}_{val_num}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate multiple splits
    for i in range(num_folds):
        seed = i + 1
        print(f"\nGenerating MUUFL fold {i+1} with seed {seed}...")
        
        gt_flat = full_gt.reshape(np.prod(full_gt.shape[:2]), ).astype(np.int32)
        
        train_index, val_index, test_index = samplingFixedNum(train_num, val_num, gt_flat, seed)
        
        # Create train, val and test masks
        train_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        train_data[train_index] = gt_flat[train_index]
        
        val_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        val_data[val_index] = gt_flat[val_index]
        
        test_data = np.zeros(np.prod(full_gt.shape[:2]), dtype=np.int32)
        test_data[test_index] = gt_flat[test_index]
        
        # Reshape back to original dimensions
        train_data = train_data.reshape(full_gt.shape[0], full_gt.shape[1])
        val_data = val_data.reshape(full_gt.shape[0], full_gt.shape[1])
        test_data = test_data.reshape(full_gt.shape[0], full_gt.shape[1])
        
        # Save the data
        output_path = os.path.join(output_dir, f'muufl_train_val_test_gt_{i+1}.mat')
        scio.savemat(output_path,
                    {'train_data': train_data, 'val_data': val_data, 'test_data': test_data,
                     'train_index': train_index, 'val_index': val_index, 'test_index': test_index,
                     'full_gt': full_gt, 'original_labels': labels})
        
        print(f"Saved MUUFL fold {i+1} to {output_path}")
        
        # Analyze the generated split
        analyze_dataset(f"MUUFL Fold {i+1}", full_gt, train_data, val_data, test_data)
    
    # Save the HSI and LiDAR data for easy access
    data_output_path = os.path.join(output_dir, 'muufl_data.mat')
    scio.savemat(data_output_path,
                {'hsi_data': hsi_data, 'lidar_data': lidar_data,
                 'full_gt': full_gt, 'original_labels': labels})
    
    print(f"\nMUUFL data generation completed!")
    print(f"Data saved to: {output_dir}")

def analyze_dataset(dataset_name, full_gt, train_data, val_data, test_data):
    """
    Analyze dataset statistics for train, validation, and test splits.
    """
    unique_classes = np.unique(full_gt)
    unique_classes = unique_classes[unique_classes > 0]
    
    print(f"\n{dataset_name} Dataset Analysis:")
    print("=" * 60)
    print(f"Image size: {full_gt.shape}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Total pixels: {np.prod(full_gt.shape)}")
    print(f"Labeled pixels: {np.sum(full_gt > 0)}")
    print(f"Labeled percentage: {np.sum(full_gt > 0) / np.prod(full_gt.shape) * 100:.2f}%")
    
    print("\nClass Distribution:")
    print("Class\tTrain\tVal\tTest\tTotal")
    print("-" * 40)
    
    train_flat = train_data.ravel()
    val_flat = val_data.ravel()
    test_flat = test_data.ravel()
    full_flat = full_gt.ravel()
    
    for class_id in unique_classes:
        train_count = np.sum(train_flat == class_id)
        val_count = np.sum(val_flat == class_id)
        test_count = np.sum(test_flat == class_id)
        total_count = np.sum(full_flat == class_id)
        print(f"{class_id}\t{train_count}\t{val_count}\t{test_count}\t{total_count}")

# Main execution
if __name__ == '__main__':
    train_num = 100  # Number of training samples per class
    val_num = 50     # Number of validation samples per class
    
    print("Starting data generation with GDAL...")
    print("=" * 60)
    
    # Generate Houston2013 data
    print("\n1. PROCESSING HOUSTON2013 DATASET")
    print("-" * 40)
    #generate_houston2013_data(train_num=train_num, val_num=val_num, num_folds=1)
    
    print("\n" + "=" * 60)
    
    # Generate Trento data
    print("\n2. PROCESSING TRENTO DATASET")
    print("-" * 40)
    generate_trento_data(train_num=train_num, val_num=val_num, num_folds=1)
    
    print("\n" + "=" * 60)
    
    # Generate MUUFL data
    print("\n3. PROCESSING MUUFL DATASET")
    print("-" * 40)
    generate_muufl_data(train_num=train_num, val_num=val_num, num_folds=1)
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETED SUCCESSFULLY!")