import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
import sys
import time
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add model import paths
model_path = '/beluga/Hackathon15/saeid_model'
sys.path.insert(0, model_path)

try:
    from MultiLayerFusionClassifier import MultiLayerFusionClassifier
    print(f"✅ Successfully imported model from: {model_path}")
except ImportError as e:
    print(f"❌ Failed to import model from {model_path}: {e}")
    sys.exit(1)



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





def classification_map(map, groundTruth, dpi, savePath):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi=dpi)
    plt.close(fig)
    return 0

def create_model_config_from_json(config_path, hsi_channels, lidar_channels, num_classes):
    """Load model configuration from JSON file saved during training"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update with actual channel dimensions
    config['hsi_channels'] = hsi_channels
    config['lidar_channels'] = lidar_channels
    config['num_classes'] = num_classes
    
    return config

def generate_classification_map(dataset_name, model_path, data_path):
    """
    Generate classification map for a specific dataset using the trained MultiLayerFusionClassifier
    """
    print(f"\n=== Generating Classification Map for {dataset_name} using MultiLayerFusionClassifier ===")
    
    # Get the model directory (where outputs will be saved)
    model_dir = os.path.dirname(model_path)
    
    # Load dataset specific parameters
    if dataset_name.lower() == 'houston2013':
        data_file = os.path.join(data_path, f'train_val_test_100_50', 'houston2013_data.mat')
        num_classes = 15
        l1 = 144
        l2 = 1
        
        class_names = [
            'Background', 'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
            'Soil', 'Water', 'Residential', 'Commercial', 
            'Road', 'Highway', 'Railway', 'Parking Lot 1',
            'Parking Lot 2', 'Tennis Court', 'Running Track'
        ]
        colors = [
            [0.0, 0.0, 0.0],
            [51/255, 72/255, 155/255],
            [57/255, 78/255, 161/255],
            [65/255, 96/255, 176/255],
            [70/255, 150/255, 213/255],
            [78/255, 195/255, 236/255],
            [107/255, 198/255, 189/255],
            [131/255, 198/255, 129/255],
            [173/255, 210/255, 51/255],
            [231/255, 230/255, 26/255],
            [234/255, 191/255, 35/255],
            [232/255, 137/255, 35/255],
            [236/255, 74/255, 38/255],
            [238/255, 37/255, 37/255],
            [192/255, 32/255, 42/255],
            [124/255, 24/255, 24/255]
        ]
        
    elif dataset_name.lower() == 'trento':
        data_file = os.path.join(data_path, f'train_val_test_100_50', 'trento_data.mat')
        num_classes = 6
        l1 = 63
        l2 = 1
        
        class_names = [
            'Background', 'Apples', 'Buildings', 'Ground', 
            'Woods', 'Vineyard', 'Roads'
        ]
        colors = [
            [0.0, 0.0, 0.0],
            [61/255, 86/255, 168/255],
            [80/255, 200/255, 235/255],
            [154/255, 204/255, 105/255],
            [255/255, 209/255, 122/255],
            [238/255, 52/255, 39/255],
            [124/255, 21/255, 22/255]
        ]
        
    elif dataset_name.lower() == 'muufl':
        data_file = os.path.join(data_path, f'train_val_test_100_50', 'muufl_data.mat')
        num_classes = 11
        l1 = 64
        l2 = 2
        
        class_names = [
            'Background',
            'Trees',
            'Mostly grass', 
            'Mixed ground surface',
            'Dirt and sand',
            'Road',
            'Water',
            'Building shadow',
            'Building',
            'Side walk', 
            'Yellow curb',
            'Cloth Panels'
        ]
        colors = [
            [0.0, 0.0, 0.0],
            [0/255, 129/255, 5/255],
            [0/255, 254/255, 1/255],
            [0/255, 255/255, 255/255],
            [253/255, 207/255, 2/255],
            [254/255, 0/255, 52/255],
            [0/255, 0/255, 202/255],
            [102/255, 0/255, 203/255],
            [251/255, 124/255, 145/255],
            [196/255, 101/255, 0/255],
            [254/255, 251/255, 0/255],
            [204/255, 24/255, 97/255]
        ]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load data
    data_mat = scio.loadmat(data_file)
    Data1 = data_mat['hsi_data'].astype(np.float32)
    Data2 = data_mat['lidar_data'].astype(np.float32)
    
    # Ensure LiDAR data has proper shape
    if len(Data2.shape) == 2:
        Data2 = Data2.reshape([Data2.shape[0], Data2.shape[1], -1])
    
    [m1, n1, l1_actual] = np.shape(Data1)
    [m2, n2, l2_actual] = np.shape(Data2)
    
    print(f"Data1 shape: {Data1.shape}")
    print(f"Data2 shape: {Data2.shape}")
    print(f"Expected bands - HSI: {l1}, LiDAR: {l2}")
    print(f"Actual bands - HSI: {l1_actual}, LiDAR: {l2_actual}")
    
    # Parameters - MUST MATCH TRAINING PARAMETERS
    patchsize = 32  # Same as training
    pad_width = int(patchsize / 2)
    
    # Data preprocessing with progress bar
    print("Preprocessing HSI data...")
    for i in tqdm(range(l1_actual), desc="HSI Normalization"):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    
    print("Preprocessing LiDAR data...")
    for i in tqdm(range(l2_actual), desc="LiDAR Normalization"):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2

    # Create padded arrays
    print("Creating padded arrays...")
    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1_actual), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2_actual), dtype='float32')

    for i in tqdm(range(l1_actual), desc="HSI Padding"):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
        
    for i in tqdm(range(l2_actual), desc="LiDAR Padding"):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2

    # Load model - CORRECTED to match your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get actual channel dimensions from data
    hsi_channels = l1_actual
    lidar_channels = l2_actual
    
    print(f"HSI input channels: {hsi_channels}, LiDAR input channels: {lidar_channels}")
    
    # Load model configuration from training
    config_path = model_path.replace('.pth', '_config.json').replace('.pkl', '_config.json')
    if os.path.exists(config_path):
        print(f"Loading model configuration from: {config_path}")
        model_config = create_model_config_from_json(config_path, hsi_channels, lidar_channels, num_classes)
    else:
        # Fallback configuration (use your training parameters)
        print("⚠️  Config file not found, using fallback parameters")
        model_config = {
            'hsi_channels': hsi_channels,
            'lidar_channels': lidar_channels,
            'feature_dims': [64, 128, 256],  # From your training logs
            'num_layers': 3,
            'num_classes': num_classes,
            'use_downsampling': True,
            'downsample_method': 'conv',
            'layer_clusters': [50, 25, 15],  # From your training logs
            'patch_size': 32,
            'mamba_config': {
                'd_state': 16,
                'd_conv': 4,
                'expand': 2
            }
        }
    
    # Initialize model with correct architecture
    model = MultiLayerFusionClassifier(**model_config)
    
    # Load model weights - handle both .pth and .pkl
    if model_path.endswith('.pth') or model_path.endswith('.pkl'):
        if torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    
    model.eval()
    print(f"Model loaded from: {model_path}")
    print(f"Model architecture: {model_config['num_layers']} layers")
    print(f"Feature dimensions: {model_config['feature_dims']}")
    print(f"Layer clusters: {model_config['layer_clusters']}")

    # Generate predictions for entire image
    batch_size = 32  # Use same batch size as training
    pred_all = np.zeros((m1 * n1, 1), dtype='float32')
    
    total_pixels = m1 * n1
    number_of_batches = (total_pixels + batch_size - 1) // batch_size
    
    print(f"Generating predictions for {total_pixels} pixels in {number_of_batches} batches...")
    
    # Main processing loop with progress bar
    with torch.no_grad():
        with tqdm(total=number_of_batches, desc="Processing batches") as pbar:
            for batch_idx in range(number_of_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_pixels)
                current_batch_size = end_idx - start_idx
                
                # Initialize batch arrays with correct shape
                # MultiLayerFusionClassifier expects [batch_size, channels, height, width]
                D1 = np.empty((current_batch_size, hsi_channels, patchsize, patchsize), dtype='float32')
                D2 = np.empty((current_batch_size, lidar_channels, patchsize, patchsize), dtype='float32')
                
                # Process each pixel in the current batch
                for count, pixel_idx in enumerate(range(start_idx, end_idx)):
                    row = pixel_idx // n1
                    col = pixel_idx % n1
                    row2 = row + pad_width
                    col2 = col + pad_width
                    
                    # Extract patches
                    patch1 = x1_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                    patch2 = x2_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                    
                    # Reshape for model input: [bands, height, width]
                    patch1 = np.transpose(patch1, (2, 0, 1))  # [hsi_channels, patchsize, patchsize]
                    patch2 = np.transpose(patch2, (2, 0, 1))  # [lidar_channels, patchsize, patchsize]
                    
                    D1[count, :, :, :] = patch1
                    D2[count, :, :, :] = patch2

                # Convert to tensors and move to device
                temp1 = torch.from_numpy(D1).float()
                temp2 = torch.from_numpy(D2).float()
                
                if torch.cuda.is_available():
                    temp1 = temp1.cuda()
                    temp2 = temp2.cuda()
                
                # Forward pass - MultiLayerFusionClassifier returns (logits, layer_outputs)
                logits, _ = model(temp1, temp2)
                predictions = torch.max(logits, 1)[1].squeeze()
                pred_all[start_idx:end_idx, 0] = predictions.cpu().numpy()
                
                # Clean up memory
                del temp1, temp2, logits, predictions, D1, D2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.update(1)

    # Reshape and save results
    pred_all = np.reshape(pred_all, (m1, n1))
    
    # Convert to 1-indexed for background (0 = background)
    pred_all = pred_all + 1
    
    # Save MATLAB file
    mat_save_path = os.path.join(model_dir, f'{dataset_name}_classification_map.mat')
    scio.savemat(mat_save_path, {'pred_all': pred_all})
    
    # Create and save classification map
    print("Creating classification map...")
    best_G = pred_all
    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    
    with tqdm(total=best_G.shape[0], desc="Color mapping") as pbar:
        for i in range(best_G.shape[0]):
            for j in range(best_G.shape[1]):
                class_id = int(best_G[i][j])
                if 0 <= class_id < len(colors):
                    hsi_pic[i, j, :] = colors[class_id]
                else:
                    hsi_pic[i, j, :] = [0, 0, 0]  # Black for unknown
            pbar.update(1)

    # Save classification map image
    img_save_path = os.path.join(model_dir, f'{dataset_name}_classification_map.png')
    classification_map(hsi_pic, best_G, 24, img_save_path)
    
    print(f"Results saved to: {model_dir}")
    print(f"  - MATLAB file: {mat_save_path}")
    print(f"  - Image file: {img_save_path}")
    
    # Print class-color mapping
    print(f"\nClass-Color Mapping for {dataset_name}:")
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if i < len(colors):
            rgb_color = tuple(int(c * 255) for c in color)
            print(f"  Class {i}: {class_name} -> RGB{rgb_color}")

if __name__ == '__main__':
    # Define dataset paths and model locations - UPDATED to use FusionMamba directory
    datasets = [
    #    {
    #         'name': 'Houston2013',
    #         'data_path': '/beluga/Hackathon15/dataset/Houston2013',
    #         'model_path': '/beluga/Hackathon15/trained_models_train_val_test/Houston2013/MultiLayerFusion/MultiLayerFusion_Houston2013_fold1.pth'
    #     },
        # {
        #     'name': 'Trento', 
        #     'data_path': '/beluga/Hackathon15/dataset/Trento',
        #     'model_path': '/beluga/Hackathon15/trained_models_train_val_test/Trento/MultiLayerFusion/MultiLayerFusion_Trento_fold1.pth'
        # },
        {
            'name': 'MUUFL',
            'data_path': '/beluga/Hackathon15/dataset/MUUFL',
            'model_path': '/beluga/Hackathon15/trained_models_train_val_test/MUUFL/MultiLayerFusion/MultiLayerFusion_MUUFL_fold1.pth'
        }
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset['model_path']):
            generate_classification_map(
                dataset['name'],
                dataset['model_path'],
                dataset['data_path']
            )
        else:
            print(f"Model not found: {dataset['model_path']}")
            print("Please train the model first using train_fusion_mamba_saeid.py")