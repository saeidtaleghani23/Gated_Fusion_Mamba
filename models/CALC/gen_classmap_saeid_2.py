import scipy.io as scio
import numpy as np
import torch
import torch.utils.data as dataf
from utils import setup_seed
import sys
import time
import os
from CALC_Saeid import Network
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm

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
    plt.close(fig)  # Close the figure to free memory
    return 0

def generate_classification_map(dataset_name, model_path, data_path):
    """
    Generate classification map for a specific dataset
    
    Args:
        dataset_name: Name of the dataset ('Houston2013', 'Trento', or 'MUUFL')
        model_path: Path to the trained model
        data_path: Base path to dataset
    """
    print(f"\n=== Generating Classification Map for {dataset_name} ===")
    
    # Get the model directory (where outputs will be saved)
    model_dir = os.path.dirname(model_path)
    
    # Load dataset specific parameters
    if dataset_name.lower() == 'houston2013':
        data_file = os.path.join(data_path, 'train_test_100', 'houston2013_data.mat')
        num_classes = 15
        l1 = 144  # HSI bands for Houston
        l2 = 1    # LiDAR bands for Houston
        
        class_names = [
            'Background', 'Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
            'Soil', 'Water', 'Residential', 'Commercial', 
            'Road', 'Highway', 'Railway', 'Parking Lot 1',
            'Parking Lot 2', 'Tennis Court', 'Running Track'
        ]
        colors = [
            [0.0, 0.0, 0.0],                    # Black - Background (class 0)
            [51/255, 72/255, 155/255],          # Class 1 - Healthy grass
            [57/255, 78/255, 161/255],          # Class 2 - Stressed grass
            [65/255, 96/255, 176/255],          # Class 3 - Synthetic grass
            [70/255, 150/255, 213/255],         # Class 4 - Trees
            [78/255, 195/255, 236/255],         # Class 5 - Soil
            [107/255, 198/255, 189/255],        # Class 6 - Water
            [131/255, 198/255, 129/255],        # Class 7 - Residential
            [173/255, 210/255, 51/255],         # Class 8 - Commercial
            [231/255, 230/255, 26/255],         # Class 9 - Road
            [234/255, 191/255, 35/255],         # Class 10 - Highway
            [232/255, 137/255, 35/255],         # Class 11 - Railway
            [236/255, 74/255, 38/255],          # Class 12 - Parking Lot 1
            [238/255, 37/255, 37/255],          # Class 13 - Parking Lot 2
            [192/255, 32/255, 42/255],          # Class 14 - Tennis Court
            [124/255, 24/255, 24/255]           # Class 15 - Running Track
        ]
        
    elif dataset_name.lower() == 'trento':
        data_file = os.path.join(data_path, 'train_test_100', 'trento_data.mat')
        num_classes = 6
        l1 = 63   # HSI bands for Trento
        l2 = 1    # LiDAR bands for Trento
        
        class_names = [
            'Background', 'Apples', 'Buildings', 'Ground', 
            'Woods', 'Vineyard', 'Roads'
        ]
        colors = [
            [0.0, 0.0, 0.0],      # Black - Background
            [61/255, 86/255, 168/255],      # Apples
            [80/255, 200/255, 235/255],      # Buildings  
            [154/255, 204/255, 105/255],      # Ground
            [255/255, 209/255, 122/255],      # Woods
            [238/255, 52/255, 39/255],      # Vineyard
            [124/255, 21/255, 22/255]       # Roads
        ]
        
    elif dataset_name.lower() == 'muufl':
        # Updated path for MUUFL dataset with train_val_test structure
        data_file = os.path.join(data_path, f'train_val_test_100_50', 'muufl_data.mat')
        num_classes = 11
        l1 = 64   # HSI bands for MUUFL
        l2 = 2    # LiDAR bands for MUUFL (first and last return)
        
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
        # Convert RGB colors (0-255) to normalized (0-1) for matplotlib
        colors = [
            [0.0, 0.0, 0.0],                    # Black - Background (class 0)
            [0/255, 129/255, 5/255],            # Class 1 - Trees
            [0/255, 254/255, 1/255],            # Class 2 - Mostly grass
            [0/255, 255/255, 255/255],          # Class 3 - Mixed ground surface
            [253/255, 207/255, 2/255],          # Class 4 - Dirt and sand
            [254/255, 0/255, 52/255],           # Class 5 - Road
            [0/255, 0/255, 202/255],            # Class 6 - Water
            [102/255, 0/255, 203/255],          # Class 7 - Building shadow
            [251/255, 124/255, 145/255],        # Class 8 - Building
            [196/255, 101/255, 0/255],          # Class 9 - Side walk
            [254/255, 251/255, 0/255],          # Class 10 - Yellow curb
            [204/255, 24/255, 97/255]           # Class 11 - Cloth Panels
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
    
    # Parameters
    patchsize = 16
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

    # Create padded arrays with progress bars
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

    # Load model
    cnn = Network(l1=l1_actual, l2=l2_actual, Classes=num_classes)
    if torch.cuda.is_available():
        cnn.cuda()
        cnn.load_state_dict(torch.load(model_path))
    else:
        cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    cnn.eval()
    print(f"Model loaded from: {model_path}")

    # Generate predictions for entire image
    part = 100
    pred_all = np.ones((m1 * n1, 1), dtype='float32')
    
    total_pixels = m1 * n1
    number = total_pixels // part
    
    print(f"Generating predictions for {total_pixels} pixels...")
    
    # Main processing loop with progress bar
    with tqdm(total=number, desc="Processing batches") as pbar:
        for i in range(number):
            D1 = np.empty((part, l1_actual, patchsize, patchsize), dtype='float32')
            D2 = np.empty((part, l2_actual, patchsize, patchsize), dtype='float32')
            count = 0
            
            # Process each pixel in the current batch
            for j in range(i * part, (i + 1) * part):
                row = j // n1
                col = j - row * n1
                row2 = row + pad_width
                col2 = col + pad_width
                patch1 = x1_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                patch2 = x2_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                patch1 = np.reshape(patch1, (patchsize * patchsize, l1_actual))
                patch2 = np.reshape(patch2, (patchsize * patchsize, l2_actual))
                patch1 = np.transpose(patch1)
                patch2 = np.transpose(patch2)
                patch1 = np.reshape(patch1, (l1_actual, patchsize, patchsize))
                patch2 = np.reshape(patch2, (l2_actual, patchsize, patchsize))
                D1[count, :, :, :] = patch1
                D2[count, :, :, :] = patch2
                count += 1

            temp1 = torch.from_numpy(D1)
            temp1_2 = torch.from_numpy(D2)
            if torch.cuda.is_available():
                temp1 = temp1.cuda()
                temp1_2 = temp1_2.cuda()
            
            _, _, temp2, _, _ = cnn(temp1, temp1_2)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_all[i * part:(i + 1) * part, 0] = temp3.cpu()
            del temp1, temp1_2, _, temp2, temp3, D1, D2

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'processed': f"{(i + 1) * part}/{total_pixels}",
                'progress': f"{(i + 1) * part / total_pixels * 100:.1f}%"
            })

    # Process remaining pixels with a separate progress bar
    remaining = total_pixels - (number) * part
    if remaining > 0:
        print(f"Processing remaining {remaining} pixels...")
        D1 = np.empty((remaining, l1_actual, patchsize, patchsize), dtype='float32')
        D2 = np.empty((remaining, l2_actual, patchsize, patchsize), dtype='float32')
        
        with tqdm(total=remaining, desc="Remaining pixels") as pbar:
            count = 0
            for j in range(number * part, total_pixels):
                row = j // n1
                col = j - row * n1
                row2 = row + pad_width
                col2 = col + pad_width
                patch1 = x1_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                patch2 = x2_pad[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
                patch1 = np.reshape(patch1, (patchsize * patchsize, l1_actual))
                patch1 = np.transpose(patch1)
                patch2 = np.reshape(patch2, (patchsize * patchsize, l2_actual))
                patch2 = np.transpose(patch2)
                patch1 = np.reshape(patch1, (l1_actual, patchsize, patchsize))
                patch2 = np.reshape(patch2, (l2_actual, patchsize, patchsize))
                D1[count, :, :, :] = patch1
                D2[count, :, :, :] = patch2
                count += 1
                pbar.update(1)

        temp1 = torch.from_numpy(D1)
        temp1_2 = torch.from_numpy(D2)
        if torch.cuda.is_available():
            temp1 = temp1.cuda()
            temp1_2 = temp1_2.cuda()
        
        _, _, temp2, _, _ = cnn(temp1, temp1_2)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[number * part:total_pixels, 0] = temp3.cpu()
        del temp1, temp1_2, _, temp2, temp3, D1, D2

    # Reshape and save results
    # Add 1 to all predicted labels to account for background class (0 = background, 1+ = actual classes)
    pred_all = np.reshape(pred_all, (m1, n1)) + 1
    
    # Save MATLAB file directly in model directory
    mat_save_path = os.path.join(model_dir, f'{dataset_name}_pred_all.mat')
    scio.savemat(mat_save_path, {'pred_all': pred_all})
    
    # Create and save classification map with progress bar
    print("Creating classification map...")
    best_G = pred_all
    hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
    
    # Color mapping with progress bar
    print(f"Using {len(colors)} colors for {len(class_names)} classes")
    with tqdm(total=best_G.shape[0], desc="Color mapping") as pbar:
        for i in range(best_G.shape[0]):
            for j in range(best_G.shape[1]):
                class_id = int(best_G[i][j])
                if 0 <= class_id < len(colors):
                    hsi_pic[i, j, :] = colors[class_id]
                else:
                    hsi_pic[i, j, :] = [0, 0, 0]  # Black for unknown classes
            pbar.update(1)

    # Save classification map image directly in model directory
    img_save_path = os.path.join(model_dir, f'{dataset_name}_classmap.png')
    classification_map(hsi_pic[2:-2, 2:-2, :], best_G[2:-2, 2:-2], 24, img_save_path)
    
    print(f"Results saved to: {model_dir}")
    print(f"  - MATLAB file: {mat_save_path}")
    print(f"  - Image file: {img_save_path}")
    
    # Print class-color mapping for reference
    print(f"\nClass-Color Mapping for {dataset_name}:")
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        print(f"  Class {i}: {class_name} -> RGB{tuple(color)}")

if __name__ == '__main__':
    # Define dataset paths and model locations - UPDATED to include MUUFL
    datasets = [
        {
            'name': 'Houston2013',
            'data_path': '/beluga/Hackathon15/dataset/Houston2013',
            'model_path': '/beluga/Hackathon15/trained_models_train_val_test/Houston2013/CALC/CALC_Houston2013.pkl'
        },
        {
            'name': 'Trento', 
            'data_path': '/beluga/Hackathon15/dataset/Trento',
            'model_path': '/beluga/Hackathon15/trained_models_train_val_test/Trento/CALC/CALC_Trento.pkl'
        },
        {
            'name': 'MUUFL',
            'data_path': '/beluga/Hackathon15/dataset/MUUFL',
            'model_path': '/beluga/Hackathon15/trained_models_train_val_test/MUUFL/CALC/CALC_MUUFL.pkl'
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
            print("Please train the model first using train_saeid_2.py")