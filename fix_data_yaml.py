#!/usr/bin/env python3
import os
import yaml

def fix_data_yaml():
    data_file = os.path.join('dataset', 'data.yaml')
    
    # Read the data.yaml file
    with open(data_file, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f'Original class names: {data["names"]}')
    print(f'Number of classes: {data["nc"]}')
    
    # Update paths to absolute paths
    print('Updating train/val/test paths')
    base_dir = os.path.abspath('dataset')
    data['train'] = os.path.join(base_dir, 'train', 'images')
    data['val'] = os.path.join(base_dir, 'valid', 'images')
    data['test'] = os.path.join(base_dir, 'test', 'images')
    
    # Write updated data.yaml
    with open(data_file, 'w') as f:
        yaml.dump(data, f)
    
    print(f'Updated data.yaml with absolute paths')

if __name__ == "__main__":
    fix_data_yaml() 