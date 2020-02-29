import os
import shutil
import sys

"""
root
    train
        sequences
            image
            depth
            labels
            velodyne
    val
    test
"""
def create_list(root):
    dest_path = './data/'
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)
    split_folders = os.listdir(root)
    for split in split_folders:
        if split == '.DS_Store':
            continue
        abs_split = os.path.join(root, split)
        sequence_folders = os.listdir(abs_split)
        for sequence in sequence_folders:
            if sequence == '.DS_Store':
                continue
            abs_sequence = os.path.join(abs_split, sequence)
            image_folder = os.path.join(abs_sequence, 'image')
            label_folder = os.path.join(abs_sequence, 'labels')
            # context_folder = os.path.join(abs_sequence, 'context_temporal_road_prior')
            labels = os.listdir(label_folder)
            for label in labels:
                if label == '.DS_Store':
                    continue
                label_path = os.path.join(label_folder, label)
                image_path = os.path.join(image_folder, label)
                # context_path = os.path.join(context_folder, label)
                if split == 'train':
                    with open(dest_path+'train.lst', "a") as f:
                        f.write(image_path + "    " + label_path +"\n")
                if split == 'val':
                    if (sequence == 'vindhya_2') or (sequence == 'stadium_3'):
                        with open(dest_path+'test.lst', "a") as f:
                            f.write(image_path + "    " + label_path+"\n")
                    else:
                        with open(dest_path+'val.lst', "a") as f:
                            f.write(image_path + "    " + label_path+"\n")

    return

if __name__ == "__main__":
    root = '/scratch/ash/IIIT_Labels/'
    create_list(root)
