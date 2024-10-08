import numpy as np
import os.path as path
import nibabel as nib
from skimage.transform import resize
import os
import glob
import h5py
import json


def data_prpcess_normal(image_path,label_path,to_path):
    """for MSD NIH SYN datasets"""
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    names = glob.glob(path.join(image_path,'*.gz'))
    names.sort()
    names = [path.split(f)[1] for f in names]

    #总结：根据GT，对GT向外扩充32个像素，然后裁剪
    pad = [32,32,32]
    for img_name in names:
        label_name = 'label' + img_name.split('_')[1] # TODO: modify this.
        # label_name = 'label' + img_name.split('_')[0][5:8] # for synapse

        image = nib.load(path.join(image_path, img_name))
        spacing = image.affine[[0,1,2], [0,1,2]]
        
        # deciding the direction
        ind = ((-spacing>0)-0.5)*2
        image = image.get_data()
        image = np.transpose(image,[1,0,2])
        image = image[::int(ind[1]),::int(ind[0]),::int(ind[2])]
        
        # resample to 1mm
        new_size = (np.array(image.shape)*np.abs(spacing)).astype(np.int)
        image = resize(image.astype(np.float64),new_size)

        label = nib.load(path.join(label_path, label_name))
        spacing = label.affine[[0,1,2],[0,1,2]]
        label = label.get_data()
        label = np.transpose(label,[1,0,2])
        ind = ((-spacing>0)-0.5)*2
        label = label[::int(ind[1]),::int(ind[0]),::int(ind[2])]
        label = resize(label.astype(np.float64),new_size,anti_aliasing=False,order=0)
        print(img_name, 'loaded', new_size, spacing)
        
        # get the bounding box of foreground
        tempL = np.array(np.where(label>0))
        print(tempL[0].shape)
        bbox = np.array([[max(0, np.min(tempL[0])-pad[0]), min(label.shape[0], np.max(tempL[0])+pad[0])], \
        [max(0, np.min(tempL[1])-pad[1]), min(label.shape[1], np.max(tempL[1])+pad[1])], \
        [max(0, np.min(tempL[2])-pad[2]), min(label.shape[2], np.max(tempL[2])+pad[2])]])
        center = np.mean(bbox,1).astype(int)
        bbL = bbox[:,1]-bbox[:,0]
        L = int(np.max(bbox[:,1]-bbox[:,0]))
        print(L)

        # extract a cubic box that contain all the foreground
        out = \
            image[max(0,center[0]-int(L/2)):min(label.shape[0],center[0]-int(L/2)+L),\
            max(0,center[1]-int(L/2)):min(label.shape[1],center[1]-int(L/2)+L),\
            max(0,center[2]-int(L/2)):min(label.shape[2],center[2]-int(L/2)+L)]
        Shape = list(out.shape)
        Shape.append(2)
        print(Shape)
        Out_img = out
        Out_label = \
        label[max(0,center[0]-int(L/2)):min(label.shape[0],center[0]-int(L/2)+L),\
        max(0,center[1]-int(L/2)):min(label.shape[1],center[1]-int(L/2)+L),\
        max(0,center[2]-int(L/2)):min(label.shape[2],center[2]-int(L/2)+L)]
        
        path_prefix = path.join(to_path, img_name.split('.')[0])
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        np.save(path.join(path_prefix, 'img.npy'), Out_img.astype(np.int16))
        np.save(path.join(path_prefix, 'label.npy'), Out_label.astype(np.int8))
        np.save(path.join(path_prefix, 'merge.npy'), np.stack((Out_img,Out_label),axis=-1).astype(np.int16))

def data_prpcess_ACDC(from_ACDC_path,to_path):
    """for ACDC datasets"""
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    with open(from_ACDC_path + f"/train.list", "r") as f1:
            train_list = f1.readlines()
    train_list = [item.replace("\n", "") for item in train_list]
    with open(from_ACDC_path + f"/val.list", "r") as f1:
            val_list = f1.readlines()
    val_list = [item.replace("\n", "") for item in val_list]
    with open(from_ACDC_path + f"/test.list", "r") as f1:
            test_list = f1.readlines()
    test_list = [item.replace("\n", "") for item in test_list]
    
    total_list = train_list+ val_list + test_list
    
    names = glob.glob(path.join(from_ACDC_path,"data",'*.h5'))
    names.sort()
    names = [path.split(f)[1] for f in names]
    dataset_info_dict = {"ACDC_train":[],"ACDC_val":[],"ACDC_test":[]}

    for img_name in names:
        h5f = h5py.File(path.join(from_ACDC_path,"data", img_name), "r")
        Out_img = h5f["image"][:]
        Out_label = h5f["label"][:]

        path_prefix = path.join(to_path, img_name.split('.')[0])
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        np.save(path.join(path_prefix, 'img.npy'),Out_img)
        np.save(path.join(path_prefix, 'label.npy'), Out_label.astype(np.int8))
        np.save(path.join(path_prefix, 'merge.npy'), np.stack((Out_img,Out_label),axis=-1))
        if img_name.split('.')[0] in train_list:
            dataset_info_dict["ACDC_train"].append(path.join(img_name.split('.')[0], 'merge.npy'))
        if img_name.split('.')[0] in val_list:
            dataset_info_dict["ACDC_val"].append(path.join(img_name.split('.')[0], 'merge.npy'))
        if img_name.split('.')[0] in test_list:
            dataset_info_dict["ACDC_test"].append(path.join(img_name.split('.')[0], 'merge.npy'))
    
    with open(path.join(to_ACDC_path,"Multi_all.json"), 'w') as json_file:
        json.dump(dataset_info_dict, json_file, indent=4)  # indent=4 格式化输出

if __name__ == "__main__":
    image_path = '<path-to-the-image>/nih_data/Pancreas-CT/data' # TODO: modify this.
    label_path = '<path-to-the-data>/nih_data/Pancreas-CT/TCIA_pancreas_labels-02-05-2017' # TODO: modify this.
    to_path = 'data/nih' # TODO: modify this.
    # data_prpcess_normal(image_path,label_path,to_path) # for MSD NIH SYN datasets
    from_ACDC_path = "/media/grozta/SOYO/DATASET/ACDC"
    to_ACDC_path = "/media/grozta/SOYO/DATASET/ACDC4AVE"
    data_prpcess_ACDC(from_ACDC_path,to_ACDC_path)