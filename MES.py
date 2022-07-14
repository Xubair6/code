"""
Implementation of MEV-SFS algorithm
Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 1
"""

from osgeo import gdal
import numpy as np
from tqdm_progress import TqdmUpTo
import scipy.io as scio
import matplotlib.pyplot as plt
import h5py
import mat73
from PIL import Image
import PIL
import cv2
import time
import os
def load_gdal_data(data_path):
    dataset: gdal.Dataset = gdal.Open(data_path, gdal.GA_ReadOnly)
    if not dataset:
        return None

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    bands = dataset.RasterCount

    with TqdmUpTo(unit='%', unit_scale=False, miniters=1, total=100,
                  desc="reading file %s" % data_path) as t:
        databuf_in: np.ndarray = dataset.ReadAsArray(callback=t.update_to)
    databuf_in = databuf_in.reshape(bands, rows * cols)
    return databuf_in.transpose()


def mev_sfs(data: np.ndarray, sel_band_count, removed_bands=None):
    bands = data.shape[1]
    band_idx_map = np.arange(bands)

    if not (removed_bands is None):
        data = np.delete(data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)

    data_mean = np.mean(data, axis=0)
    data = data - data_mean
    data_var = np.var(data, axis=0)
    cov_mat = np.matmul(data.transpose(), data) / (data.shape[0])

    sel_bands = np.array([np.argmax(data_var)])
    current_selected_count = 1
    while current_selected_count < sel_band_count:
        max_mev = 0
        new_sel_band = -1
        list_sel_bands=[]
        for i in range(bands):
            if not (i in sel_bands):
                new_sel_bands = np.append(sel_bands, i)
                a = np.ix_(new_sel_bands, new_sel_bands)
                new_cov_mat = cov_mat[np.ix_(new_sel_bands, new_sel_bands)]
                mev = np.linalg.det(new_cov_mat)
                if mev > max_mev:
                    max_mev = mev
                    new_sel_band = i
        sel_bands = np.append(sel_bands, new_sel_band)
        #list_sel_bands = list_sel_bands.append(sel_bands)
        current_selected_count += 1
    print(sel_bands)
    print(band_idx_map[sel_bands] + 1)
    print(np.sort(band_idx_map[sel_bands] + 1))
    #im = Image.fromarray(sel_bands, 'RGB');
    #im = im.save('0041.jpg')
    #print("inside function", list_sel_bands)
    return sel_bands
    #cv2.imwrite('00411.jpg',sel_bands) 
def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
st = time.process_time()
def main():
    File_path = './HS_images/' 
    HS_files = os.listdir(File_path)
    print("HS_files", HS_files)
    num = 0;
    save_path = './HS_Results/'
    mkdir(save_path)
    sal_result_path = os.path.join(save_path, 'sal_result')
    mkdir(sal_result_path)

    remove_bands = []
    for file in HS_files:
        input_img = os.path.join(File_path, file)
        print("Reading File[%d/%d]: %s" % (num, len(HS_files), input_img))
        mat_dict = mat73.loadmat(input_img)
        
    
        for key in mat_dict:
            if type(mat_dict[key]) is np.ndarray:
                image_data = mat_dict[key]  # type: np.ndarray
    
        cols = image_data.shape[1]
        rows = image_data.shape[0]
        bands = image_data.shape[2]
        image_data = image_data.reshape(cols * rows , bands)
        sel_bands_list = []
        sel_bands_list = mev_sfs(image_data, 3, remove_bands)
        for j in range(0,81): 
            #print("i value is", i)
            if sel_bands_list[0] == j:
                image_out_0 = image_data[:,j]
            else: 
                j=j+1
        for j in range(0,81): 
            #print("i value is", i)
            if sel_bands_list[1] == j:
                image_out_1 = image_data[:,j]
            else: 
                j=j+1
        for j in range(0,81): 
            #print("i value is", i)
            if sel_bands_list[2] == j:
                image_out_2 = image_data[:,j]
            else: 
                j=j+1
        image_out = np.dstack((image_out_0,image_out_1, image_out_2))
        image_out_reshape = np.reshape(image_out, (768, 1024,3))
        et = time.process_time()
        tot_time = et-st
        print("total time for spu processing is", tot_time)
        file = file.rsplit('.', 1)
        save_sal_name = sal_result_path + '/' + file[0] + '.mat'
    #cv2.imwrite(save_sal_name, salmap_final)
        scio.savemat(save_sal_name, {'mydata': image_out_reshape})
    
    
if __name__ == "__main__":
    main()
