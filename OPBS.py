"""
Implementation of OPBS algorithm

Detail can be found in paper "A Geometry-Based Band Selection
Approach for Hyperspectral Image Analysis" in Algorithm 2
"""


import numpy as np
#from MEV_SFS import load_gdal_data
import scipy.io as scio
import mat73
import cv2
import os
from os import mkdir
import time
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
def opbs(image_data, sel_band_count, removed_bands=None):
    if image_data is None:
        return None

    bands = image_data.shape[1]
    band_idx_map = np.arange(bands)

    if not (removed_bands is None):
        image_data = np.delete(image_data, removed_bands, 1)
        bands = bands - len(removed_bands)
        band_idx_map = np.delete(band_idx_map, removed_bands)

    # Compute covariance and variance for each band
    # TODO: data normalization to all band
    data_mean = np.mean(image_data, axis=0)
    image_data = image_data - data_mean
    data_var = np.var(image_data, axis=0)
    h = data_var * image_data.shape[0]
    op_y = image_data.transpose()

    sel_bands = np.array([np.argmax(data_var)])
    last_sel_band = sel_bands[0]
    current_selected_count = 1
    sum_info = h[last_sel_band]
    while current_selected_count < sel_band_count:
        for t in range(bands):
            if not (t in sel_bands):
                op_y[t] = op_y[t] - np.dot(op_y[last_sel_band], op_y[t]) / h[last_sel_band] * op_y[last_sel_band]

        max_h = 0
        new_sel_band = -1
        for t in range(bands):
            if not (t in sel_bands):
                h[t] = np.dot(op_y[t], op_y[t])
                if h[t] > max_h:
                    max_h = h[t]
                    new_sel_band = t
        sel_bands = np.append(sel_bands, new_sel_band)
        last_sel_band = new_sel_band
        sum_info += max_h
        estimate_percent = sum_info / (sum_info + (bands - sel_bands.shape[0]) * max_h)
        #print(estimate_percent)
        current_selected_count += 1

    #print(band_idx_map[sel_bands] + 1)
    #print(np.sort(band_idx_map[sel_bands] + 1))

    return sel_bands

st = time.process_time()
def main():
    File_path = './HS_images/' 
    HS_files = os.listdir(File_path)
    #print("HS_files", HS_files)
    num = 0;
    save_path = './HS_Results/'
    mkdir(save_path)
    sal_result_path = os.path.join(save_path, 'sal_result')
    mkdir(sal_result_path)

    remove_bands = []
    for file in HS_files:
        input_img = os.path.join(File_path, file)
        #print("Reading File[%d/%d]: %s" % (num, len(HS_files), input_img))
        mat_dict = mat73.loadmat(input_img)
        
        for key in mat_dict:
            if type(mat_dict[key]) is np.ndarray:
                image_data = mat_dict[key]  # type: np.ndarray
        cols = image_data.shape[1]
        rows = image_data.shape[0]
        bands = image_data.shape[2]
        #print("shape of image data before reshaping", np.shape(image_data))
        image_data = image_data.reshape(cols * rows , bands)
        sel_bands_list = []
        sel_bands_list = opbs(image_data, 3, remove_bands)
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
        print("total cpu calculation time for opbs is", tot_time)
        file = file.rsplit('.', 1)
        save_sal_name = sal_result_path + '/' + file[0] + '.mat'
    #cv2.imwrite(save_sal_name, salmap_final)
        scio.savemat(save_sal_name, {'mydata': image_out_reshape})
    
    
if __name__ == "__main__":
    main()
