import os
import medpy
from PIL import Image, ImageOps
import numpy as np
from medpy import metric
from sklearn.metrics import mean_absolute_error
import math
import glob
#import pprint
#searchdir = r'/Desktop/Zubair/Objective_3/Objective_3/Validate_obj_3/direct_extrapolation/sel_18 (copy)/HS_Results/results_auto'  # your search starts in this directory (your root) 
def nss(s_map,gt):
	#gt = discretize_gt(gt)
	s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

	x,y = np.where(gt==1)
	temp = []
	for i in zip(x,y):
		temp.append(s_map_norm[i[0],i[1]])
	return np.mean(temp)


def cc(s_map,gt):
	s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
	gt_norm = (gt - np.mean(gt))/np.std(gt)
	a = s_map_norm
	b= gt_norm
	r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
	return r

def Load_data(File_path):
    HS_files = os.listdir(File_path)
    #print(HS_files)
    final_results =[]
    #ind_res = []
    #global highest = [] 
    global numpydata_in, numpydata_out
    for file in HS_files:
        list_sal = []
        highest = []
        highest_list = []
        input_img = os.path.join(File_path, file)
        HS_files_local = os.listdir(input_img)
       # print("HS_files_local", HS_files_local)
        for fil in HS_files_local:
            current = 0.0
            ind_res = []
            #print("fil is", fil)
            #(base, ext) = os.path.splitext(fil)
            if fil.endswith('.jpg'):
                
                gt = Image.open(File_path + './'+file+'/' + fil)
                #print("ground truth", gt)
                numpydata_in = np.array(gt).astype('uint8')
                numpydata_in = np.max(numpydata_in, axis = 2)
                numpydata_in = numpydata_in / 255
    
                #print(numpydata_in.shape)
            if fil.endswith('.png'):
                pred_im = Image.open(File_path + './'+file+'/' + fil)
                list_sal.append(pred_im)    
                #print("list sal", list_sal)
             
        for i in range(len(list_sal)):
            numpydata_out = np.array(list_sal[i]).astype('uint8')
            numpydata_out = numpydata_out / 255
            #print("numpydata_out",numpydata_out)
            mae = mean_absolute_error(numpydata_in, numpydata_out)
            ns = nss(numpydata_out, numpydata_in)
            csee = cc(numpydata_out, numpydata_in)
            prec_1 = medpy.metric.binary.precision(numpydata_out, numpydata_in)
            rec_1 = medpy.metric.binary.recall(numpydata_out, numpydata_in)
            if (0.09*prec_1+rec_1) != 0:
                f1_1  =  1.09*((prec_1*rec_1))/(0.09*prec_1+rec_1)

            else:
                f1_1 = 0 
                i =i+1             
#if f1_1 >= current:
            ind_res.append(prec_1)
            ind_res.append(rec_1)
            ind_res.append(f1_1)
            ind_res.append(ns)
            ind_res.append(csee)
            ind_res.append(mae)
        
                     
       # print("ind_res", ind_res)  
        #print("ind_res", len(ind_res))
        ind_res_arr = np.asarray(ind_res)
        ind_res_reshape = ind_res_arr.reshape(9,6)
          
        #print("f1 scores", ind_res_arr[:][2])
        for j in range(9):
            if ind_res_reshape[j][2] >= current:
                highest = ind_res_reshape[j][:]
                current = ind_res_reshape[j][2]
            else:
                j = j+1; 

        #print("highest", highest)
        highest_list.append(highest)
        #print("highest_list", highest_list)
        #print("highest list length", len(highest_list))
        final_results.append(highest_list)
        #print("final results", final_results)
        #print("final results shape", final_results.shape())
        log_path = File_path + '/' + file + '/' + "chosen_res.txt"
        with open(log_path, "w") as chosen_res:
            chosen_res.write("prec: %f Rec: %f f1: %f NSS: %f CC: %f MAE: %f \n" %(prec_1, rec_1, f1_1, ns, csee, mae))
    final_results_arr = np.asarray(final_results)
    
    average_res = np.mean(final_results_arr, axis =0)
    #print("average results", average_res)
    print(np.shape(final_results))
    #print("fnal resulsts", final_results)
   # log_path_final = File_path + '/'+ "final_res.txt"
    #with open(log_path_final, "w") as final_res:
      #  final_res.write("final results", final_results)
    log_path_ave = File_path + '/' + "final_result.txt"
    with open(log_path_ave,"w") as final_res:
        for ent in average_res:
            final_res.write("%s \t" %ent)    
File_path = './HS_images/'
count = 0
Load_data(File_path)
   
  
