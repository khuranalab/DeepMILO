#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:23:21 2018

@author: tat2016

To take a variant-file from user, generate input files and run the model to calculate loop probabilities

"""

import multiprocessing as mp
import time
from keras.models import load_model
import os

import h5py

import numpy as np

from keras.utils import multi_gpu_model
import sys

#import variant_class
import boundary_class
import data_generator
import common_data_generation as cdg

jobID = str(int(time.time()))
nbr_job = 4
segment_size = 4000
ref_genome_file = 'Homo_sapiens.GRCh37.75.dna.primary_assembly.fa'
consitutive_loop_data_sampleid = lambda x : '%s/%s_loop_%s.mat' % (jobID, x, jobID)        

if not os.path.exists(jobID):
    os.makedirs(jobID) 
    
    
outputFolder = str(jobID) + '_output'
inputFolder = str(jobID)

loopFile = 'loopDB/constitutive_loops.xlsx'
inputF = 'tsv'
ssmFileName = sys.argv[1]
svFileName = sys.argv[2]


def process_patient(loops, varlist, segment_size, ref_genome_file, output_file):
    
    [looplist, _] = cdg.get_loop(loops, varlist, segment_size, isNormalize = False, isNonLoop = False, noLargeSV=False)
    
    
    looplist = [x for x in looplist if len(x.b1.variants) > 0 or len(x.b2.variants) > 0]
    if len(looplist) > 0:
        print(output_file)
        print('number of affected loops: %d' % (len(looplist)))
    
        if output_file:
            cdg.output_loop(looplist, ref_genome_file, segment_size, output_file, None)



#variants = variant_class.extract_variants(fileName, '', all_variant = True)

ssmVariants = {}
svVariants = {}

print('Reading variants ...')

if ssmFileName:
    ssmVariants = cdg.getSSM(ssmFileName)
if svFileName:
    svVariants = cdg.getSV(svFileName)

variants = {**ssmVariants, **svVariants}

if '.xlsx' in loopFile:
    loops = cdg.getLoopXLSX(loopFile)
else:
    loops = boundary_class.read_loop(loopFile)

print('Creating input data ...')

pool = mp.Pool(processes = nbr_job)    
result = [pool.apply_async(process_patient, args=(loops, variants[k], segment_size,
                                                      ref_genome_file, consitutive_loop_data_sampleid(k))) for k in variants]
pool.close()
pool.join()

print('Done preparing data')

''' '''

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
loop_model_file = 'loop_pred_4k_1gpu.h5' 
ngpu = 4

batch_size = 24 * ngpu

segment_size = 4000
version = ''
nbr_feature = 10
rnn_len = 800
params = {'dim': (segment_size, nbr_feature),
          'batch_size': batch_size,
          'n_channels': 1,
          'rnn_len': rnn_len}


def get_prediction(model_file, data_file, isloop=True):
    
    if isinstance(model_file, str):
        model = load_model(model_file)
    else:
        model = model_file
    
    rs = {}
    
    if isloop:
        generator = data_generator.DataGeneratorLoopSeq(data_file, None, shuffle = False, use_reverse=False, **params)
    else:
        generator = data_generator.DataGenerator(data_file, None, shuffle = False, use_reverse=False, **params)

    prediction = model.predict_generator(generator, verbose=1)
    #print(prediction, len(generator.ids_list))
    prediction = np.array(prediction).ravel()
    
    labels = generator.ids_list
    for i,k in enumerate(labels):
        if k in rs and abs(rs[k] - prediction[i]) > 0.00001:
            print('error in get_prediction, sample order is messed up')
        rs[k] = prediction[i]
    
    return(rs)
    
def output_prob(rs, file_name):
    fout = open(file_name,'w')
    for k,v in rs.items():
        fout.write('%s: \t %.10f\n' % (k,v))
      
    fout.close()
    

if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)    


print('Loading model')
model = load_model(loop_model_file) 
model = multi_gpu_model(model, gpus = ngpu)


patient_files = os.listdir(inputFolder)

print('Running model on input data')
for i in patient_files:
    
    input_file = os.path.join(inputFolder, i)
    
    dataset = h5py.File(input_file, 'r')
    
    print(input_file)
    if len(dataset.items()) == 0:
        continue
    
    output_file = os.path.join(outputFolder, i.replace('.mat','.txt'))
    
    rs = get_prediction(model, input_file)
    
    output_prob(rs, output_file)




