#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:23:21 2018

@author: tat2016

To take a variant file from ICGC and generate input files and run the model to calculate loop probabilities

The program requires Homo_sapiens.GRCh37.75.dna.primary_assembly.fa in the same folder

Input:
The program requires 2 input files:
    + a simple somatic mutation file and
    + a structural variant file from ICGC (in .tsv format)


Run:
     python main.py ssm.tsv sv.tsv

Output: 2 folders
    + one folder contain input files, one for each patient, contains sequences of loops taking into consideration mutations from patients
    + another folder contains output loop probability prediction for each patient
-

"""

import multiprocessing as mp
import time
from keras.models import load_model
import os

import h5py

import numpy as np

from keras.utils import multi_gpu_model

import copy

import argparse
import sys
import re

from common_object import Loop, Variant, Boundary
import common_function as cf
import output_function as of
import data_generator


ref_genome_file = 'Homo_sapiens.GRCh37.75.dna.primary_assembly.fa'
cons_loop_file = 'loopDB/all_insulator_loop.bed'


# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
loop_model_file = 'model/loop_pred_4k_1gpu.h5'
ngpu = 1 # number of gpu to use, combine with os.environ above if necessary

nbr_job = 1  # number of parallel jobs when creating data for patients

# parameters to load data
batch_size = 24 * ngpu

# constants
segment_size = 4000
version = ''
nbr_feature = 5
rnn_len = 800


def getSSM_bedfile(ssm_file, cohort='Eric_CLL'):

    cohort = cohort.replace('Eric_','')

    variants = {}
    count = 0
    with open(ssm_file) as fi:
        for ln in fi.readlines():
            st = re.split('[\t\n]+',ln)

            # print('line:', ln)
            # print('st:', st)

            if st[6] != cohort:
                continue


            chrom = st[0]
            start = int(st[1])
            end = int(st[2])
            sample_id = st[3]

            if sample_id not in variants:
                variants[sample_id] = []

            ref = st[4]
            alt = st[5]
            vt = 'snp'
            svtype = None
            gt = '1|1'
            count += 1
            var = Variant(chrom, start, end, vt, svtype, ref, alt, gt)

            variants[sample_id].append(var)

    print('number of samples:{}, number of mutation:{}, mutations/sample:{}'.format(len(variants), count,
                                                                                    float(count)/len(variants)))
    return variants


def getSSM(ssm_file):
    variants = {}  # icgc_sample_id: variants
    processed_mut = set()  # processsed mutation to handle duplicate records

    # for i in range(len(ssm)):
    with open(ssm_file, 'r') as fin:

        ln = fin.readline()

        fields = re.split('\t', ln)
        field2Id = {}
        for i in range(len(fields)):
            field2Id[fields[i]] = i

        icgc_mutation_id = field2Id['icgc_mutation_id']
        chromosome_id = field2Id['chromosome']
        chromosome_start_id = field2Id['chromosome_start']
        chromosome_end_id = field2Id['chromosome_end']
        mutation_type_id = field2Id['mutation_type']
        reference_genome_allele_id = field2Id['reference_genome_allele']

        if 'tumour_genotype' in fields:
            tumour_genotype_id = field2Id['tumour_genotype']
        else:
            mutated_to_allele_id = field2Id['mutated_to_allele']

        icgc_sample_id = field2Id['icgc_sample_id']

        for ln in fin.readlines():

            st = re.split('\t', ln)

            mut_id = st[icgc_mutation_id]

            if mut_id in processed_mut:
                #print(mut_id)
                continue

            processed_mut.add(mut_id)

            chrid = str(st[chromosome_id]).upper()
            if not re.search('^[0-9XY]+', chrid):
                print(chrid)
                continue

            start = int(st[chromosome_start_id]) - 1
            end = int(st[chromosome_end_id])
            vt = st[mutation_type_id]

            ref = st[reference_genome_allele_id]

            if 'tumour_genotype' in fields:
                # control_gt = ssm.loc[i, 'control_genotype']
                tumor_gt = st[tumour_genotype_id]
            else:
                tumor_gt = st[mutated_to_allele_id]

            sample_id = st[icgc_sample_id]

            if sample_id not in variants:
                variants[sample_id] = []

            '''
            mutating by replacing ref. with alt.
            if alt == '', in insertation, it means no insertation ( if ref == '')
                          in deletion, it means deleting ref

            if insertion, ref is always -, must be converted to ''
            '''
            # Variant(sample, rc.CHROM, start, end, dvt, dsvtype, rc.REF, rc.ALT, gt )

            ref = '' if ref == '-' else ref

            if re.search('substitution', vt):
                vt = 'snp'
                svtype = ''
            elif re.search('deletion', vt):
                vt = 'indel'
                svtype = 'del'
            elif re.search('insertion', vt):
                vt = 'indel'
                svtype = 'ins'

            # alternative
            alt = re.split('[|/]', tumor_gt)
            alt = [x if x != '-' else '' for x in alt]  # convert '-' to empty

            if len(alt) == 1:  # if there is only one allele, make another from it
                alt.append(alt[0])

            # 0: for reference seq, therefore + 1
            # if insertion, gt = '' can be 0
            gt = '|'.join([str(alt.index(x) + 1) for x in alt])

            if vt == 'snp' and ref == '':
                print('error, snp but ref. is not available')

            #print('chrom:{}, start:{}, end:{}, vt:{}, subtype:{}, ref:{}, alt:{}'.format(chrid, start, end, vt, svtype, ref, alt))

            var = Variant('chr' + chrid, start, end, vt, svtype, ref, alt, gt)
            variants[sample_id].append(var)

    print('Number of sample:{}, number of variants{}'.format(len(variants), len(processed_mut)))
    return variants


def getSV(stvm_file):
    variants = {}

    processed_sv = set()

    with open(stvm_file, 'r') as fin:
        ln = fin.readline()
        fields = re.split('\t', ln)
        field2Id = {}
        for i in range(len(fields)):
            field2Id[fields[i]] = i

        sv_header_id = field2Id['sv_id']
        variant_type_id = field2Id['variant_type']
        chr_from_id = field2Id['chr_from']
        chr_to_id = field2Id['chr_to']
        icgc_sample_id = field2Id['icgc_sample_id']
        chr_from_bkpt_id = field2Id['chr_from_bkpt']
        chr_to_bkpt_id = field2Id['chr_to_bkpt']

        for ln in fin.readlines():
            st = re.split('\t', ln)

            sv_id = st[sv_header_id]

            if sv_id in processed_sv:
                continue

            processed_sv.add(sv_id)

            svtype = st[variant_type_id]

            #        if svtype == 'unbalanced translocation':
            #            continue

            chrid_from = st[chr_from_id].upper()
            chrid_to = st[chr_to_id].upper()

            # ignore inter-chromosome SV
            if (chrid_from != chrid_to) or (not re.search('^[0-9XY]+', chrid_from)) \
                    or (not re.search('^[0-9XY]+', chrid_to)):
                continue

            sample_id = st[icgc_sample_id]
            if sample_id not in variants:
                variants[sample_id] = []

            chrom = 'chr' + chrid_from
            start = int(st[chr_from_bkpt_id]) - 1
            end = int(st[chr_to_bkpt_id])

            if svtype == 'deletion':
                svtype = 'DEL'
            elif svtype == 'inversion':
                svtype = 'INV'
            elif svtype == 'tandem duplication':
                svtype = 'DUP'
            else:
                sys.stderr.write('wrong svtype:{}\n'.format(svtype))


            ref = ''
            alt = ''
            gt = '1|1'
            vt = 'sv'

            var = Variant(chrom, start, end, vt, svtype, ref, alt, gt)
            variants[sample_id].append(var)

    for sample, vars in variants.items():
        for i in range(len(vars) - 1):
            if vars[i].chrid == vars[i + 1].chrid and vars[i].start == vars[i + 1].start\
                and vars[i].end == vars[i + 1].end:

                print('duplicate variant, sample:{}, variants:{}'.format(sample, str(vars[i])))

    print('Number of SV samples:', len(variants))
    return variants

def makeOutputFolder():
    jobID = str(int(time.time()))

    # generate names and create output folders
    if not os.path.exists(jobID):
        os.makedirs(jobID)

    outputFolder = str(jobID) + '_output'
    inputFolder = str(jobID)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    return(jobID, inputFolder, outputFolder)


###########################################################
def get_prediction(model_file, data_file, isloop=True):
    """Run prediction for a data file"""

    params = {'dim': (segment_size, nbr_feature),
              'batch_size': batch_size,
              'n_channels': 1,
              'rnn_len': rnn_len}

    if isinstance(model_file, str):
        model = load_model(model_file)
    else:
        model = model_file

    rs = {}

    generator = data_generator.DataGeneratorLoopSeq(data_file, None, shuffle=False, use_reverse=False, **params)

    prediction = model.predict_generator(generator, verbose=1)
    # print(prediction, len(generator.ids_list))
    prediction = np.array(prediction).ravel()

    labels = generator.ids_list
    for i, k in enumerate(labels):
        if k in rs and abs(rs[k] - prediction[i]) > 0.00001:
            print('error in get_prediction, sample order is messed up')
        rs[k] = prediction[i]

    return (rs)


def output_prob(rs, file_name):
    """Output prediction """
    fout = open(file_name, 'w')
    for k, v in rs.items():
        fout.write('%s: \t %.10f\n' % (k, v))

    fout.close()

def process_eachpatient(consloop, varlist, segment_size, ref_genome_file, output_file):

    if os.path.exists(output_file):
        return

    loops = copy.deepcopy(consloop)


    boundaries = [x.b1 for x in loops] + [x.b2 for x in loops]

    cf.overlap_variants(boundaries, varlist)

    varlens = [len(x.variants) for x in boundaries]
    print('min: {}, max:{}, mean:{}, median{}, sum:{}'.format(np.min(varlens), np.max(varlens), np.mean(varlens), np.median(varlens),np.sum(varlens)))

    looplist = [x for x in loops if len(x.b1.variants) > 0 or len(x.b2.variants) > 0]

    if len(looplist) > 0:
        print(output_file)
        print('number of affected loops: %d' % (len(looplist)))

        if output_file:

                of.output_loop(looplist, ref_genome_file, segment_size, output_file, None)


def generate_patientdata(variants, loops, nbr_job, jobID):
    print('Creating input data for all patients...')

    # data file for each patient
    outputfile_loop_data_sampleid = lambda x: '%s/%s_loop_%s.mat' % (jobID, x, jobID)

    if nbr_job > 1:
        pool = mp.Pool(processes=nbr_job)
        result = [pool.apply_async(process_eachpatient, args=(loops, variants[k], segment_size,
                                                          ref_genome_file, outputfile_loop_data_sampleid(k))) for k in
                  variants]
        pool.close()
        pool.join()

    else:
        for k in variants:
            process_eachpatient(loops, variants[k], segment_size, ref_genome_file, outputfile_loop_data_sampleid(k))

    print('Done preparing data')


def makePrediction(loop_model_file, inputFolder, outputFolder):

    print('Loading model')
    model = load_model(loop_model_file)
    if ngpu > 1:
        model = multi_gpu_model(model, gpus=ngpu)

    patient_files = os.listdir(inputFolder)

    print('Running model on input data')
    for i in patient_files:

        input_file = os.path.join(inputFolder, i)

        dataset = h5py.File(input_file, 'r')

        print(input_file)
        if len(dataset.items()) == 0:
            continue

        output_file = os.path.join(outputFolder, i.replace('.mat', '.txt'))

        rs = get_prediction(model, input_file)

        output_prob(rs, output_file)

def read_loop(loop_file):
    '''
    Loops are in bed format: chr1	803273	807273	.	.	chr1	1225118	1229118	.	.
    :param loop_file:
    :return:
    '''
    loop_list = []
    with open(loop_file) as fi:
        for ln in fi.readlines():
            st = ln.split()
            b1 = Boundary(st[0], st[1], st[2])
            b2 = Boundary(st[5], st[6], st[7])
            lp = Loop(b1, b2)
            loop_list.append(lp)

    return loop_list



############################## Generating data

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ssm', dest='ssm', default="", help='simple somatic mutation file')
    parser.add_argument('-sv', dest='sv', default="", help='structural variant file')
    parser.add_argument('-format', dest='format', default="tsv", help='file format (.bed) for TCGA WGS data, specify -ssm only (not -sv)')
    parser.add_argument('-loop', dest='loop_file', default=cons_loop_file, help='loop file')

    opts = parser.parse_args()

    # retrieve mutations file name from arguments
    inputF = opts.format #format file
    ssmFileName = opts.ssm
    svFileName = opts.sv

    loop_file = opts.loop_file

    ssmVariants = {}
    svVariants = {}

    print('Reading variants ...')

    if inputF == 'tsv':
        if ssmFileName:
            ssmVariants = getSSM(ssmFileName)
        if svFileName:
            svVariants = getSV(svFileName)
        variants = {**ssmVariants, **svVariants}

    elif inputF == 'bed':
        variants = getSSM_bedfile(ssmFileName)

    loops = read_loop(loop_file)


    #calculate probability for loops without mutations
    if not ssmFileName and not svFileName:
        print('No variants found, calculating loop probability for loops ...')

        output_seq_file = loop_file.replace(".bed", ".mat")

        of.output_loop(loops, ref_genome_file, segment_size, output_seq_file, None)

        rs = get_prediction(loop_model_file, output_seq_file)

        output_prob(rs, output_seq_file.replace(".mat", "_no_mutation_probability.txt"))

        os.remove(output_seq_file)

        return

    jobID, inputDataFolder, outputFolder = makeOutputFolder()

    generate_patientdata(variants, loops, nbr_job, jobID)

    makePrediction(loop_model_file, inputDataFolder, outputFolder)


main()
