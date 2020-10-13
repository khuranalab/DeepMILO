import mutation_function
from constant import *

import re
import numpy as np
import sys

from Bio import SeqIO
from Bio.Seq import MutableSeq
import h5py


def line_to_tensor(line, all_letters='ACGTN'):
    '''
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    '''

    # ignore this letter if it is not ACGTN
    line = re.sub('[^ACGTN]', '', line.upper())

    if re.search('[^ACGTN]', line.upper()):
        sys.stderr.write('Unknown char in sequence: {}'.format(str(re.search('[^ACGTN]', line.upper()))))
        return

    n = len(all_letters)

    tensor = np.zeros((len(line), n), dtype=int)
    for li, letter in enumerate(line):
        tensor[li][LETTERTOINDEX[letter]] = 1

    return tensor

def test_line_to_tensor():

    line = 'AGCTN'
    ts = line_to_tensor(line)

    ex = np.zeros((len(line), 5), dtype=int)
    ex[0][0] = 1
    ex[2][1] = 1
    ex[1][2] = 1
    ex[3][3] = 1
    ex[4][4] = 1
    assert np.array_equal(ts, ex)





def output_seq(hdf5_file, name, seq):
    """Output seq1 and seq2 to hdf5 file"""

    seq = line_to_tensor(str(seq).upper())

    hdf5_file.create_dataset(name, data=seq,
                             compression="gzip", compression_opts=5)


def get_chromlist(ref_genome_file):
    """Get chromosomes in SeqIO record"""
    # chrom list
    chrom_list = {}

    for seq_record in SeqIO.parse(ref_genome_file, "fasta"):
        # only consider chromosome 1,2,... and X and Y
        if seq_record.name.startswith("GL") or seq_record.name.startswith("MT"):
            continue

        chrom = "chr" + seq_record.name
        chrom_list[chrom] = seq_record

    return chrom_list


def output_boundary(boundaries, ref_genome_file, segment_size, data_file, label_file, isreverse=True):
    '''
    output boundary data to files
    isreverse: output sequence of the reverse strand as well

    '''
    chrom_list = get_chromlist(ref_genome_file)

    hdf5_file = None
    hdf5_label_file = None

    try:
        hdf5_file = h5py.File(data_file, mode='w')

        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')

        for b in boundaries:
            # print(b.chrom, b.start, b.end)

            #print('start: {}, end: {}'.format(b.start, b.end))

            seq = mutation_function.infer_true_seq(b, chrom_list[b.chrom], segment_size)

            if len(seq) != segment_size:
                raise Exception('length of sequences is wrong: %d' % (len(seq)))

            output_seq(hdf5_file, b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_" + b.suffix + "_1", seq)

            # complement sequences
            if isreverse:
                if type(seq) is MutableSeq:
                    seq2 = seq.toseq().reverse_complement()
                else:
                    # print(type(seq1))
                    seq2 = seq2.reverse_complement()

                output_seq(hdf5_file, b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_" + b.suffix + "_2", seq2)

            if hdf5_label_file:
                hdf5_label_file.create_dataset(b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_" + b.suffix + "_1", data=b.label)

                if isreverse:
                    hdf5_label_file.create_dataset(b.chrom + "_" + str(b.start) + "_" + str(b.end) + "_" + b.suffix + "_2", data=b.label)

    finally:
        if hdf5_file is not None:
            hdf5_file.close()

        if hdf5_label_file is not None:
            hdf5_label_file.close()

def output_boundary_npa(boundaries, ref_genome_file, segment_size, data_file, label_file, n_features=5, isreverse=True):
    '''
    output boundary data to files
    isreverse: output sequence of the reverse strand as well

    '''
    chrom_list = get_chromlist(ref_genome_file)

    hdf5_file = None
    hdf5_label_file = None

    try:
        hdf5_file = h5py.File(data_file, mode='w')
        dt = hdf5_file.create_dataset('data', (len(boundaries) * 2, segment_size, n_features), compression="gzip", compression_opts=5)

        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')
            label = hdf5_label_file.create_dataset('label', (len(boundaries) * 2))

        k = 0
        for b in boundaries:
            # print(b.chrom, b.start, b.end)

            #print('start: {}, end: {}'.format(b.start, b.end))

            seq = mutation_function.infer_true_seq(b, chrom_list[b.chrom], segment_size)

            if len(seq) != segment_size:
                raise Exception('length of sequences is wrong: %d' % (len(seq)))

            seq = line_to_tensor(str(seq).upper())
            dt[k,] = seq
            if hdf5_label_file:
                label[k] = b.label

            k += 1

            # complement sequences
            if isreverse:
                if type(seq) is MutableSeq:
                    seq2 = seq.toseq().reverse_complement()
                else:
                    # print(type(seq1))
                    seq2 = seq2.reverse_complement()

                dt[k,] = seq2

                if hdf5_label_file:
                    label[k] = b.label

                k += 1

    finally:
        if hdf5_file is not None:
            hdf5_file.close()

        if hdf5_label_file is not None:
            hdf5_label_file.close()


def output_loop(data, ref_genome_file, segment_size, data_file, label_file):
    """Output loop data (with 2 boundaries) to files
    data_file: contain IDs of 2 boundaries of loops, boundaries refer
    to common_data_file for sequence data

    lable_file: 0: no loop, 1: loop
    """

    chrom_list = get_chromlist(ref_genome_file)

    hdf5_file = None
    hdf5_label_file = None

    try:
        hdf5_file = h5py.File(data_file, mode='w')
        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')

        for loop in data:
            # print(b.chrom, b.start, b.end)
            b1, b2 = loop.b1, loop.b2

            seq1 = mutation_function.infer_true_seq(b1, chrom_list[b1.chrom], segment_size)
            seq2 = mutation_function.infer_true_seq(b2, chrom_list[b2.chrom], segment_size)

            if len(seq1) != segment_size or len(seq2) != segment_size:
                raise Exception(
                    'length of sequences is wrong: %d, %d' % (len(seq1), len(seq2)))

            seq1_code = line_to_tensor(str(seq1).upper())
            seq2_code = line_to_tensor(str(seq2).upper())

            seq_forward = np.hstack((seq1_code, seq2_code))

            hdf5_file.create_dataset(
                b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_1",
                data=seq_forward, compression="gzip", compression_opts=5)

            # complement sequences

            seq5 = seq1.toseq().reverse_complement() if type(seq1) is MutableSeq else seq1.reverse_complement()
            seq6 = seq2.toseq().reverse_complement() if type(seq2) is MutableSeq else seq2.reverse_complement()

            if seq5 is None or seq6 is None:
                raise Exception('type sequences is wrong {}'.format(type(seq1)))

            if len(seq5) != segment_size or len(seq6) != segment_size:
                raise Exception(
                    'length of sequences is wrong: %d, %d' % (len(seq5), len(seq6)))

            seq5_code = line_to_tensor(str(seq5).upper())
            seq6_code = line_to_tensor(str(seq6).upper())

            seq_reverse = np.hstack((seq6_code, seq5_code))

            # b.seq = seq
            hdf5_file.create_dataset(
                b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_2",
                data=seq_reverse, compression="gzip", compression_opts=5)


            if hdf5_label_file:
                hdf5_label_file.create_dataset(
                    b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_1",
                    data=loop.label)
                hdf5_label_file.create_dataset(
                    b1.chrom + "_" + str(b1.start) + "_" + str(b1.end) + "_" + str(b2.start) + "_" + str(b2.end) + "_2",
                    data=loop.label)

    finally:
        if hdf5_file is not None:
            hdf5_file.close()

        if hdf5_label_file is not None:
            hdf5_label_file.close()


def output_loop_npa(data, ref_genome_file, segment_size, data_file, label_file, n_feature=5):
    '''
    output loop data (with 2 boundaries) to files
    data_file: contain IDs of 2 boundaries of loops, boundaries refer to common_data_file for sequence data
    lable_file: 0: no loop, 1: loop

    '''

    chrom_list = get_chromlist(ref_genome_file)


    lb = None
    hdf5_file = None
    hdf5_label_file = None

    try:
        hdf5_file = h5py.File(data_file, mode='w')

        #                                       loop, boundary1 or 2, 4000 x 5
        dt = hdf5_file.create_dataset('data', (len(data) * 2, 2, segment_size, n_feature), dtype='b1', compression="gzip", compression_opts=5)

        if label_file:
            hdf5_label_file = h5py.File(label_file, mode='w')
            lb = hdf5_label_file.create_dataset('label', (len(data) * 2), dtype='b1', compression="gzip", compression_opts=5)

        # len(data) * 2 loops -- including reverse, each has 2 boundaries, each boundary is of size: segment_size * n_feature
        #dt = np.zeros((len(data) * 2, segment_size * n_feature * 2))
        #lb = np.zeros((len(data) * 2))

        k = 0
        for loop in data:
            # print(b.chrom, b.start, b.end)
            b1, b2 = loop.b1, loop.b2

            seq1 = mutation_function.infer_true_seq(b1, chrom_list[b1.chrom], segment_size)
            seq2 = mutation_function.infer_true_seq(b2, chrom_list[b2.chrom], segment_size)

            if len(seq1) != segment_size or len(seq2) != segment_size:
                raise Exception(
                    'length of sequences is wrong: %d, %d' % (len(seq1), len(seq2)))

            seq1_code = line_to_tensor(str(seq1).upper())
            seq2_code = line_to_tensor(str(seq2).upper())

            dt[k,0,] = seq1_code
            dt[k,1,] = seq2_code
            if lb:
                lb[k] = loop.label
            k += 1

            # complement sequences

            seq5 = seq1.toseq().reverse_complement() if type(seq1) is MutableSeq else seq1.reverse_complement()
            seq6 = seq2.toseq().reverse_complement() if type(seq2) is MutableSeq else seq2.reverse_complement()

            if seq5 is None or seq6 is None:
                raise Exception('type sequences is wrong {}'.format(type(seq1)))

            if len(seq5) != segment_size or len(seq6) != segment_size:
                raise Exception(
                    'length of sequences is wrong: %d, %d' % (len(seq5), len(seq6)))

            seq5_code = line_to_tensor(str(seq5).upper())
            seq6_code = line_to_tensor(str(seq6).upper())

            dt[k,0,] = seq6_code
            dt[k,1,] = seq5_code
            if lb:
                lb[k] = loop.label
            k += 1

    finally:
        if hdf5_file:
            hdf5_file.close()

        if hdf5_label_file:
            hdf5_label_file.close()

