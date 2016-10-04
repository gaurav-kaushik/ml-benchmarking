"""
usage: munger_v2.py

optional arguments:
  -h, --help            show this help message and exit
  -f FILES [FILES ...], --files FILES [FILES ...]
                        TCGA Gene Expression TXT files
  -c, --csv
  -t, --transpose
  -o OUTPUT_FILENAME, --output_filename OUTPUT_FILENAME
  -r FILE_INDEX, --file_index FILE_INDEX
"""
from __future__ import print_function
import pandas as pd
import argparse
import sys

# TODO: remove row with "Name" and "TPM" are the values -- df1 = df1[:-1]
# TODO: first column should be the index --
# TODO: change merger criteria on the index column


def get_dataframe_list(files, file_index, data_type):
    """ get a list of dataframes from -f or -r """

    # -f: if you passed files to munger.py via -f, return files or return empty array
    files = files or []

    # -r: if you have an index as the file pointer, use this to get the files
    if file_index:
        with open(file_index) as fp:
            files.extend(fp.readlines())
    sorted_files = sorted(filter(None, set([f.strip() for f in files])))

    # now iterate over the files and get the list of dataframes
    dfs = []
    for f in sorted_files:
        try:
            if data_type == 'gene':
                dfs.append(pd.read_table(f, usecols=('gene', 'raw_counts'))) # Get only specific columns you want with 'usecols'
            elif data_type == 'transcript':
                # get the TCGA code from filename (e.g. from TCGA-OR-A5LA-01A_quant.sf)
                TCGA_code = f.split('/').pop().split('_')[0]
                dfs.append(pd.read_table(f, delim_whitespace=True, header=None, index_col=None, names=['transcript', TCGA_code])[:-1])
        except:
            continue
    return dfs, sorted_files


def get_metadata_tag(filename):
    """ Gets a filename (without extension) from a provided path """
    # UNCID = filename.split('/')[-1].split('.')[0]
    TCGA = filename.split('/')[-1].split('.')[1] # IMPORTANT! this will depend heavily on the file type, might make sense to invoke for GeneExpQuant only or make true parser based on data type
    return TCGA


def merge_texts(files, file_index, data_type):
    """ merge the dataframes in your list """
    dfs, filenames = get_dataframe_list(files, file_index, data_type)
    # enumerate over the list, merge, and rename columns
    try:
        df = dfs[0]

        print(*[df_.columns for df_ in dfs],sep='\n')
        for i, frame in enumerate(dfs[1:]):
            if data_type == 'gene':
                try:
                    # rename first columns to metadata value
                    df = df.rename(columns={'raw_counts': get_metadata_tag(filenames[0])})
                    df = df.merge(frame, on='gene').rename(columns={'raw_counts':'raw_counts_' + get_metadata_tag(filenames[i-1])})
                except:
                    continue
            elif data_type == 'transcript':
                try:
                    df = df.merge(frame, on='transcript')
                    # df = df.merge(frame, on=frame.index)
                except:
                    continue
        return df
    except:
        print("Could not merge dataframe")

def save_csv(df, csv, output_filename, filename, header_opt=False, index_opt=False):
    """ if csv is true and an output filename is given, rename """
    if csv and output_filename:
        return df.to_csv(path_or_buf=filename, header=header_opt, index=index_opt)

def main(files, csv, transpose, output_filename, file_index, data_type):
    """ main: get a matrix/df of RNAseq counts in TPM """

    df = merge_texts(files, file_index, data_type)

    # Save to disk
    save_csv(df, csv, output_filename, filename=str(output_filename) + '_by_{}.csv'.format(data_type), header_opt=True)

    # Save Transpose to disk
    if transpose:
        df_transpose = df.transpose()
        df_transpose = df_transpose.rename(index = {str(data_type):'case'})
        save_csv(df_transpose, csv, output_filename,
                                    filename=str(output_filename) + '_by_case.csv',
                                    header_opt=False, index_opt=True)

    try:
        print(df.head())
    except:
        print("No dataframe merged")
    return df

if __name__ == "__main__":
    # Parse your args
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", help="TCGA Gene Expression TXT files", nargs="+")
    parser.add_argument("-c", "--csv", action="store_true", default=False)
    parser.add_argument("-t", "--transpose", action="store_true", default=False)
    parser.add_argument("-o", "--output_filename", type=str, default="GEX_dataframe")
    parser.add_argument("-r", "--file_index", type=str, default=None)
    parser.add_argument("-d", "--data_type", type=str, default="gene")
    args = vars(parser.parse_args())
    files = args['files']
    csv = args['csv']
    transpose = args['transpose']
    output_filename = args['output_filename']
    file_index = args['file_index']
    data_type = args['data_type']

    # Check to make sure you have your files or indices (XOR)
    if not files and not file_index:
        parser.print_help()
        sys.exit(0)

    df = main(files, csv, transpose, output_filename, file_index, data_type)