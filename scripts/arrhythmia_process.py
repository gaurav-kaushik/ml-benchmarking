import argparse
import sklearn
import pandas as pd

def importDF(fname, ftype):
    if ftype == 'csv':
        df = pd.read_csv()
    elif ftype == 'tsv':
        return        

def main(args):
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--filetype", default='csv')
    parser.add_argument("-s", "--save", action="store_true", default=False)
    args = vars(parser.parse_args())

    # Run main
    main(args)