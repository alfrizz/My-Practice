# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(args):
    # read data
    df = get_data(args.input_data)

    output_df = df.to_csv((Path(args.output_datastore) / "diabetes.csv"), index = False)

# function that reads the data
def get_data(path):
    df = pd.read_csv(path)

    # Count the rows and print the result
    row_count = (len(df))
    print('Analyzing {} rows of data'.format(row_count))
    
    return df

# The parse_args function is used in scripts that need to handle input and output paths dynamically, such as when running as part of a larger job in environments like Azure Machine Learning. This function allows the script to accept these paths as command-line arguments, making it flexible and adaptable to different data sources and destinations
def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    # The main function reads the data from args.input_data and writes the processed data to args.output_datastore (provided in the job script below)
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_datastore", dest='output_datastore',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
