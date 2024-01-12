import pandas as pd
import os


class CSVValidationError(Exception):
    """Custom exception for CSV validation errors."""
    pass


def validate_csv(filename):
    # Check if the filename ends with '.csv'
    if not filename.endswith('.csv'):
        raise CSVValidationError("Inputted csv at {0} does not end with .csv".format(filename))

    # Check if the file exists
    if not os.path.isfile(filename):
        raise CSVValidationError("Inputed csv at {0} does not exist".format(filename))

    # Read and check for 'images' column
    try:
        df = pd.read_csv(filename)
        if 'Path' not in df.columns:
            raise CSVValidationError("{0} does not contain 'Path' column".format(filename))
        if not os.path.isfile(df['Path'][0]):
            raise CSVValidationError("first element {0} does not exist, check that filenames have the correct path in your input csv".format(
                df['Path'][0]
            ))
    except Exception as e:
        raise CSVValidationError(f"Error reading CSV file: {e}")

    print("{} is valid".format(filename))
    
