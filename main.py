__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import sys
import pandas as pd
from datetime import datetime

import global_conf_variables
from stream_manager import probe_stream

values = global_conf_variables.get_values()

cams = values[0]


def main():
    while True:
        try:
            df = pd.read_csv(cams)

            print('Looping through camera list......\n')

            for index, row in df.iterrows():
                print(row['cam_name'], '->', row['address'])

                # to check if stream exists
                probe_stream(row['address'], row['cam_name'])

        except Exception as e:
            if e == 'maximum recursion depth exceeded while calling a Python object':
                print(e, 'Main - ', datetime.now())
                sys.exit()


if __name__ == "__main__":
    main()
