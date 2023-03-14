__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import pandas as pd

import global_conf_variables
from stream_manager import probe_stream

values = global_conf_variables.get_values()

cams = values[0]


def main():
    while True:
        # try:
        df = pd.read_csv(cams)

        print('Looping through camera list......\n')

        for index, row in df.iterrows():
            print(row['cam_name'], '->', row['address'])

            # to check if stream exists
            probe_stream(row['address'], row['cam_name'])

        # except Exception as e:
        #     print(e)


if __name__ == "__main__":
    main()
