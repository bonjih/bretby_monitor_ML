__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"


import time
import sys
import pandas as pd
from datetime import datetime

import global_conf_variables
from stream_manager import probe_stream, restart_nssm_service

values = global_conf_variables.get_values()
cams = values[0]
service_name = 'bretbyDetect'


def main():
    while True:
        try:
            df = pd.read_csv(cams)
            print('Looping through camera list......\n')

            restart_required = False  # Flag to control restart behavior

            for index, row in df.iterrows():
                print(row['cam_name'], '->', row['address'])

                is_stream_available = probe_stream(row['address'], row['cam_name'])

                if not is_stream_available:
                    print(f"Camera {row['cam_name']} not available.")
                    restart_required = True
                    break  # exit the camera loop

            if restart_required:
                print("Restarting application in 61 seconds...")
                time.sleep(61)
                restart_nssm_service(service_name)
                print("Restarting application...")
            else:
                break  # exit the main loop if all streams are available

        except Exception as e:
            if e == 'maximum recursion depth exceeded while calling a Python object':
                print(e, 'Main - ', datetime.now())
                sys.exit()


if __name__ == "__main__":
    main()
