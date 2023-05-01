__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import time

import pandas as pd
import pyodbc
from sqlalchemy import create_engine
from datetime import datetime

import config_parser
import global_conf_variables

values = global_conf_variables.get_values()

db_user = values[5]
db_pw = values[6]
db_host = values[7]
db_table = values[8]
db_driver = values[9]
db_server = values[10]


def check_if_df_empty(df):
    return len(df.index) == 0


def db_manager_controller():
    data = pd.read_csv('temp_out.csv')

    data['df_truth'] = pd.notnull(data['Image'])
    data = data[data['df_truth'] == True]

    db_fields = config_parser.db_parser()
    result = check_if_df_empty(data)

    if not result:
        if 'df_truth' in data:
            data.drop('df_truth', axis=1, inplace=True)
            sql = SQL(db_user, db_pw, db_host, db_table, db_driver, db_server)
            sql.insert(data, db_fields)
    else:
        pass


class SQL:
    def __init__(self, user, pwd, host, db, driver, server):
        self.user = user
        self.pwd = pwd
        self.host = host
        self.db = db
        self.driver = driver
        self.server = server
        self.engine = create_engine(f"mssql+pyodbc://{user}:{pwd}@{server}/{db}?driver={driver}",
                                    use_setinputsizes=False)
        self.conn = pyodbc.connect(user=user, password=pwd, host=host, database=db, driver=driver, server=server)

    def insert(self, data, db_fields):
        cur = self.conn.cursor()

        cur.execute(
            "SELECT IIF(Image IS NOT NULL, 'TRUE', 'FALSE' ) AS is_not_null FROM BretbyDetect")
        result_img = cur.fetchall()

        if not result_img or result_img[0][0] == 'FALSE':
            data.insert(loc=0, column='DateTime', value=datetime.now())
            data.columns = db_fields
            data = data.iloc[:1]
            data.to_sql('BretbyDetect', con=self.engine, if_exists='append', index=False)
