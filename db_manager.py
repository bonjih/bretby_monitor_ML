__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import pyodbc
from sqlalchemy import create_engine
import pandas as pd


class SQL:
    def __init__(self, user, pwd, host, db, driver, server):
        self.user = user
        self.pwd = pwd
        self.host = host
        self.db = db
        self.driver = driver
        self.server = server
        self.engine = create_engine(f"mssql+pyodbc://{user}:{pwd}@{server}/{db}?driver={driver}")
        self.conn = pyodbc.connect(user=user, password=pwd, host=host, database=db, driver=driver, server=server)

    def insert(self, data, db_fields):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO BretbyDetect (DateTime) VALUES (CURRENT_TIMESTAMP)", )
        data.columns = db_fields
        print(data,  'fff')

        data.to_sql('BretbyDetect', con=self.engine, if_exists='append', index=False)
