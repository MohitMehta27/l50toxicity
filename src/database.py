import mysql.connector
from mysql.connector import Error
import pandas as pd
from src.logger import logging
from src.exception import CustomException
#from IPython.display import display

class MySQLDataLoader:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect_to_database(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
                )
            if self.connection.is_connected():
                    self.cursor = self.connection.cursor()
                    logging.info("Connected to MySQL database")
        except Error as e:
            raise CustomException(f"Error while connecting to MySQL: {e}")


    def fetch_data(self, query):
        try:
            self.cursor.execute(query)
            records = self.cursor.fetchall()
            logging.info(f"Total rows: {self.cursor.rowcount}")
            return records
        except Error as e:
            raise CustomException(f"Error fetching data from MySQL table: {e}")

    def disconnect_from_database(self):
        if self.connection.is_connected():
            self.cursor.close()
            self.connection.close()
            logging.info("MySQL connection is closed")

    def load_data_to_csv(self, records, file_path):
        column_names = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(records, columns=column_names)
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")

if __name__ == "__main__":
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "Mohit@7117",
        "database": "model",
    }

    query = "SELECT * FROM model.qsar_fish_toxicity"
    file_path = "qsar.csv"

    try:
        data_loader = MySQLDataLoader(**db_config)
        data_loader.connect_to_database()
        records = data_loader.fetch_data(query)
        data_loader.load_data_to_csv(records, file_path)
    except CustomException as ce:
        logging.error(str(ce))
    finally:
        if data_loader:
            data_loader.disconnect_from_database()
