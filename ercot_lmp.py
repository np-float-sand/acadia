import requests
import pandas as pd

class ERCOTLMP:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.base_url = "http://www.ercot.com/content/faq/lmp/"

    def fetch_data(self):
        # Constructing the API endpoint based on provided dates
        endpoint = f"{self.base_url}?start_date={self.start_date}&end_date={self.end_date}"
        response = requests.get(endpoint)

        if response.status_code == 200:
            # Assuming the response content is in CSV format
            data = pd.read_csv(response.content)
            return data
        else:
            raise Exception(f"Error fetching data: {response.status_code}")

if __name__ == '__main__':
    ercot_lmp = ERCOTLMP(start_date='2022-01-01', end_date='2022-01-31')
    try:
        lmp_data = ercot_lmp.fetch_data()
        print(lmp_data)
    except Exception as e:
        print(e)
