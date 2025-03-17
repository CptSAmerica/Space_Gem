import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse


def retreive_data() -> pd.DataFrame:
    '''
    - Look for the data locally.
    - If exists, create 2 dfs: train and test
    - If not, query the database on BigQuery, download it locally and create the dfs
    '''

    print(Fore.MAGENTA + "\n ⭐️ Use case: create dfs" + Style.RESET_ALL)
