from django.db import models
import pandas as pd
import pathlib
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
sample = pd.read_csv(DATA_PATH.joinpath("travelverse-dataset.csv"))

tf_1=sample["'ORIGIN'"].unique()
tf_2=sample["'COUNTRY'"].unique()
tf_3=sample["'TICKETING_AIRLINE'"].unique()
tf_4=sample["'DESTINATION'"].unique()

origin_choice = list(zip(tf_1,tf_1))
country_choice = list(zip(tf_2,tf_2))
ticketing_airline_choice = list(zip(tf_3,tf_3))
destination_choice = list(zip(tf_4,tf_4))

class airlineModel(models.Model):
    origin=models.CharField(max_length=100, choices=origin_choice)
    country=models.CharField(max_length=100, choices=country_choice)
    ticketing_airline=models.CharField(max_length=100, choices=ticketing_airline_choice)
    destination=models.CharField(max_length=100, choices=destination_choice)