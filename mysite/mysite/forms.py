from django import forms  
from django.forms import ModelForm
import pandas as pd
import pathlib
from mysite.models import origin_choice , country_choice , ticketing_airline_choice , destination_choice



class airlineForm(forms.Form):
    origin=forms.ChoiceField(choices=origin_choice)
    country=forms.ChoiceField(choices=country_choice)
    ticketing_airline=forms.ChoiceField(choices=ticketing_airline_choice)
    destination=forms.ChoiceField(choices=destination_choice)

class twitterForm(forms.Form):
    tweet=forms.CharField(max_length=1000)   