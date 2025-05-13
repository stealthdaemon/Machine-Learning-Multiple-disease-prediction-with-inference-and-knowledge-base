from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import pickle
import csv


# Load data and model files once during server initialization
DF_FILE = 'data/dummy_df.csv'
MODEL_FILE = 'data/model.pkl'
LABEL_ENCODER_FILE = 'data/label_encoder.pkl'
DESCRIPTION_FILE = 'data/symptom_Description.csv'
PRECAUTION_FILE = 'data/symptom_precaution.csv'

df = pd.read_csv(DF_FILE)
symptoms = list(df.columns)

with open(MODEL_FILE, 'rb') as model_file:
    model = pickle.load(model_file)

with open(LABEL_ENCODER_FILE, 'rb') as le_file:
    label_encoder = pickle.load(le_file)


def format_symptoms(symptoms):
    """Format symptom names for rendering."""
    return [symptom.strip().replace('_', ' ').capitalize() for symptom in symptoms]


def index(request):
    """Render the index page with formatted symptoms."""
    formatted_symptoms = format_symptoms(symptoms)
    return render(request, 'index.html', {'symptoms': formatted_symptoms})


def parse_symptoms(input_symptoms):
    """Parse and match input symptoms with the known symptoms."""
    input_symptoms_list = [symptom.strip().lower() for symptom in input_symptoms.split(',')]
    matched_symptoms = {}

    for input_symptom in input_symptoms_list:
        for symptom in symptoms:
            pattern = symptom.replace('_', ' ')
            if pattern == input_symptom:
                matched_symptoms[symptom] = 1

    return matched_symptoms


def get_disease_description(disease_name):
    """get the description of the diagnosed disease."""
    with open(DESCRIPTION_FILE, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['Disease'].strip() == disease_name:
                return row['Description'].strip()
    return None


def get_disease_precautions(disease_name):
    """get the precautions for the diagnosed disease."""
    precautions = []
    with open(PRECAUTION_FILE, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['Disease'].strip() == disease_name:
                for i in range(1, 5):  # Precaution_1 to Precaution_4
                    precaution = row.get(f'Precaution_{i}', '').strip().capitalize()
                    if precaution:
                        precautions.append(precaution)
                break
    return precautions


def diagnose(request):
    """perform diagnosis based on user symptoms."""
    data = {}
    dummy_df = df.copy()

    if request.method == 'POST':
        input_symptoms = request.POST.get('symptoms', '')
        symptoms_dict = parse_symptoms(input_symptoms)

        if len(symptoms_dict) < 3:
            data['Error'] = 'Some symptoms you entered are not valid. Enter at least 3 symptoms, select from the suggestions below.'
        else:
            # Update the dataframe with the matched symptoms
            for key, value in symptoms_dict.items():
                dummy_df.loc[0, key] = value

            # Predict the disease from the model retrieved after unpickling
            prediction = model.predict(dummy_df)
            disease = label_encoder.inverse_transform(prediction)[0]

            # Fetch disease details
            data['Success'] = True
            data['Disease'] = disease
            data['Description'] = get_disease_description(disease)
            data['Precautions'] = get_disease_precautions(disease)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse(data)

    return index(request)
