import pandas as pd
from fastapi import FastAPI

from catboost import CatBoostClassifier

from config import Config
from data_tools import get_oldpeak_cat, get_age_cat, get_resting_bp_cat, get_cholesterol_cat, get_max_hr_cat
from schemas import SData

app = FastAPI()


def predict(data: SData):
    df = pd.DataFrame(
        {
            'c_age': get_age_cat(data.age),
            'c_chest_pain': data.chest_pain_type,
            'c_resting_bp': get_resting_bp_cat(data.resisting_bg),
            'c_cholesterol': get_cholesterol_cat(data.cholesterol),
            'c_resting_ecg': data.resisting_ecg,
            'c_max_hr': get_max_hr_cat(data.max_hr),
            'c_exercise_angina': 'Y' if data.exercise_angina else 'N',
            'c_oldpeak': get_oldpeak_cat(data.oldpeak),
            'c_st_slope': data.st_slope,
            'b_sex': data.sex,
            'b_fasting_bs': '1' if data.fasting_bs else '0'
        },
        index=[0]
    )

    model = CatBoostClassifier()
    model.load_model(Config.MODEL_PATH, format='json')
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0][1]

    return prediction, round(prediction_proba, 4) * 100


@app.post('/')
async def get_prediction(data: SData):
    prediction, likehood = predict(data)
    return {
        'prediction': int(prediction),
        'likehood': likehood

    }
