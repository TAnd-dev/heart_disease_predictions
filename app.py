from enum import Enum

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from catboost import CatBoostClassifier

from config import Config
from data_tools import get_oldpeak_cat

app = FastAPI()


class Sex(str, Enum):
    m = 'M'
    f = 'F'


class ChestPainType(str, Enum):
    ta = 'TA'
    ata = 'ATA'
    nap = 'NAP'
    asy = 'ASY'


class ResistingECG(str, Enum):
    normal = 'Normal'
    st = 'ST'
    lvh = "LVH"


class STSlope(str, Enum):
    up = 'Up'
    flat = 'Flat'
    down = 'Down'


class SData(BaseModel):
    age: int = Query(ge=0, le=150)
    sex: Sex
    chest_pain_type: ChestPainType
    resisting_bg: int = Query(ge=0)
    cholesterol: int = Query(ge=0)
    fasting_bs: bool
    resisting_ecg: ResistingECG
    max_hr: int = Query(ge=60, le=205)
    exercise_angina: bool
    oldpeak: float
    st_slope: STSlope


def get_age_cat(age):
    if age <= 32.9:
        return '(27.951, 32.9]'
    elif age <= 37.8:
        return '(32.9, 37.8]'
    elif age <= 42.7:
        return '(37.8, 42.7]'
    elif age <= 47.6:
        return '(42.7, 47.6]'
    elif age <= 52.5:
        return '(47.6, 52.5]'
    elif age <= 57.4:
        return '(52.5, 57.4]'
    elif age <= 62.3:
        return '(57.4, 62.3]'
    elif age <= 67.2:
        return '(62.3, 67.2]'
    elif age <= 72.1:
        return '(67.2, 72.1]'
    return '(72.1, 77.0]'


def get_resting_bp_cat(resting_bp):
    if resting_bp <= 120:
        return '(-0.001, 120.0]'
    elif resting_bp <= 128:
        return '(120.0, 128.0]'
    elif resting_bp <= 135.2:
        return '(128.0, 135.2]'
    elif resting_bp <= 145:
        return '(135.2, 145.0]'
    return '(145.0, 200.0]'


def get_cholesterol_cat(cholesterol):
    if cholesterol <= 84.999:
        return '[0, 0]'
    elif cholesterol <= 217:
        return '(84.999, 217.0]'
    elif cholesterol <= 263:
        return '(217.0, 263.0]'
    return '(263.0, 603.0]'


def get_max_hr_cat(max_hr):
    if max_hr <= 103:
        return '(59.999, 103.0]'
    elif max_hr <= 115:
        return '(103.0, 115.0]'
    elif max_hr <= 122:
        return '(115.0, 122.0]'
    elif max_hr <= 130:
        return '(122.0, 130.0]'
    elif max_hr <= 138:
        return '(130.0, 138.0]'
    elif max_hr <= 144:
        return '(138.0, 144.0]'
    elif max_hr <= 151:
        return '(144.0, 151.0]'
    elif max_hr <= 160:
        return '(151.0, 160.0]'
    elif max_hr <= 170:
        return '(160.0, 170.0]'
    return '(170.0, 202.0]'


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
