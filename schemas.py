from enum import Enum

from fastapi import Query
from pydantic import BaseModel


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
