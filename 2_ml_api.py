from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# pydantic to pass the information
class scoring_item(BaseModel):
    YearsAtCompany: float #1,
    EmployeeSatisfaction: float # 0.01,
    Position: str # "Non-Manager",
    Salary: int # 4
    
model = joblib.load("rfmodel.pkl")
le = joblib.load("le.pkl")


# @app.post('/')
# async def scoring_endpoint(item: scoring_item):
#     df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
#     y_pred = model.predict(df)
#     return {"prediction":int(y_pred)}

@app.post('/')
async def scoring_endpoint(item: scoring_item):
    data_dict = item.dict()
    # Transform the Position value using the loaded LabelEncoder
    try:
        data_dict['Position'] = int(le.transform([data_dict['Position']])[0])
    except ValueError:
        return {"error": f"Unrecognized Position value: {data_dict['Position']}"}

    df = pd.DataFrame([data_dict])
    y_pred = model.predict(df)
    return {"prediction": int(y_pred[0])}