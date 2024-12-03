from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

# Инициализация FastAPI приложения
app = FastAPI()

# Загрузка сохранённых данных: модели, скейлера и признаков
with open("lasso_model_data.pickle", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]  
scaler = model_data["scaler"]  
FEATURES = model_data["features"]  

# Маппинг входных признаков на признаки модели
FEATURES_MAPPING = {
    "year": "year",
    "km_driven": "km_driven",
    "mileage": "mileage (kmpl)",
    "engine": "engine (CC)",
    "max_power": "max_power (bhp)",
    "seats": "seats"
}

# Функция для преобразования входных данных
def map_features(input_data):
    return {FEATURES_MAPPING[key]: value for key, value in input_data.items() if key in FEATURES_MAPPING}

# Класс для описания одного объекта для предсказания
class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    seats: float

# Класс для описания коллекции объектов
class Items(BaseModel):
    objects: List[Item]

# Эндпоинт для предсказания стоимости на основе данных одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> dict:
    try:
        # Преобразование объекта в DataFrame
        input_data = pd.DataFrame([map_features(item.dict())])

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(input_data.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Пропущены обязательные признаки. Ожидаются: {FEATURES}"
            )

        # Стандартизация данных
        input_scaled = scaler.transform(input_data[FEATURES])

        # Предсказание стоимости
        prediction = model.predict(input_scaled)[0]
        return {"predicted_price": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для предсказания стоимости для коллекции объектов
@app.post("/predict_items")
def predict_items(items: Items) -> dict:
    try:
        # Преобразование коллекции объектов в DataFrame
        input_data = pd.DataFrame([map_features(item.dict()) for item in items.objects])

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(input_data.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Пропущены обязательные признаки. Ожидаются: {FEATURES}"
            )

        # Стандартизация данных
        input_scaled = scaler.transform(input_data[FEATURES])

        # Предсказание стоимости
        predictions = model.predict(input_scaled)
        return {"predicted_prices": [round(float(pred), 2) for pred in predictions]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для обработки файла CSV
@app.post("/predict_csv")
def predict_csv(file: UploadFile) -> FileResponse:
    try:
        # Чтение данных из CSV-файла
        df = pd.read_csv(file.file)

        # Преобразование названий колонок
        df = df.rename(columns=FEATURES_MAPPING)

        # Проверка наличия всех необходимых признаков
        if not set(FEATURES).issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"Некорректный файл. Ожидаются признаки: {FEATURES}"
            )

        # Стандартизация данных
        input_scaled = scaler.transform(df[FEATURES])

        # Предсказание стоимости
        predictions = model.predict(input_scaled)

        # Добавление предсказаний в DataFrame
        df["predicted_price"] = [round(float(pred), 2) for pred in predictions]

        # Сохранение результата в новый файл
        output_file = "predictions.csv"
        df.to_csv(output_file, index=False)

        # Возвращение файла пользователю
        return FileResponse(output_file, media_type='text/csv', filename="predictions.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))