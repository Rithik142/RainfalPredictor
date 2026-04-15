import pandas as pd
from rainfall_system.features import make_feature_frame
from rainfall_system.models import RainfallModelService

model_service = RainfallModelService()

data = {
    "time": ["2026-04-15T00:00:00Z"],
    "lat": [59.3293],
    "lon": [18.0686],
    "precipitation": [0.0],
    "relative_humidity_2m": [80.0],
    "dew_point_2m": [5.0],
    "pressure_msl": [1015.0],
    "temperature_2m": [10.0],
    "wind_speed_10m": [5.0],
}

df = pd.DataFrame(data)
features = make_feature_frame(df)
result = model_service.predict_from_features(features)

print("Prediction:")
print(result)