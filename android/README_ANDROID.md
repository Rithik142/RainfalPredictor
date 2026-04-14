# Android Studio Integration Guide (Rainfall API)

This guide shows how to connect your Android app to the backend endpoint:

- `POST /predict_rainfall`

## 1) Recommended architecture

- Backend (this repo): FastAPI service (host it locally/cloud)
- Android app: Kotlin + Retrofit + Coroutines
- Data flow: App sends location/date/horizon -> backend returns rainfall probabilities + quantiles + risk

## 2) Add Android dependencies

In `app/build.gradle`:

```gradle
dependencies {
    implementation "com.squareup.retrofit2:retrofit:2.11.0"
    implementation "com.squareup.retrofit2:converter-gson:2.11.0"
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1"
}
```

## 3) Kotlin models

```kotlin
data class RainfallPredictRequest(
    val region: String,
    val lat: Double,
    val lon: Double,
    val horizon_hours: Int,
    val start_date: String
)

data class RainfallPredictResponse(
    val prediction_id: Int,
    val rain_probability_24h: Double,
    val rainfall_mm_p10_24h: Double,
    val rainfall_mm_p50_24h: Double,
    val rainfall_mm_p90_24h: Double,
    val extreme_rain_risk: Double,
    val extreme_threshold_mm: Double,
    val model_version: String
)
```

## 4) Retrofit service

```kotlin
import retrofit2.http.Body
import retrofit2.http.POST

interface RainfallApi {
    @POST("predict_rainfall")
    suspend fun predictRainfall(@Body request: RainfallPredictRequest): RainfallPredictResponse
}
```

## 5) Retrofit client

```kotlin
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    // Android emulator to local machine backend:
    // 10.0.2.2 maps to localhost on your computer.
    private const val BASE_URL = "http://10.0.2.2:8000/"

    val rainfallApi: RainfallApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(RainfallApi::class.java)
    }
}
```

## 6) Call from ViewModel

```kotlin
viewModelScope.launch {
    try {
        val response = ApiClient.rainfallApi.predictRainfall(
            RainfallPredictRequest(
                region = "Stockholm",
                lat = 59.3293,
                lon = 18.0686,
                horizon_hours = 72,
                start_date = "2026-04-15"
            )
        )
        // Use response in UI
    } catch (e: Exception) {
        // Handle network or parsing error
    }
}
```

## 7) Android manifest permission

In `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

## 8) Local testing checklist

1. Start backend on computer: `uvicorn rainfall_system.api:app --reload`
2. Ensure Android emulator uses `http://10.0.2.2:8000/`
3. Hit endpoint from app
4. Confirm prediction appears in app and record stored in backend DB

## 9) Production setup

- Deploy backend to cloud (Render/Railway/Fly.io/AWS)
- Replace `BASE_URL` with HTTPS domain
- Add auth token headers and CORS restrictions
- Use PostgreSQL in production via `DATABASE_URL`
