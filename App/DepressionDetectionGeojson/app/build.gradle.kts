plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.depressiondetectiongeojson"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.depressiondetectiongeojson"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        buildConfigField("String", "API_BASE_URL", "\"https://YOUR_API/\"")
        buildConfigField("String", "API_TOKEN", "\"YOUR_TOKEN\"")
        buildConfigField("String", "MAPS_API_KEY", "\"YOUR_GOOGLE_MAPS_API_KEY\"")
        manifestPlaceholders += mapOf(
            "MAPS_API_KEY" to "YOUR_GOOGLE_MAPS_API_KEY"
        )

    }

    buildFeatures {
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}
val tfliteVersion = "2.14.0"

dependencies {

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    // Google Maps
    implementation("org.osmdroid:osmdroid-android:6.1.18")

    // Networking
    implementation("com.squareup.retrofit2:retrofit:2.11.0")
    implementation("com.squareup.retrofit2:converter-moshi:2.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // TFLite (if not already added)
    implementation("org.tensorflow:tensorflow-lite:$tfliteVersion")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    // NEW: Flex delegate for Select TF Ops
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:$tfliteVersion")

    // Coroutines + lifecycle scope
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
}