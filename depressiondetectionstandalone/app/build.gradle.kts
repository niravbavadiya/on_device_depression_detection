plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("kotlin-kapt")
}

android {
    namespace = "com.example.depressiondetectionstandalone"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.depressiondetectionstandalone"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // AWARE-style provider IDs (not strictly required but useful for schema consistency)
        resValue("string", "provider_locations", "com.aware.provider.locations")
        resValue("string", "provider_screen", "com.aware.provider.screen")
        resValue("string", "provider_steps", "com.aware.provider.steps")
        resValue("string", "provider_sleep", "com.aware.provider.sleep")
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
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // Core AndroidX
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)

    // Location & Activity APIs
    implementation("com.google.android.gms:play-services-location:21.0.1")

    // Google Fit (sleep tracking)
    implementation("com.google.android.gms:play-services-fitness:21.1.0")

    // Optional: Google Sign-In (required for Fit API auth)
    implementation("com.google.android.gms:play-services-auth:20.7.0")

    // Testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // ✅ Room components
    implementation("androidx.room:room-runtime:2.6.1")
    kapt("androidx.room:room-compiler:2.6.1")  // for annotation processing
    implementation("androidx.room:room-ktx:2.6.1") // Kotlin extensions

    // ✅ Lifecycle (for coroutines, LiveData etc. if needed)
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
}
