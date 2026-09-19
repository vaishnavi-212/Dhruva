plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.dhruva.nav"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.dhruva.nav"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // ONNX Runtime ships native code for four chip types (132 MB). Real phones from the last
        // several years are arm64-v8a (32 MB); the other three are emulators and old 32-bit phones.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.material)
    testImplementation(libs.junit)
    testImplementation("org.json:json:20240303")   // real org.json for JVM unit tests (Android's is a stub there)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    implementation("com.google.android.gms:play-services-location:21.3.0")
    implementation("org.osmdroid:osmdroid-android:6.1.18")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation(libs.onnxruntime.android)
}