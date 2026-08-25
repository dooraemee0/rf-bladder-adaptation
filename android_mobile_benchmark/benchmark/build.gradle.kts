plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("androidx.benchmark")
}

android {
    namespace = "org.dgist.bladderbenchmark.microbenchmark"
    compileSdk = 36
    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.benchmark.junit4.AndroidBenchmarkRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    testBuildType = "release"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    androidTestImplementation(project(":pipeline"))
    androidTestImplementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.benchmark:benchmark-junit4:1.4.1")
}
