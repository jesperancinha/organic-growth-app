plugins {
    alias(libs.plugins.kotlin.jvm)
}

group = "org.jesperancinha.questions"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(libs.kotlin.deeplearning.api)
    implementation(libs.kotlin.deeplearning.impl)
    implementation(libs.kotlin.deeplearning.onnx)
    implementation(libs.kotlin.deeplearning.dataset)
    implementation(libs.kotlin.deeplearning.tensorflow)
    testImplementation(libs.junit.jupiter.api)
    testImplementation(libs.junit.jupiter.engine)
}

tasks.test {
    useJUnitPlatform()
}

tasks.register("prepareKotlinBuildScriptModel"){}
