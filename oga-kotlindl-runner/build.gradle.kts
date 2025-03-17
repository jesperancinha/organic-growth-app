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
    implementation(libs.kotlin.deeplearning.tensorflow)
    implementation(libs.kotlin.deeplearning.api)
    testImplementation(platform(libs.junit.jupiter.bom))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.junit.jupiter:junit-jupiter-api")
    testImplementation("org.junit.jupiter:junit-jupiter-engine")
    testImplementation("org.junit.platform:junit-platform-engine")
    testImplementation("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}

tasks.register("prepareKotlinBuildScriptModel"){}
