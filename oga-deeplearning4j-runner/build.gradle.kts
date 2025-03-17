plugins {
    id("java")
}

group = "org.jesperancinha.oga"
version = "0.0.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-M2.1")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
    implementation("org.datavec:datavec-api:1.0.0-M2.1")
    testImplementation(platform(libs.junit.jupiter.bom))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.junit.platform:junit-platform-engine")
    testImplementation("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
}