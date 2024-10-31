plugins {
    alias(libs.plugins.kotlin.jvm)
}

group = "org.jesperancinha.oga"
version = "0.0.0"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(libs.junit.jupiter.api)
    testImplementation(libs.junit.jupiter.engine)
    implementation(libs.langchain4j.open.ai)
    implementation(libs.langchain4j)
}

tasks.test {
    useJUnitPlatform()
}