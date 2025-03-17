plugins {
    alias(libs.plugins.kotlin.jvm)
}

group = "org.jesperancinha.oga"
version = "0.0.0"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform(libs.junit.jupiter.bom))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testImplementation("org.junit.jupiter:junit-jupiter-api")
    testImplementation("org.junit.jupiter:junit-jupiter-engine")
    testImplementation("org.junit.platform:junit-platform-engine")
    testImplementation("org.junit.platform:junit-platform-launcher")
    implementation(libs.langchain4j.open.ai)
    implementation(libs.langchain4j)
}

tasks.test {
    useJUnitPlatform()
}