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
    implementation("dev.langchain4j:langchain4j-open-ai")
    implementation("dev.langchain4j:langchain4j")
    implementation(platform(libs.langchain4j.bom))
}

tasks.test {
    useJUnitPlatform()
}