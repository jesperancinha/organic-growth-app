plugins {
    id("org.springframework.boot") version "3.3.0"
    id("io.spring.dependency-management") version "1.1.0"
    kotlin("jvm") version "2.0.20"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_17

repositories {
    mavenCentral()
    maven("https://repo.spring.io/milestone")
    maven("https://repo.spring.io/snapshot")
}

dependencies {
    implementation("org.springframework.boot:spring-boot-starter-web")
    implementation("org.springframework.boot:spring-boot-starter-actuator")
    implementation("org.springframework.ai:spring-ai-openai-spring-boot-starter")
    testImplementation("org.springframework.boot:spring-boot-starter-test")
}

dependencyManagement {
    imports {
        mavenBom("org.springframework.ai:spring-ai-bom:1.0.0-SNAPSHOT")
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}
