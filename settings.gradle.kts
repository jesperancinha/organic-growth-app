plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.8.0"
}

rootProject.name = "organic-growth-app"
include("oga-simple-service")
include("oga-kotlindl-runner")
include("oga-deeplearning4j-runner")
