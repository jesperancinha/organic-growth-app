plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.9.0"
}

rootProject.name = "organic-growth-app"
include("oga-simple-service")
include("oga-kotlindl-runner")
include("oga-deeplearning4j-runner")
include("org-langchain4j-runner")
