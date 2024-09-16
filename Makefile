include Makefile.mk

b: wrapper
	./gradlew build
wrapper:
	gradle wrapper
install-spring-boot-cli:
	brew tap spring-cli-projects/spring-cli
	brew install spring-cli
install-home-brew:
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
create-new-service:
	spring boot new --from ai --name oga-simple-service
deps-plugins-update:
	curl -sL https://raw.githubusercontent.com/jesperancinha/project-signer/master/pluginUpdatesOne.sh | bash -s -- $(PARAMS)
deps-java-update:
	curl -sL https://raw.githubusercontent.com/jesperancinha/project-signer/master/javaUpdatesOne.sh | bash
deps-gradle-update:
	curl -sL https://raw.githubusercontent.com/jesperancinha/project-signer/master/gradleUpdatesOne.sh | bash
deps-quick-update: deps-gradle-update deps-plugins-update deps-java-update
accept-prs:
	curl -sL https://raw.githubusercontent.com/jesperancinha/project-signer/master/acceptPR.sh | bash
