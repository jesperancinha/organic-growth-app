b:
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
