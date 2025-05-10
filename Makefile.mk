SHELL := /bin/sh
GRADLE_VERSION ?= 8.14

b:
	gradle build
test:
	gradle test
install:
	sdk install kotlin
install-locust:
	pip install locust
install-locust-linux:
	sudo apt-get install locust
install-python:
	pip install charset_normalizer
upgrade-pip:
	pip install --upgrade pip
