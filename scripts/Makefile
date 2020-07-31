ROOTDIR ?= $(realpath ..)

DOCKER ?= docker
DOCKER_IMG_NAME ?= interpretml/quickbuild:latest
DOCKER_VOL_NAME ?= interpretml-quickbuild
DOCKER_MNT_PATH ?= /mnt/interpret
PYTHON ?= python
NPM ?= npm
ifeq ($(OS),Windows_NT)
	BUILD_NATIVE ?= $(ROOTDIR)/build.bat
	MKDIR_P ?= mkdir
	RMDIR ?= rmdir /Q /S
	COPY ?= copy
	TOUCH ?= echo.>
else
	BUILD_NATIVE ?= $(ROOTDIR)/build.sh
	MKDIR_P ?= mkdir -p
	RMDIR ?= rm -rf
	COPY ?= cp
	TOUCH ?= touch
endif
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))

docker:
	$(DOCKER) build -t $(DOCKER_IMG_NAME) .
	$(DOCKER) volume create --name $(DOCKER_VOL_NAME)
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):/usr/staging \
		--entrypoint /bin/bash $(DOCKER_IMG_NAME) $(DOCKER_MNT_PATH)/scripts/mirror-repo-docker.sh /usr/staging/interpret
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):/usr/staging \
		--entrypoint /bin/sh $(DOCKER_IMG_NAME) -c 'cd /usr/staging/interpret/scripts && /bin/bash'

clean:
	-$(RMDIR) "$(ROOTDIR)/staging"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/interpret/lib"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/js/dist"
	-$(DOCKER) volume rm $(DOCKER_VOL_NAME)
	-$(DOCKER) image rm $(DOCKER_IMG_NAME)

build: build-python

build-python: build-javascript build-native
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/staging
	cd $(ROOTDIR)/python/interpret && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/staging

build-javascript:
	cd $(ROOTDIR)/python/interpret-core/js && $(NPM) install
	cd $(ROOTDIR)/python/interpret-core/js && $(NPM) run build-prod
	-$(MKDIR_P) "$(ROOTDIR)/python/interpret-core/interpret/lib"
	cd $(ROOTDIR)/python/interpret-core/js/dist && $(COPY) "interpret-inline.js" "$(ROOTDIR)/python/interpret-core/interpret/lib/"

build-native:
	$(BUILD_NATIVE)

test: test-native test-javascript test-python

test-python:
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) -m pytest -vv -n auto --runslow --doctest-modules --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html

test-javascript:
	@echo JavaScript tests are not available.

test-native:
	@echo Native tests are not yet connected.