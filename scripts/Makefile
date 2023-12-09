ROOTDIR ?= $(realpath ..)

DOCKER ?= docker
DOCKER_IMG_NAME ?= interpretml/quickbuild:latest
DOCKER_VOL_NAME ?= interpretml-quickbuild
DOCKER_MNT_PATH ?= /mnt/interpret
DOCKER_CWD ?= /usr/staging/interpret

PYTHON ?= python
PIP ?= pip
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

docker: prepare-docker
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):$(DOCKER_CWD) \
	    -p 127.0.0.1:8888:8888 \
	    -p 127.0.0.1:7001:7001 \
		-e INTERPRET_DOCKER_MODE='dev' \
		--entrypoint /bin/sh $(DOCKER_IMG_NAME) -c "\
			cd $(DOCKER_CWD)/python/interpret-core && pip install -e . &&\
			cd $(DOCKER_CWD)/python/interpret && pip install -e . &&\
			cd $(DOCKER_CWD)/scripts && /bin/bash"

run-docker-notebook: prepare-docker
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):$(DOCKER_CWD) \
	    -p 127.0.0.1:8888:8888 \
	    -p 127.0.0.1:7001:7001 \
		-e INTERPRET_DOCKER_MODE='dev' \
		--entrypoint /bin/bash $(DOCKER_IMG_NAME) -c "\
			cd $(DOCKER_CWD)/python/interpret-core && pip install -e . &&\
			cd $(DOCKER_CWD)/python/interpret && pip install -e . &&\
			cd $(DOCKER_CWD)/examples/python/notebooks &&\
			python -m jupyter notebook --ip=0.0.0.0 --allow-root"

build-in-docker: prepare-docker
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):$(DOCKER_CWD) \
		--entrypoint /bin/bash $(DOCKER_IMG_NAME) -c "cd $(DOCKER_CWD)/scripts && make build"

prepare-docker:
	$(DOCKER) build -t $(DOCKER_IMG_NAME) .
	$(DOCKER) volume create --name $(DOCKER_VOL_NAME)
	$(DOCKER) run --rm -it -v $(ROOTDIR):$(DOCKER_MNT_PATH) -v $(DOCKER_VOL_NAME):$(DOCKER_CWD) \
		--entrypoint /bin/bash $(DOCKER_IMG_NAME) $(DOCKER_MNT_PATH)/scripts/mirror-repo-docker.sh $(DOCKER_CWD)

clean:
	-$(RMDIR) "$(ROOTDIR)/bld"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/interpret/lib"
	-$(RMDIR) "$(ROOTDIR)/shared/vis/dist"
	-$(DOCKER) volume rm $(DOCKER_VOL_NAME)
	-$(DOCKER) image rm $(DOCKER_IMG_NAME)

install: build
	-$(PIP) uninstall -y interpret
	-$(PIP) uninstall -y interpret-core
	cd $(ROOTDIR)/python/interpret-core && $(PIP) install -e .
	cd $(ROOTDIR)/python/interpret && $(PIP) install -e .
install-core: build
	-$(PIP) uninstall -y interpret
	-$(PIP) uninstall -y interpret-core
	cd $(ROOTDIR)/python/interpret-core && $(PIP) install -e .


build: build-wheel
build-wheel: build-javascript build-native
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/bld/bdist
	cd $(ROOTDIR)/python/interpret && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/bld/bdist

build-python:
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/bld/bdist
	cd $(ROOTDIR)/python/interpret && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/bld/bdist

build-javascript:
	cd $(ROOTDIR)/shared/vis && $(NPM) install
	cd $(ROOTDIR)/shared/vis && $(NPM) run build-prod
	-$(MKDIR_P) "$(ROOTDIR)/python/interpret-core/interpret/lib"
	cd $(ROOTDIR)/shared/vis/dist && $(COPY) "interpret-inline.js" "$(ROOTDIR)/python/interpret-core/interpret/lib/"
	cd $(ROOTDIR)/shared/vis/dist && $(COPY) "interpret-inline.js.LICENSE.txt" "$(ROOTDIR)/python/interpret-core/interpret/lib/"

build-native:
	$(BUILD_NATIVE)

test: test-native test-javascript test-python

test-python:
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) -m pytest -vv -n auto --runslow --junitxml=junit/test-results.xml --cov=com --cov-report=xml --cov-report=html

test-javascript:
	@echo JavaScript tests are not available.

test-native:
	@echo Native tests are not yet connected.
