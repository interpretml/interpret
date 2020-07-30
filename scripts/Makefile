ROOTDIR := $(realpath ..)

PYTHON ?= python
ifeq ($(OS),Windows_NT)
	BUILD_NATIVE ?= $(ROOTDIR)/build.bat
	RMDIR ?= rmdir /Q /S
	TOUCH ?= echo.>
else
	BUILD_NATIVE ?= $(ROOTDIR)/build.sh
	RMDIR ?= rm -rf
	TOUCH ?= touch
endif
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))

docker:
	docker build -t interpretml/quickbuild:latest .
	docker run -it -v $(ROOTDIR):/usr/staging/interpret interpretml/quickbuild:latest

clean:
	-$(RMDIR) "$(ROOTDIR)/staging"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/interpret/lib"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/js/dist"
	-$(RMDIR) "$(ROOTDIR)/python/interpret-core/js/node_modules"

build: build-python

build-python: build-javascript build-native
	cd $(ROOTDIR)/python/interpret-core && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/staging
	cd $(ROOTDIR)/python/interpret && $(PYTHON) setup.py bdist_wheel -d $(ROOTDIR)/staging

build-javascript:
	cd $(ROOTDIR)/python/scripts && $(PYTHON) build-js-bundles.py devops

build-native:
	$(BUILD_NATIVE)