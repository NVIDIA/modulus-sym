install:
	pip install --upgrade pip && \
		pip install -e .

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	# pre-commit run interrogate -a
	echo "Interrogate CI stage not currently implemented"

lint:
	# pre-commit run markdownlint -a
	echo "Lint CI stage not currently implemented"

license: 
	pre-commit run license -a

doctest:
	# coverage run \
	# 	--rcfile='test/coverage.docstring.rc' \
	# 	-m pytest \
	# 	--doctest-modules modulus/ --ignore-glob=*internal*
	echo "Doctest CI stage not currently implemented"

pytest: 
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest test/

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=50 && \
		coverage html

# For arch naming conventions, refer
# https://docs.docker.com/build/building/multi-platform/
# https://github.com/containerd/containerd/blob/v1.4.3/platforms/platforms.go#L86
ARCH := $(shell uname -p)

ifeq ($(ARCH), x86_64)
    TARGETPLATFORM := "linux/amd64"
else ifeq ($(ARCH), aarch64)
    TARGETPLATFORM := "linux/arm64"
else
    $(error Unknown CPU architecture ${ARCH} detected)
endif

MODULUS_SYM_GIT_HASH = $(shell git rev-parse --short HEAD)

container-deploy:
	docker build -t modulus-sym:deploy --build-arg TARGETPLATFORM=${TARGETPLATFORM} --build-arg MODULUS_SYM_GIT_HASH=${MODULUS_SYM_GIT_HASH} --target deploy -f Dockerfile .

container-ci:
	docker build -t modulus-sym:ci --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target ci -f Dockerfile .

container-docs:
	docker build -t modulus-sym:docs --build-arg TARGETPLATFORM=${TARGETPLATFORM} --target docs -f Dockerfile .

