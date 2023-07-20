install:
	pip install --upgrade pip && \
		pip install -e .

get-data:
	mkdir -p /data && \
		mkdir -p /data/nfs/ && \
		git -C /data/nfs/modulus-data pull || \
		git clone https://gitlab-master.nvidia.com/modulus/modulus-data.git /data/nfs/modulus-data

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	# pre-commit run interrogate -a
	echo "Interrogate CI stage not currently implemented"

lint:
	pre-commit run markdownlint -a

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
		-m pytest 

pytest-internal:
	cd test/internal && \
		pytest && \
		cd ../../

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=50 && \
		coverage html

container-deploy:
	docker build -t modulus-sym:deploy --target deploy -f Dockerfile .

container-ci:
	docker build -t modulus-sym:ci --target ci -f Dockerfile .

container-docs:
	docker build -t modulus-sym:docs --target docs -f Dockerfile .

