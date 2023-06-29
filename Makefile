install:
	pip install --upgrade pip && \
		pip install -e .

black: 
	black --check --exclude=docs/ ./

interrogate:
	echo "Interrogate CI stage not currently implemented"

license: 
	python test/ci_tests/header_check.py

doctest:
	coverage run \
                --rcfile='test/coverage.docstring.rc' \
                -m pytest \
                --doctest-modules modulus/ --ignore-glob=*internal*

pytest: 
	coverage run \
                --rcfile='test/coverage.pytest.rc' \
                -m pytest 

coverage:
	coverage combine && \
		coverage report --show-missing --omit=*test* --omit=*internal* --fail-under=80 && \
		coverage html

container-deploy:
	docker build -t modulus-sym:deploy --target deploy -f Dockerfile .

container-ci:
	docker build -t modulus-sym:ci --target ci -f Dockerfile .

container-docs:
	docker build -t modulus-sym:docs --target docs -f Dockerfile .
