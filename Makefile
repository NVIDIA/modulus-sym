install:
	pip install --upgrade pip && \
		pip install -e .

black: 
	black --check --exclude='/(docs|deps)/' ./

interrogate:
	echo "Interrogate CI stage not currently implemented"

license: 
	python test/ci_tests/header_check.py

doctest:
	echo "Doctest CI stage not currently implemented"

pytest: 
	coverage run \
		--rcfile='test/coverage.pytest.rc' \
		-m pytest test/

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
