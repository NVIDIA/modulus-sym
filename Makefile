container-deploy:
	docker build -t modulus-sym:deploy --target with-pysdf -f Dockerfile .

container-docs:
	docker build -t modulus-sym:docs --target docs -f Dockerfile .
