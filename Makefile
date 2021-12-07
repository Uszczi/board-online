


###############################
### Linting
###########
   # TODO Make paths as variable
lint:
	python -m autoflake --in-place --recursive --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables --remove-all-unused-imports .
	python -m black .
	python -m isort .
	python -m mypy --ignore-missing-imports . 



###############################
### Dependency
#############
sync-deps:
	python -m piptools sync "./requirements/dev.txt"
	pip install -e .

compile-deps:
	python -m piptools compile --no-header "./requirements/dev.in"

recompile-deps:
	python -m piptools compile --no-header --upgrade "./requirements/dev.in"







