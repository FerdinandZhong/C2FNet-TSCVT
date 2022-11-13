PY_SOURCE_FILES=My*.py #this can be modified to include more files

install:
	pip install flake8 black isort autoflake

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name '*.pyc' -type f -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +

format:
	autoflake --in-place --remove-all-unused-imports --recursive ${PY_SOURCE_FILES}
	isort ${PY_SOURCE_FILES}
	black ${PY_SOURCE_FILES}