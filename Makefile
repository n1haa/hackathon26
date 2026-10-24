.PHONY: install build test lint run check
install build test lint run check:
	python manage.py $@
