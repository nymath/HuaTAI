build_push_docs:
	@mkdocs build
	@mkdocs gh-deploy

build_cython:
	@python setup.py build_ext --inplace

build_package:
	@python setup.py sdist bdist_wheel

publish_package:
	@twine upload --skip-existing dist/*

git_log:
	@git log --all --graph --decorate --oneline
	
ls:
	@ls $(0)