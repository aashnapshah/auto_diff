#!/usr/bin/env bash
# File       : run_tests.sh
# Description: Test suite driver script
# Copyright 2022 Harvard University. All Rights Reserved.

# # Identify path to package source code
# PACKAGEPATH="$(pwd -P)/../src"

# # List of test cases to run
# TESTS=(
#     autodiff_tests/test_elem.py
#     autodiff_tests/test_autoDiff.py
#     autodiff_tests/test_dual.py
# )

# # Decide what driver to use (depending on arguments given)
# if [[ $# -gt 0 && ${1} == '-x' ]]; then
#     DRIVER= "pytest --cov=autoDiff --cov-report=xml"
# elif [[ $# -gt 0 && ${1} == '-v' ]]; then
#     DRIVER="pytest -v --cov=autoDiff --cov-report=term-missing"
# else
#     DRIVER="pytest --cov=autoDiff --cov-report=term-missing"
# fi

# # Add module source path
# export PYTHONPATH="${PACKAGEPATH}:${PYTHONPATH}"

# # Run the tests
# ${DRIVER} ${TESTS[@]}


tests=(
    autodiff_tests/test_elem.py
    autodiff_tests/test_ADclass.py
    autodiff_tests/test_dual.py
    autodiff_tests/test_differentiation.py
    autodiff_tests/test_node.py

)

# export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}
PACKAGEPATH="$(pwd -P)/../src"

# decide what driver to use (depending on arguments given)
if [[ $# -gt 0 && ${1} == 'coverage' ]]; then
    driver="${@} -m unittest"
elif [[ $# -gt 0 && ${1} == 'pytest' ]]; then
    driver="${@}"
elif [[ $# -gt 0 && ${1} == 'CI' ]]; then
    # Assumes the package has been installed and dependencies resolved.  This
    # would be the situation for a customer.  Uses `pytest` for testing.
    shift
    unset PYTHONPATH
    driver="pytest ${@}"
else
    driver="python ${@} -m unittest"
fi

export PYTHONPATH="${PACKAGEPATH}:${PYTHONPATH}"

# run the tests
${driver} ${tests[@]}