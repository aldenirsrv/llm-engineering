# Source code
mkdir -p src/{domain/{entities,services},application/{interfaces,use_cases},infrastructure/{repositories,external,config},main}

# Tests
mkdir -p tests/{unit/{domain,application,infrastructure},integration}

# Data
mkdir -p data

find src -type d -exec touch {}/__init__.py \;