FROM julia:latest

WORKDIR /app

RUN julia -e 'using Pkg; Pkg.add("Dynare"); Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("ArgParse"); Pkg.add("YAML")'