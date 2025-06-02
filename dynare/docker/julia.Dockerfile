FROM julia:1.10.5

WORKDIR /app

RUN julia -e 'using Pkg; Pkg.add("DataFrames"); Pkg.add("CSV"); Pkg.add("ArgParse"); Pkg.add("YAML"); Pkg.add("Dynare"); Pkg.add(PackageSpec(name="TransformVariables", version="0.6.2")); using Dynare;'