# docker build -f ./docker/julia_dynare/julia.Dockerfile

docker run -it \
  -v ./docker/julia_dynare/dynare_models:/app/input \
  -v ./data/raw/:/app/output \
  -v ./docker/julia_dynare/main.jl:/app/main.jl \
  julia-dynare julia main.jl --input_dir input --output_dir output

python lib/dynare_traj2rl_transitions.py