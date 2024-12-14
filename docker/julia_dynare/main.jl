using ArgParse
using Dynare
using CSV
using DataFrames

function run_model(input_file::String, output_file::String, max_retries=3)
    retries = 0
    while retries < max_retries
        println("Running model: $input_file (attempt $(retries + 1))")
        try
            context = dynare(input_file)
            data = context.results.model_results[1].simulations[1].data
            dataframe = DataFrame(data)
            CSV.write(output_file, dataframe)
            println("Model $input_file completed successfully.")
            return
        catch e
            println("Error running model $input_file: $e")
            retries += 1
            if retries < max_retries
                println("Retrying model $input_file...")
            else
                println("Failed to run model $input_file after $max_retries attempts.")
            end
        end
    end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--input_dir"
            help = "Path to the models directory to run"
            required = true
        "--output_dir"
            help = "Path to the trajectories directory"
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

input_dir = parsed_args["input_dir"]
output_dir = parsed_args["output_dir"]

input_files = filter(f -> endswith(f, ".mod"), readdir(input_dir))

for file_name in input_files
    input_file = joinpath(input_dir, file_name)
    output_file = joinpath(output_dir, join([split(file_name, ".")[1], "raw.csv"], "_"))
    run_model(input_file, output_file)
    println("output_file $output_file")
end