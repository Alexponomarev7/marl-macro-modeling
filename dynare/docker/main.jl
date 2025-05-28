using ArgParse
using Dynare
using CSV
using DataFrames
using YAML
using Random

function sample_from_range(range::Vector{<:Real})
    return rand() * (range[2] - range[1]) + range[1]
end

function run_model(input_file::String, output_file::String, parameters::Vector{String}, max_retries=3)
    retries = 0
    while retries < max_retries
        println("Running model: $input_file (attempt $(retries + 1))")
        try
            # Run model with parameters
            context = dynare(input_file, parameters...)
            simul_length = length(context.results.model_results[1].simulations)
            if simul_length > 0
                println("Successful simulation!")
            else
                println("Simulation failed!")
            end

            # Save results
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
        "--config_path"
            help = "Path to YAML file with config"
            required = true
        "--num_samples"
            help = "Number of parameter combinations to sample"
            arg_type = Int
            default = 10
    end

    return parse_args(s)
end

function generate_parameter_combinations(model_settings, num_samples)
    parameter_combinations = Vector{Vector{String}}()
    parameter_values = Vector{Dict{String,Float64}}()
    
    for _ in 1:num_samples
        current_combination = String[]
        current_values = Dict{String,Float64}()
        
        # Handle periods separately since it's not a range
        if haskey(model_settings, "periods")
            push!(current_combination, "-Dperiods=$(model_settings["periods"])")
            current_values["periods"] = model_settings["periods"]
        end
        
        # Sample from parameter ranges
        if haskey(model_settings, "parameter_ranges")
            for (param, range) in model_settings["parameter_ranges"]
                value = sample_from_range(range)
                push!(current_combination, "-D$(param)=$(value)")
                current_values[param] = value
            end
        end
        
        push!(parameter_combinations, current_combination)
        push!(parameter_values, current_values)
    end
    
    return parameter_combinations, parameter_values
end

function main()
    # Parse command line arguments
    parsed_args = parse_commandline()

    input_dir = parsed_args["input_dir"]
    output_dir = parsed_args["output_dir"]
    config_path = parsed_args["config_path"]
    num_samples = parsed_args["num_samples"]

    config = YAML.load_file(config_path)
    model_names = collect(keys(config))

    for model_name in model_names
        model_settings = config[model_name]["dynare_model_settings"]
        parameter_combinations, parameter_values = generate_parameter_combinations(model_settings, num_samples)

        for (i, (parameters, values)) in enumerate(zip(parameter_combinations, parameter_values))
            println("Running model $model_name with parameters: ", parameters)
            input_file = joinpath(input_dir, model_name * ".mod")
            base_name = join([model_name, "config_$(i)"], "_")
            output_file = joinpath(output_dir, base_name * "_raw.csv")
            config_file = joinpath(output_dir, base_name * "_config.yml")
            
            # Save parameter config
            YAML.write_file(config_file, values)
            
            run_model(input_file, output_file, parameters)
            println("Output saved to $output_file")
            println("Config saved to $config_file")
        end
    end
end

# Run main function
main()