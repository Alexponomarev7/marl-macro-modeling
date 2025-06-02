using ArgParse
using Dynare
using CSV
using DataFrames
using YAML
using Random

function sample_from_range(range::Vector{<:Real})
    return rand() * (range[2] - range[1]) + range[1]
end

function dump_context_work(context, output_path::String = "config_dump.yml")
    config = Dict{String, Any}()

    for name in fieldnames(typeof(context.work))
        field_value = getfield(context.work, name)

        try
            # Преобразуем к сериализуемому виду
            if isa(field_value, AbstractDict)
                config[string(name)] = Dict(string(k) => string(v) for (k, v) in field_value)
            elseif isa(field_value, AbstractArray)
                config[string(name)] = [string(x) for x in field_value]
            elseif isa(field_value, Number) || isa(field_value, String) || isa(field_value, Bool)
                config[string(name)] = field_value
            else
                # fallback для любых прочих типов
                config[string(name)] = string(field_value)
            end
        catch e
            @warn "Skipping field $name due to error: $e"
        end
    end

    YAML.write_file(output_path, config)
    println("✅ Dumped context.work to $output_path")
end

function run_model(input_file::String, output_file::String, periods::Int, parameters::Vector{String}, max_retries=3)
    retries = 0
    while retries < max_retries
        println("Running model: $input_file (attempt $(retries + 1))")
        try
            shock_vals_A = "-Dshock_vals_A=[" * join([string((1+0.02)^t) for t in 0:periods-1], ";") * "]"
            shock_vals_L = "-Dshock_vals_L=[" * join([string((1+0.01)^t) for t in 0:periods-1], ";") * "]"
            context = dynare(input_file, parameters...)
            simul_length = length(context.results.model_results[1].simulations)
            if simul_length > 0
                println("Successful simulation!")
            else
                println("Simulation failed!")
            end

            # Save simulation results
            data = context.results.model_results[1].simulations[1].data
            println(data)
            dataframe = DataFrame(data)
            CSV.write(output_file, dataframe)

            params_output = replace(output_file, ".csv" => "_params.yml")
            dump_context_work(context, params_output)            
            println("Model $input_file completed successfully.")
            return
        catch e
            println("Error running model $input_file:")
            println("Error message: $e")
            println("Stack trace:")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
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
            default = 1
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
            
            run_model(input_file, output_file, model_settings["periods"], parameters)
            println("Output saved to $output_file")
            println("Config saved to $config_file")
        end
    end
end

# Run main function
main()