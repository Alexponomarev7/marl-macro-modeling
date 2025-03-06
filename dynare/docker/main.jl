using ArgParse
using Dynare
using CSV
using DataFrames
using YAML
using Distributions
using Random
using SHA


function run_model(input_file::String, output_file::String, parameters::Vector{String}, max_retries=3)
    retries = 0
    while retries < max_retries
        println("Running model: $input_file (attempt $(retries + 1))")
        try
            # Запуск модели с параметрами
            context = dynare(input_file, parameters...)
            simul_length = length(context.results.model_results[1].simulations)
            if simul_length > 0
                println("Successful simulation!")
            else
                println("Simulation failed!")
            end

            # Сохранение результатов
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
    end

    return parse_args(s)
end

function generate_parameter_combinations(model_settings)
    # Разделяем параметры и флаги
    params = get(model_settings, "parameters", Dict())
    flags = get(model_settings, "flags", Dict())

    periods = model_settings["periods"]

    # Генерация случайных параметров
    param_combos = _generate_param_combinations(params)

    # Генерация всех возможных комбинаций флагов
    flag_combos = _generate_flag_combinations(flags)

    # Генерация шума
    n = 10  # Количество точек
    start_value = 0.1  # Начальное значение
    min_step = 0.0  # Минимальный шаг уменьшения
    max_step = 0.3 # Максимальный шаг уменьшения
    shock_values = _generate_monotonic_decreasing_noise(n, start_value, min_step, max_step)

    # Комбинируем параметры и флаги
    full_combinations = Vector{Dict{String,Any}}()
    for p in param_combos, f in flag_combos
        combined = merge(p, f)
        combined["periods"] = periods
        # combined["shock_values"] = shock_values
        push!(full_combinations, combined)
    end

    # Преобразуем в аргументы командной строки
    return [["-D$(k)=$(v)" for (k, v) in combo] for combo in full_combinations]
end

function _generate_monotonic_decreasing_noise(n::Int, start_value::Float64, min_step::Float64, max_step::Float64)
    noise = Float64[start_value]
    current_value = start_value

    for _ in 2:n
        step = rand(min_step:max_step)  # Случайный шаг уменьшения
        current_value -= step
        if current_value <= 0  # Ограничение, чтобы значения не становились отрицательными
            push!(noise, 0.0)
            break
        end
        push!(noise, current_value)
    end

    return noise
end

function _generate_param_combinations(params::Dict)
    combinations = [Dict{String,Any}()]

    for (param, config) in params
        new_combinations = []
        dist_type = config["distribution"]

        # Для каждого существующего сочетания добавляем новые значения
        for combo in combinations
            if dist_type == "discrete"
                for val in config["values"]
                    push!(new_combinations, merge(combo, Dict(param => val)))
                end
            else
                val = _sample_parameter(config)
                push!(new_combinations, merge(combo, Dict(param => val)))
            end
        end

        combinations = new_combinations
    end

    return combinations
end

function _generate_flag_combinations(flags::Dict)
    flag_names = collect(keys(flags))
    flag_values = collect(values(flags))

    # Генерируем все возможные комбинации флагов
    combinations = []
    for combo in Base.product(flag_values...)
        push!(combinations, Dict(zip(flag_names, combo)))
    end

    return unique(combinations)
end

function _sample_parameter(config::Dict)
    dist_type = config["distribution"]

    if dist_type == "uniform"
        rand(Uniform(config["min"], config["max"])) |> x->round(x, digits=3)
    elseif dist_type == "normal"
        rand(Normal(config["mean"], config["std"])) |> x->round(x, digits=3)
    elseif dist_type == "beta"
        rand(Beta(config["a"], config["b"])) |> x->round(x, digits=3)
    else
        error("Unsupported distribution: $dist_type")
    end
end

function generate_filename(output_dir::String, model_name::SubString{String}, combo::Vector{String})
    params = []
    flags = []

    for param in combo
        # Разделяем строку на ключ и значение
        key_value = split(param, '=')
        if length(key_value) == 2
            key = key_value[1]
            value = key_value[2]

            # Проверяем, является ли значение числом
            try
                num_value = parse(Float64, value)
                push!(params, "$key=$(round(num_value, digits=2))")
            catch
                # Если не число, считаем это флагом
                push!(flags, "$key=$value")
            end
        end
    end

    # Формируем имя файла
    hash_input = ""
    if !isempty(params)
        hash_input *= "_" * join(params, "_")
    end
    if !isempty(flags)
        hash_input *= "_" * join(flags, "-")
    end

    hash_value = bytes2hex(sha256(hash_input))
    filename = "$(model_name)_$(hash_value).csv"

    return join([output_dir, filename], "/")
end

function main()

    Random.seed!(4242)
    # Парсинг аргументов командной строки
    parsed_args = parse_commandline()

    input_dir = parsed_args["input_dir"]
    output_dir = parsed_args["output_dir"]
    config_path = parsed_args["config_path"]

    # Загрузка конфига
    config = YAML.load_file(config_path)

    # Получение списка файлов моделей
    input_files = filter(f -> endswith(f, ".mod"), readdir(input_dir))

    # Запуск моделей
    for file_name in input_files
        model_name = split(file_name, ".")[1]  # Имя модели без расширения

        if haskey(config, model_name)
            # Извлечение параметров для модели
            model_settings = config[model_name]["dynare_model_settings"]

            num_simulations = model_settings["num_simulations"]
            i = 0

            while i < num_simulations

                parameter_combinations = generate_parameter_combinations(model_settings)

                # Запуск модели для каждой комбинации параметров
                for (i, parameters) in enumerate(parameter_combinations)
                    println("Running model $model_name with parameters: ", parameters)
                    input_file = joinpath(input_dir, file_name)

                    # output_file = joinpath(output_dir, join([model_name, "config_$(i)", "raw.csv"], "_"))
                    output_file = generate_filename(output_dir, model_name, parameters)

                    run_model(input_file, output_file, parameters)
                    println("Output saved to $output_file")
                end

                i += 1
            end
        else
            # Если параметры для модели не указаны, используем пустой массив
            parameters = String[]
            println("No parameters found for $model_name. Running with default settings.")
            input_file = joinpath(input_dir, file_name)
            output_file = joinpath(output_dir, join([model_name, "raw.csv"], "_"))
            run_model(input_file, output_file, parameters)
            println("Output saved to $output_file")
        end
    end
end

# Запуск основной функции
main()