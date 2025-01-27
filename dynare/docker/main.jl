using ArgParse
using Dynare
using CSV
using DataFrames
using YAML

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
    # Создаем список всех возможных комбинаций параметров
    parameter_combinations = Vector{Vector{String}}()  # Явно указываем тип
    keys_list = collect(keys(model_settings))
    values_list = collect(values(model_settings))

    # Рекурсивная функция для генерации комбинаций
    function generate_combinations(current_combination::Vector{String}, index, keys_list, values_list)
        if index > length(keys_list)
            push!(parameter_combinations, current_combination)
            return
        end

        key = keys_list[index]
        value = values_list[index]

        if isa(value, Vector)
            # Если значение — это список, перебираем все его элементы
            for v in value
                new_combination = copy(current_combination)
                push!(new_combination, "-D$(key)=$(string(v))")
                generate_combinations(new_combination, index + 1, keys_list, values_list)
            end
        else
            # Если значение — это скаляр, добавляем его в комбинацию
            new_combination = copy(current_combination)
            push!(new_combination, "-D$(key)=$(string(value))")
            generate_combinations(new_combination, index + 1, keys_list, values_list)
        end
    end

    generate_combinations(String[], 1, keys_list, values_list)
    return parameter_combinations
end

function main()
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
            parameter_combinations = generate_parameter_combinations(model_settings)

            # Запуск модели для каждой комбинации параметров
            for (i, parameters) in enumerate(parameter_combinations)
                println("Running model $model_name with parameters: ", parameters)
                input_file = joinpath(input_dir, file_name)
                output_file = joinpath(output_dir, join([model_name, "config_$(i)", "raw.csv"], "_"))
                run_model(input_file, output_file, parameters)
                println("Output saved to $output_file")
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