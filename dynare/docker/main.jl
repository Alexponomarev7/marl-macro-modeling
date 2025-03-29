using ArgParse
using Dynare
using CSV
using DataFrames
using YAML

function run_model(input_file::String, output_file::String, variable_mapping::Dict{Any, Any}, max_retries=3)
    retries = 0
    while retries < max_retries
        println("Running model: $input_file (attempt $(retries + 1))")
        try
            # Запуск модели
            context = dynare(input_file)
            simul_length = length(context.results.model_results[1].simulations)
            if simul_length > 0
                println("Successful simulation!")
            else
                println("Simulation failed!")
            end

            # Сохранение результатов
            data = context.results.model_results[1].simulations[1].data
            dataframe = DataFrame(data)

            # Переименование колонок на основе маппинга
            for (original_name, new_name) in variable_mapping
                if hasproperty(dataframe, Symbol(original_name))
                    rename!(dataframe, Symbol(original_name) => Symbol(new_name))
                end
            end

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

function extract_model_name(file_name::String)
    """
    Извлекает базовое имя модели из имени файла, удаляя хэш и расширение.
    Например, для "Gali_2008_chapter_2_2b8f6481.mod" вернет "Gali_2008_chapter_2".
    """
    # Удаляем расширение .mod
    base_name = replace(file_name, ".mod" => "")
    # Удаляем хэш (последний сегмент после последнего '_')
    model_name = join(split(base_name, '_')[1:end-1], '_')
    return model_name
end

function extract_hash(file_name::String)
    """
    Извлекает хэш из имени файла.
    Например, для "Gali_2008_chapter_2_2b8f6481.mod" вернет "2b8f6481".
    """
    # Удаляем расширение .mod
    base_name = replace(file_name, ".mod" => "")
    # Извлекаем хэш (последний сегмент после последнего '_')
    hash = split(base_name, '_')[end]
    return hash
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
        # Извлекаем базовое имя модели (без хэша)
        model_name = extract_model_name(file_name)

        if haskey(config, model_name)
            # Извлечение маппинга переменных
            rl_env_settings = config[model_name]["rl_env_settings"]
            variable_mapping = rl_env_settings["input"]["all_columns"]

            # Путь к входному файлу модели
            input_file = joinpath(input_dir, file_name)

            # Извлекаем хэш из имени файла
            hash = extract_hash(file_name)

            # Путь к выходному файлу (с хэшем)
            output_file = joinpath(output_dir, "$(model_name)_$(hash).csv")

            # Запуск модели и переименование колонок
            run_model(input_file, output_file, variable_mapping)
            println("Output saved to $output_file")
        else
            println("No configuration found for $model_name. Skipping.")
        end
    end
end

# Запуск основной функции
main()