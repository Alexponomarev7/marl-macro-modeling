import yaml
import random
import os
import hashlib

def generate_parameter_combinations(params_config):
    combinations = [{}]

    for param, config in params_config.items():
        new_combinations = []
        dist_type = config["distribution"]

        for combo in combinations:
            if dist_type == "discrete":
                for val in config["values"]:
                    new_combo = combo.copy()
                    new_combo[param] = val
                    new_combinations.append(new_combo)
            else:
                new_combo = combo.copy()
                new_combo[param] = sample_parameter(config)
                new_combinations.append(new_combo)

        combinations = new_combinations

    return combinations

def sample_parameter(config):
    dist_type = config["distribution"]

    if dist_type == "uniform":
        return round(random.uniform(config["min"], config["max"]), 3)
    elif dist_type == "normal":
        return round(random.normalvariate(config["mean"], config["std"]), 3)
    elif dist_type == "beta":
        return round(random.betavariate(config["a"], config["b"]), 3)
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")

def generate_monotonic_decreasing_sequence(n, start_value, min_step, max_step):
    """
    Генерация монотонно убывающей последовательности.
    :param n: Количество элементов в последовательности.
    :param start_value: Начальное значение.
    :param min_step: Минимальный шаг уменьшения.
    :param max_step: Максимальный шаг уменьшения.
    :return: Список значений.
    """
    sequence = [start_value]
    current_value = start_value
    for _ in range(n - 1):
        step = random.uniform(min_step, max_step)
        current_value -= step
        if current_value < 0:  # Ограничение, чтобы значения не стали отрицательными
            current_value = 0
        sequence.append(round(current_value, 3))
    return sequence

def generate_random_periods(n, max_period):
    """
    Генерация случайных периодов для шоков.
    :param n: Количество периодов.
    :param max_period: Максимальный период (обычно равен periods из конфига).
    :return: Отсортированный список уникальных периодов.
    """
    if n > max_period:
        raise ValueError(f"Cannot generate {n} unique periods from a range of 1 to {max_period}.")
    periods = sorted(random.sample(range(1, max_period + 1), n))
    return periods

def generate_model_file(model_template, params, output_dir, model_name, combo_id):
    # Подставляем параметры в шаблон с помощью .format()
    model_content = model_template.format(**params)

    # Генерация хэша от параметров
    params_hash = hashlib.sha256(str(params).encode()).hexdigest()[:8]

    # Сохраняем файл
    output_filename = f"{model_name}_{params_hash}.mod"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        f.write(model_content)
    return output_path


def main():
    # Загрузка конфигурационного файла
    with open('./dynare/conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    input_dir = "./dynare/docker/dynare_models_templates/"  # Укажите путь к директории с шаблонами моделей
    output_dir = "./dynare/docker/dynare_models/"  # Укажите путь для сохранения сгенерированных моделей

    # Обработка каждой модели в конфигурационном файле
    for model_name, model_settings in config.items():
        dynare_settings = model_settings.get("dynare_model_settings", {})
        if not dynare_settings:
            print(f"No dynare_model_settings found for model {model_name}. Skipping.")
            continue

        # Чтение шаблона модели
        model_template_path = os.path.join(input_dir, f"{model_name}.mod")
        if not os.path.exists(model_template_path):
            print(f"Template file for model {model_name} not found at {model_template_path}. Skipping.")
            continue

        with open(model_template_path, 'r') as f:
            model_template = f.read()

        # Генерация параметров
        params_config = dynare_settings.get("parameters", {})
        num_simulations = dynare_settings.get("num_simulations", 1)

        # Параметры для генерации шоков
        shock_settings = dynare_settings.get("shocks", {})
        n_shocks = shock_settings.get("n_shocks", 3)  # Количество шоков
        start_value = shock_settings.get("start_value", 0.2)  # Начальное значение шока
        min_step = shock_settings.get("min_step", 0.05)  # Минимальный шаг уменьшения
        max_step = shock_settings.get("max_step", 0.1)  # Максимальный шаг уменьшения

        # Общее количество периодов
        periods = dynare_settings.get("periods", 100)

        for i in range(num_simulations):
            # Генерация нового набора параметров для каждой симуляции
            parameter_combinations = generate_parameter_combinations(params_config)

            for combo_id, params in enumerate(parameter_combinations):
                # Добавляем periods в параметры, если он указан в конфиге
                if "periods" in dynare_settings:
                    params["periods"] = dynare_settings["periods"]

                # Генерация монотонно убывающей последовательности для шоков
                shock_values = generate_monotonic_decreasing_sequence(n_shocks, start_value, min_step, max_step)
                params["shock_values"] = " ".join(map(str, shock_values))

                # Генерация случайных периодов для шоков
                try:
                    shock_periods = generate_random_periods(n_shocks, periods)
                    params["shock_periods"] = " ".join(map(str, shock_periods))
                except ValueError as e:
                    print(f"Error generating periods for model {model_name}: {e}")
                    continue

                output_path = generate_model_file(model_template, params, output_dir, model_name, combo_id)
                print(f"Generated model {model_name} with parameters: {params} at {output_path}")

if __name__ == "__main__":
    main()