import pandas as pd
import numpy as np
from tqdm import tqdm
from MARLMacroeconomicEnv import MARLMacroeconomicEnv


def generate_dataset(env, num_steps=1000, random_seed=42):
    """
    Генерирует датасет взаимодействия с средой MARL.

    :param env: Экземпляр среды MARLMacroeconomicEnv.
    :param num_steps: Количество шагов симуляции.
    :param random_seed: Сид для воспроизводимости.
    :return: DataFrame с историей взаимодействия.
    """
    np.random.seed(random_seed)
    env.reset(seed=random_seed)

    # Списки для хранения данных
    data = {
        "step": [],
        "capital": [],
        "labor": [],
        "technology": [],
        "output": [],
        "consumer_actions": [],
        "firm_actions": [],
        "gov_action": [],
        "consumer_rewards": [],
        "firm_rewards": [],
        "gov_reward": [],
    }

    for step in tqdm(range(num_steps), desc="Generating dataset"):
        # Случайные действия агентов
        consumer_actions = np.random.rand(env.num_consumers, 1)  # Leisure choices
        firm_actions = np.random.rand(env.num_firms, 2)  # Prices, wages
        gov_action = np.random.rand(2)  # Income tax, corporate tax

        # Действия в формате для среды
        actions = {
            "Consumers": consumer_actions,
            "Firms": firm_actions,
            "Government": gov_action,
        }

        # Выполнить шаг в среде
        state, rewards, done, truncated, info = env.step(actions)

        # Сохранить данные текущего шага
        data["step"].append(step)
        data["capital"].append(state["Capital"][0])
        data["labor"].append(state["Labor"][0])
        data["technology"].append(state["Technology"][0])
        data["output"].append(state["Output"][0])
        data["consumer_actions"].append(consumer_actions.flatten().tolist())
        data["firm_actions"].append(firm_actions.flatten().tolist())
        data["gov_action"].append(gov_action.tolist())
        data["consumer_rewards"].append(rewards["Consumers"])
        data["firm_rewards"].append(rewards["Firms"])
        data["gov_reward"].append(rewards["Government"])

    # Создать DataFrame
    df = pd.DataFrame(data)

    # Распаковать вложенные списки для удобства анализа
    df = df.explode("consumer_actions").explode("firm_actions")
    df["consumer_actions"] = df["consumer_actions"].astype(float)
    df["firm_actions"] = df["firm_actions"].astype(float)

    return df


def save_dataset(df, filename="marl_dataset.csv"):
    """
    Сохраняет датасет в файл.

    :param df: DataFrame с данными.
    :param filename: Имя файла для сохранения.
    """
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    # Создать экземпляр среды
    env = MARLMacroeconomicEnv()

    # Генерация датасета
    dataset = generate_dataset(env, num_steps=1000)

    # Сохранение в файл
    save_dataset(dataset, filename="marl_dataset.csv")