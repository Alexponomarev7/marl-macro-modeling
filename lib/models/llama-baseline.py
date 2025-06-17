import math
from lib.dataset import ACTION_MAPPING, ENV_MAPPING, STATE_MAPPING
from lib.envs import NAME_TO_ENV
from lib.envs.environment_base import ENV_TO_ID, AbstractEconomicEnv
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMBaseline(nn.Module):

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def _construct_model_description_prompt(self, env: AbstractEconomicEnv, model_params: torch.Tensor) -> str:
        model_params_str = ", ".join([f"{k}={model_params[i]}" for i, k in enumerate(sorted(env.params.keys()))])
        return f"""You are an expert in the field of macroeconomics. 
        You are given a model description and a set of model parameters. You need to output the next action to take. 
        The model description is: {env.name} and the model parameters are: {model_params_str}.
        The state descriptions are: {env.state_description} and the action descriptions are: {env.action_description}.
        """

    def _construct_trajectory_prompt(self, env: AbstractEconomicEnv, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> str:
        prompt = ""
        for i in range(len(states)):
            prompt += f"Tick {i} (State: {states[i][:len(env.state_description)]}, Action: {actions[i][:len(env.action_description)]}, Reward: {rewards[i]})\n"
        return prompt


    def forward(
        self,
        states: torch.Tensor,
        states_info: torch.Tensor,
        actions: torch.Tensor,
        actions_info: torch.Tensor,
        rewards: torch.Tensor,
        task_ids: torch.Tensor,
        model_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = task_ids.shape[0]
        outputs = []
        for i in range(batch_size):
            task_id = int(task_ids[i])
            inverse_mapping = {v: k for k, v in ENV_MAPPING.items()}
            env_name: str = inverse_mapping[task_id]
            env = NAME_TO_ENV[env_name]

            prompt_description = self._construct_model_description_prompt(env, task_id, model_params[i])
            prompt_trajectory = self._construct_trajectory_prompt(env, states[i], actions[i], rewards[i])
            prompt = prompt_description + "\n\n" + prompt_trajectory
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            output = self.model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
            outputs.append(self.tokenizer.decode(output[0], skip_special_tokens=True))
        return torch.tensor(outputs, dtype=torch.float32), None

    def _get_state_info(self, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state_values, state_ids = [], []
        for state_name, state_value in state.items():
            assert state_name in STATE_MAPPING, f"State {state_name} not found in STATE_MAPPING"
            state_ids.append(STATE_MAPPING[state_name])
            state_values.append(state_value)
        
        state_values += [0] * (self.state_dim - len(state_values))
        state_ids += [STATE_MAPPING["Empty"]] * (self.state_dim - len(state_ids))
        return torch.tensor(state_values, dtype=torch.float32), torch.tensor(state_ids, dtype=torch.long)

    def inference(self, env: AbstractEconomicEnv, max_steps: int = 50) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        init_state, _ = env.reset()
        init_state_values, states_info = self._get_state_info({
            state_name: init_state[state_name] for state_name in env.state_description.keys()
        })
        init_action_values, actions_info = self._get_action_info(
            {k: 0.0 for k, _ in env.action_description.items()}
        )

        state_to_plot = [{
            state_name: init_state[state_name] for state_name in env.state_description.keys()
        }]
        action_to_plot = [{k: 0.0 for k, _ in env.action_description.items()}]

        state_history = [init_state_values]
        action_history = [init_action_values]
        reward_history = [torch.tensor([0.0], dtype=torch.float32)]
        task_ids = torch.tensor([env.task_id], dtype=torch.long)
        model_params = torch.tensor([v for _, v in sorted(env.params.items())] + [0] * (self.model_params_dim - len(env.params)), dtype=torch.float32)

        for _ in range(max_steps):
            out, _ = self.forward(
                states=torch.stack(state_history).unsqueeze(0).to(self.device),
                states_info=states_info.unsqueeze(0).to(self.device),
                actions=torch.stack(action_history).unsqueeze(0).to(self.device),
                actions_info=actions_info.unsqueeze(0).to(self.device),
                rewards=torch.stack(reward_history).unsqueeze(0).to(self.device),
                task_ids=task_ids.to(self.device),
                model_params=model_params.unsqueeze(0).to(self.device)
            )

            action = float(out[0][-1][0])
            next_state, reward, _, _, _ = env.step(action) # type: ignore
            state_to_plot.append({
                state_name: next_state[state_name] for state_name in env.state_description.keys()
            })
            action_to_plot.append({k: next_state[k] for k, _ in env.action_description.items()})

            state_history.append(self._get_state_info(state_to_plot[-1])[0])
            action_history.append(self._get_action_info(action_to_plot[-1])[0])
            reward_history.append(torch.tensor([reward], dtype=torch.float32))

        return state_to_plot, action_to_plot
