def validate_analytical_step(env: RBCEconomyWithPolicyEnv, num_steps: int = 10):
    env.reset()

    results = []
    for step in range(num_steps):
        action = env.action_space.sample()
        sim_state, sim_reward, _, _, sim_info = env.step(action)
        env.reset()  # Reset to ensure the same starting state
        env.step(action)  # Re-apply action
        ana_state, ana_reward, _, _, ana_info = env.analytical_step()
        result = {
            "Step": step + 1,
            "Simulated_Reward": sim_reward,
            "Analytical_Reward": ana_reward,
            "Reward_Diff": abs(sim_reward - ana_reward),
            "Simulated_Output": sim_info["output"],
            "Analytical_Output": ana_info["output"],
            "Output_Diff": abs(sim_info["output"] - ana_info["output"]),
        }
        results.append(result)

    results_df = pd.DataFrame(results)

    print("Validation Results:")
    print(results_df)
    print("\nSummary:")
    print(results_df.describe())

    return results_df



# env = RBCEconomyWithPolicyEnv()
# validation_results = validate_analytical_step(env)
