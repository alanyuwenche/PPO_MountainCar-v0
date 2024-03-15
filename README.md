# PPO_MountainCar-v0
PPO(Proximal Policy Optimization) has become the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance. However, if you use popular RL libraries, such as Stable Baselines3 and Deep RL Zoo, to solve “MountainCar-v0”, you would find that returns always remain at -200.

PPO2 algorithm uses a clipped surrogate objective function to update the policy, which is defined as:
![image](https://github.com/alanyuwenche/PPO_MountainCar-v0/assets/56531349/632da39b-eb95-4f18-b22e-df4144688a97)

The following plot shows one term (i.e., a single timestep) of the clipped surrogate objective function as a function of probability ratio r, for positive advantage and negative advantage. In this environment, we need to push values near the red flat line whether advantage is positive or negative. That is, the parameter K_epochs in my code must be big enough; otherwise the policy network is updated inadequately.
![image](https://github.com/alanyuwenche/PPO_MountainCar-v0/assets/56531349/6b66f99b-1d8e-4f6d-ba27-25bb0dd40cf5)

Finally, you can verify my arguments by changing K_epochs in PPO_MC_Batch.py.
