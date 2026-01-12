#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <filesystem>
#include <numeric>

using VT = std::vector<torch::Tensor>;

struct ActorCriticImpl : torch::nn::Module {
    torch::nn::Sequential head{ nullptr };
    torch::nn::Linear mu{ nullptr }, sigma{ nullptr }, critic{ nullptr };

    ActorCriticImpl(int action_dim) {
        head = register_module("head", torch::nn::Sequential(
            torch::nn::Linear(14, 128),
            torch::nn::Tanh(),
            torch::nn::Linear(128, 256),
            torch::nn::Tanh()
        ));
        mu = register_module("mu", torch::nn::Linear(256, action_dim));
        sigma = register_module("sigma", torch::nn::Linear(256, action_dim));
        critic = register_module("critic", torch::nn::Linear(256, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        if (x.dim() == 1) x = x.unsqueeze(0);
        auto h = head->forward(x);
        return { torch::tanh(mu->forward(h)), torch::softplus(sigma->forward(h)) + 1e-5, critic->forward(h) };
    }

    torch::Tensor get_log_prob(torch::Tensor m, torch::Tensor s, torch::Tensor a) {
        return -0.5 * torch::pow((a - m) / s, 2) - torch::log(s) - 0.5 * std::log(2.0 * M_PI);
    }
};
TORCH_MODULE(ActorCritic);

class PPO2 {
public:
    Env& env;
    std::unordered_map<std::string, float> hp;
    torch::Device device;
    ActorCritic policy{ nullptr };
    std::unique_ptr<torch::optim::Adam> optimizer;
    GraphWindowManager& graph_manager;

    PPO2(Env& env, const std::unordered_map<std::string, float>& hyperparameters,
        torch::Device device, GraphWindowManager& gm,
        std::string actor_model, std::string critic_model)
        : env(env), hp(hyperparameters), device(device), graph_manager(gm) {

        policy = ActorCritic(env.action_space());
        if (!actor_model.empty() && std::filesystem::exists(actor_model)) torch::load(policy, actor_model);
        policy->to(device);
        optimizer = std::make_unique<torch::optim::Adam>(policy->parameters(), torch::optim::AdamOptions(hp.at("lr")));
    }

    void learn(int total_timesteps) {
        int T = (int)hp.at("timesteps_per_batch");
        int epochs = (int)hp.at("n_updates_per_iteration");
        int action_dim = (int)env.action_space();
        torch::Tensor obs = env.reset().to(device);
        int global_step = 0;

        while (global_step < total_timesteps) {
            VT b_states, b_actions, b_logprobs, b_rewards, b_dones, b_values;
            float iteration_reward = 0;

            for (int t = 0; t < T; ++t) {
                torch::NoGradGuard no_grad;
                auto [mu, sigma, val] = policy->forward(obs);

                auto noise = at::normal(0.0, 1.0, { 1, action_dim }, std::nullopt, torch::TensorOptions().device(device));
                auto action = (mu + noise * sigma).squeeze(0);
                auto log_prob = policy->get_log_prob(mu, sigma, action.unsqueeze(0)).sum(-1);

                auto step_result = env.step(action, global_step++);
                float r = std::get<1>(step_result);
                bool d = std::get<2>(step_result);

                b_states.push_back(obs.detach());
                b_actions.push_back(action.detach());
                b_logprobs.push_back(log_prob.detach().squeeze());
                b_values.push_back(val.detach().squeeze());
                b_rewards.push_back(torch::tensor(r, device));
                b_dones.push_back(torch::tensor(d, device));

                iteration_reward += r;
                obs = std::get<0>(step_result).to(device);
                if (d) obs = env.reset().to(device);
            }

            // Record Average Reward for this batch
            graph_manager.Graph("Rewards", iteration_reward / T);

            torch::Tensor next_v;
            {
                torch::NoGradGuard no_grad;
                next_v = std::get<2>(policy->forward(obs)).detach().squeeze();
            }
            b_values.push_back(next_v);
            VT returns = compute_gae(b_rewards, b_dones, b_values, hp.at("gamma"), 0.95f);

            auto s_f = torch::stack(b_states).view({ -1, 14 });
            auto a_f = torch::stack(b_actions).view({ -1, action_dim });
            auto lp_f = torch::stack(b_logprobs).view({ -1 });
            auto ret_f = torch::stack(returns).view({ -1 });
            auto adv_f = ret_f - torch::stack(VT(b_values.begin(), b_values.end() - 1)).view({ -1 });
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8);

            update_network(s_f, a_f, lp_f, ret_f, adv_f, epochs, hp.at("clip"));
        }
    }

private:
    VT compute_gae(VT& r, VT& d, VT& v, float g, float l) {
        VT ret(r.size());
        torch::Tensor gae = torch::zeros({ 1 }, device);
        for (int t = (int)r.size() - 1; t >= 0; --t) {
            auto m = 1.0 - d[t].to(torch::kFloat);
            auto delta = r[t] + g * v[t + 1] * m - v[t];
            gae = delta + g * l * m * gae;
            ret[t] = gae + v[t];
        }
        return ret;
    }

    void update_network(torch::Tensor s, torch::Tensor a, torch::Tensor lp, torch::Tensor ret, torch::Tensor adv, int ep, float cl) {
        for (int i = 0; i < ep; ++i) {
            auto [mu, sigma, val] = policy->forward(s);
            auto curr_lp = policy->get_log_prob(mu, sigma, a).sum(-1);
            auto entropy = (torch::log(sigma) + 0.5 + 0.5 * std::log(2.0 * M_PI)).sum(-1).mean();
            auto ratio = torch::exp(curr_lp - lp);
            auto surr1 = ratio * adv;
            auto surr2 = torch::clamp(ratio, 1.0 - cl, 1.0 + cl) * adv;
            auto loss = -torch::min(surr1, surr2).mean() + 0.5 * torch::mse_loss(val.squeeze(), ret) - 0.01 * entropy;

            optimizer->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(policy->parameters(), 0.5);
            optimizer->step();
        }
    }
};

class PPO2_Eval {
public:
    PPO2_Eval(Env& env, torch::Device& device, std::string actor_model = "")
        : env(env), device(device) {

        act_dim = (int)env.action_space();
        obs_dim = (int)env.observation_space();

        // Initialize architecture
        actor = ActorCritic(act_dim);

        if (!actor_model.empty() && std::filesystem::exists(actor_model)) {
            try {
                torch::load(actor, actor_model);
                std::cout << "[SUCCESS] Loaded actor: " << actor_model << std::endl;
            }
            catch (const c10::Error& e) {
                std::cerr << "[ERROR] Failed to load model: " << e.msg() << std::endl;
                std::cerr << "Ensure the .pt file matches the ActorCriticImpl structure." << std::endl;
            }
        }

        actor->to(device);
        actor->eval();
    }

    void eval_policy(bool render = false, float fixedTimeStepS = 0.0) {
        int ep_num = 0;

        while (true) {
            torch::Tensor obs = env.reset().to(device);
            float ep_ret = 0;
            int ep_len = 0;
            bool done = false;

            while (!done) {
                torch::NoGradGuard no_grad;

                // Get deterministic action (mu)
                auto [action, value] = get_action(obs);

                auto step_result = env.step(action, ep_len);

                obs = std::get<0>(step_result).to(device);
                ep_ret += std::get<1>(step_result);
                done = std::get<2>(step_result);
                ep_len++;

                if (render) {
                    env.render();
                }
                if (fixedTimeStepS > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds((int)(fixedTimeStepS * 1000)));
                }
            }
            log_eval((float)ep_len, ep_ret, ++ep_num);
        }
    }

private:
    std::pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor) {
        // obs_tensor: [14] -> forward expects [1, 14]
        auto [mu, sigma, val] = actor->forward(obs_tensor);

        // Return mean (mu) for evaluation and the value estimate
        return { mu.detach().squeeze(0), val.detach().squeeze(0) };
    }

    void log_eval(float ep_len, float ep_ret, int ep_num) {
        std::cout << ">>> Eval Episode: " << ep_num
            << " | Reward: " << ep_ret
            << " | Steps: " << ep_len << std::endl;
    }

    ActorCritic actor = nullptr;
    torch::Device device;
    Env& env;
    int obs_dim;
    int act_dim;
};