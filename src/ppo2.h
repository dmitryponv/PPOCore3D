#include <torch/torch.h>
#include <math.h>
#include <random>
#include <fstream>
#include <Eigen/Core>
#include <unordered_map>
#include <string>

#include "env.h"
#include "Grapher.h"

// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
static std::mt19937 re(std::random_device{}());

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module
{
    // Actor.
    torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
    torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.
    torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;

    ActorCriticImpl(int n_in, int n_out, double std)
        : // Actor.
        a_lin1_(torch::nn::Linear(n_in, 16)),
        a_lin2_(torch::nn::Linear(16, 32)),
        a_lin3_(torch::nn::Linear(32, n_out)),
        mu_(torch::full(n_out, 0.)),
        log_std_(torch::full(n_out, std)),

        // Critic
        c_lin1_(torch::nn::Linear(n_in, 16)),
        c_lin2_(torch::nn::Linear(16, 32)),
        c_lin3_(torch::nn::Linear(32, n_out)),
        c_val_(torch::nn::Linear(n_out, 1))
    {
        // Register the modules.
        register_module("a_lin1", a_lin1_);
        register_module("a_lin2", a_lin2_);
        register_module("a_lin3", a_lin3_);
        register_parameter("log_std", log_std_);

        register_module("c_lin1", c_lin1_);
        register_module("c_lin2", c_lin2_);
        register_module("c_lin3", c_lin3_);
        register_module("c_val", c_val_);
    }

    // Forward pass.
    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor>
    {
        // Actor.
        mu_ = torch::relu(a_lin1_->forward(x));
        mu_ = torch::relu(a_lin2_->forward(mu_));
        mu_ = torch::tanh(a_lin3_->forward(mu_));

        // Critic.
        torch::Tensor val = torch::relu(c_lin1_->forward(x));
        val = torch::relu(c_lin2_->forward(val));
        val = torch::tanh(c_lin3_->forward(val));
        val = c_val_->forward(val);

        if (this->is_training())
        {
            torch::NoGradGuard no_grad;
            torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
            return std::make_tuple(action, val);
        }
        else
        {
            return std::make_tuple(mu_, val);
        }
    }

    // Initialize network.
    void normal(double mu, double std)
    {
        torch::NoGradGuard no_grad;

        for (auto& p : this->parameters())
        {
            p.normal_(mu, std);
        }
    }

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution.
        return 0.5 + 0.5 * log(2 * M_PI) + log_std_;
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = (log_std_ + log_std_).exp();
        return -((action - mu_) * (action - mu_)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
    }
};

TORCH_MODULE(ActorCritic);

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO2
{
public:
    PPO2(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, GraphWindowManager& graph_manager, std::string actor_model, std::string critic_model)
        : mEnv(env),
        mHyperparameters(hyperparameters),
        mDevice(device),
        mGraphManager(graph_manager),
        mActorModel(actor_model),
        mCriticModel(critic_model),
        mAc(nullptr),
        mOpt(nullptr)
    {
        // Placeholder for initialization logic using constructor parameters.
        // The actual implementation would go here.
        n_observations = mEnv.observation_space();
        n_actions = mEnv.action_space();
        std::cout << "PPO2 constructed with provided parameters." << std::endl;
    }

    // Main learning function.
    void learn(int total_timesteps)
    {
        // Call reset at the beginning to get the initial state.
        mEnv.reset();

        // Model.
        double std = 2e-2;

        mAc = ActorCritic(n_observations, n_actions, std);
        mAc->to(torch::kF64);
        mAc->normal(0., std);
        mOpt = std::make_unique<torch::optim::Adam>(mAc->parameters(), 1e-3);

        // Training loop parameters
        int n_steps = 2048;
        int n_epochs = 15;
        int mini_batch_size = 512;
        int ppo_epochs = 4;
        double beta = 1e-3;

        VT states;
        VT actions;
        VT rewards;
        VT dones;
        VT log_probs;
        VT returns;
        VT values;

        // Counter.
        int c = 0;

        for (int i = 0; i < total_timesteps; i++)
        {
            // State of env.
            states.push_back(mEnv.get_observation().to(torch::kF64));

            // Play.
            auto av = mAc->forward(states[c]);
            actions.push_back(std::get<0>(av));
            values.push_back(std::get<1>(av));
            log_probs.push_back(mAc->log_prob(actions[c]));

            // Step the environment with the action.
            torch::Tensor next_obs;
            float reward_val;
            bool terminated;
            bool truncated;
            std::tie(next_obs, reward_val, terminated, truncated) = mEnv.step(actions[c], i);

            // New state.
            rewards.push_back(torch::tensor({ reward_val }).to(torch::kF64));
            dones.push_back(torch::tensor({ (int)(terminated || truncated) }).to(torch::kF64));

            c++;

            // Check if the episode is done
            if (terminated || truncated) {
                mEnv.reset();
            }

            // Update.
            if (c % n_steps == 0)
            {
                values.push_back(std::get<1>(mAc->forward(states[c - 1])));

                returns = returns_gae(rewards, dones, values, .99, .95);

                torch::Tensor t_log_probs = torch::cat(log_probs).detach();
                torch::Tensor t_returns = torch::cat(returns).detach();
                torch::Tensor t_values = torch::cat(values).detach();
                torch::Tensor t_states = torch::cat(states);
                torch::Tensor t_actions = torch::cat(actions);
                torch::Tensor t_advantages = t_returns - t_values.slice(0, 0, n_steps);

                update_network(t_states, t_actions, t_log_probs, t_returns, t_advantages, n_steps, ppo_epochs, mini_batch_size, beta);

                double avg_reward = average(rewards);
                printf("Ep Finished, Reward: %lf.\n", avg_reward);
                mGraphManager.Graph("Rewards", avg_reward);

                c = 0;
                states.clear();
                actions.clear();
                rewards.clear();
                dones.clear();
                log_probs.clear();
                returns.clear();
                values.clear();
            }
        }
    }

private:
    Env& mEnv;
    const std::unordered_map<std::string, float>& mHyperparameters;
    torch::Device& mDevice;
    GraphWindowManager& mGraphManager;
    std::string mActorModel;
    std::string mCriticModel;
    ActorCritic mAc;
    std::unique_ptr<torch::optim::Adam> mOpt;
    int n_actions = 0;
    int n_observations = 0;

    // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
    auto returns_gae(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
    {
        torch::Tensor gae = torch::zeros({ 1 }, torch::kFloat64);
        VT returns(rewards.size(), torch::zeros({ 1 }, torch::kFloat64));

        for (int i = rewards.size(); i-- > 0;)
        {
            // Advantage.
            auto delta = rewards[i] + gamma * vals[i + 1] * (1 - dones[i]) - vals[i];
            gae = delta + gamma * lambda * (1 - dones[i]) * gae;
            returns[i] = gae + vals[i];
        }
        return returns;
    }

    // Update the network.
    void update_network(torch::Tensor& states,
        torch::Tensor& actions,
        torch::Tensor& log_probs,
        torch::Tensor& returns,
        torch::Tensor& advantages,
        int steps, int epochs, int mini_batch_size, double beta, double clip_param = .2)
    {
        for (int e = 0; e < epochs; e++)
        {
            // Generate random indices.
            torch::Tensor cpy_sta = torch::zeros({ mini_batch_size, n_observations }, states.options());
            torch::Tensor cpy_act = torch::zeros({ mini_batch_size, n_actions }, actions.options());
            torch::Tensor cpy_log = torch::zeros({ mini_batch_size, n_actions }, log_probs.options());
            torch::Tensor cpy_ret = torch::zeros({ mini_batch_size, 1 }, returns.options());
            torch::Tensor cpy_adv = torch::zeros({ mini_batch_size, 1 }, advantages.options());

            for (int b = 0; b < mini_batch_size; b++) {
                int idx = std::uniform_int_distribution<int>(0, steps - 1)(re);
                cpy_sta[b] = states[idx];
                cpy_act[b] = actions[idx];
                cpy_log[b] = log_probs[idx];
                cpy_ret[b] = returns[idx];
                cpy_adv[b] = advantages[idx];
            }

            auto av = mAc->forward(cpy_sta); // action value pairs
            auto entropy = mAc->entropy().mean();
            auto new_log_prob = mAc->log_prob(cpy_act);

            auto old_log_prob = cpy_log;
            auto ratio = (new_log_prob - old_log_prob).exp();
            auto surr1 = ratio * cpy_adv;
            auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param) * cpy_adv;

            auto val = std::get<1>(av);
            auto actor_loss = -torch::min(surr1, surr2).mean();
            auto critic_loss = (cpy_ret - val).pow(2).mean();

            auto loss = 0.5 * critic_loss + actor_loss - beta * entropy;

            mOpt->zero_grad();
            loss.backward();
            mOpt->step();
        }
    }

    double average(const std::vector<torch::Tensor>& tensors) {
        if (tensors.empty()) {
            return 0.0f;
        }

        double sum = 0.0f;
        for (const auto& tensor : tensors) {
            // We use .item<float>() to extract the single float value from the tensor.
            sum += tensor.item<double>();
        }

        return sum / tensors.size();
    }
};