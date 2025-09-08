#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_map>
#include <tuple>
#include <optional>
#include <filesystem>
#include <iomanip>

#include <torch/torch.h>

#include "env.h"
#include "Grapher.h"

using namespace std;

//class NormalMultivariate {
//    torch::Tensor mean, stddev, var, log_std;
//    torch::Device& device;
//public:
//    NormalMultivariate(const torch::Tensor& mean, const torch::Tensor& std, torch::Device& device)
//        : mean(mean), stddev(std), var(std* std), log_std(std.log()), device(device) {
//    }
//
//    torch::Tensor sample() {
//        auto eps = torch::randn_like(mean).to(device);
//        return this->mean + eps * this->stddev;
//    }
//
//    torch::Tensor log_prob(const torch::Tensor& value) {
//        const double log_sqrt_2pi = 0.9189385332046727; // precomputed log(sqrt(2*pi))
//        return -(value - this->mean) * (value - this->mean) / (2 * this->var) - this->log_std - log_sqrt_2pi;
//    }
//};


// Main Network architecture
struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr }, layer4{ nullptr };

    QNetworkImpl(int in_dim, int act_dim, torch::Device& device);
    torch::Tensor forward(torch::Tensor obs);
};
TORCH_MODULE(QNetwork);


// The main agent class
class DQN {
public:
    DQN(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, GraphWindowManager& graph_manager, string qnetwork_path = "", string target_network_path = "");
    void learn(int total_timesteps);
    torch::Tensor get_action(const torch::Tensor& obs);

private:
    void _init_hyperparameters(const unordered_map<string, float>& hyperparameters);
    void _log_train();
    // MODIFIED: update_q_network now takes rollout data as arguments
    void update_q_network(
        const std::vector<torch::Tensor>& roll_obs,
        const std::vector<torch::Tensor>& roll_actions,
        const std::vector<float>& roll_rewards,
        const std::vector<bool>& roll_dones,
        const torch::Tensor& last_obs
    );


    Env& env;
    torch::Device& device;
    GraphWindowManager& graph_manager;

    QNetwork q_network = nullptr;
    QNetwork target_network = nullptr;
    std::unique_ptr<torch::optim::Adam> optim;

    int obs_dim;
    int act_dim;

    // Hyperparameters
    float lr, gamma;
    // MODIFIED: Replay buffer size is replaced by rollout_length
    int rollout_length;
    int target_update_freq;
    int max_timesteps_per_episode;
    int log_freq;
    optional<int> seed;

    // MODIFIED: Replay buffer is no longer needed for this approach.
    // ReplayBuffer replay_buffer;

    unordered_map<string, variant<int, float, long long, vector<int>, vector<float>>> logger;

    mt19937 gen;




    ///TEMP FUNCTIONS
    void VerifyNetworks(const std::vector<torch::Tensor>& roll_obs) {
        std::cout << "\n--- Verifying Networks ---" << std::endl;

        try {
            // Convert rollout data to tensors
            auto states = torch::stack(roll_obs).to(device);

            torch::NoGradGuard no_grad; // No gradients needed for verification

            // --- Run observations through both networks ---
            auto q_network_output = q_network->forward(states);
            auto target_network_output = target_network->forward(states);

            // --- Calculate MSE Loss between the raw outputs ---
            torch::Tensor loss = torch::mse_loss(q_network_output, target_network_output);

            std::cout << "Q-Network Output Shape: " << q_network_output.sizes() << std::endl;
            std::cout << "Target-Network Output Shape: " << target_network_output.sizes() << std::endl;
            std::cout << "Verification Loss (MSE) between Q and Target network raw outputs: " << loss.item<float>() << std::endl;

            // Optional: Print a few values to inspect
            std::cout << "\nSample Q-Network Output:\n" << q_network_output.slice(0, 0, 5) << std::endl;
            std::cout << "\nSample Target Network Output:\n" << target_network_output.slice(0, 0, 5) << std::endl;

        }
        catch (const std::exception& e) {
            std::cerr << "Exception in VerifyNetworks: " << e.what() << std::endl;
        }
        std::cout << "--- Verification Complete ---" << std::endl;
    }
};

class DQN_Eval {
public:
    DQN_Eval(Env& env, torch::Device& device, string model_path);
    void eval_policy(bool render, float fixed_time_seconds);

private:
    void log_eval(float ep_len, float ep_ret, int ep_num);
    torch::Tensor get_action(const torch::Tensor& obs);

    Env& env;
    torch::Device& device;
    QNetwork q_network = nullptr;
    int obs_dim;
    int act_dim;
};

// ===================================
// Function Definitions
// ===================================

QNetworkImpl::QNetworkImpl(int in_dim, int act_dim, torch::Device& device) {
    try {
        layer1 = register_module("layer1", torch::nn::Linear(in_dim, 256));
        layer2 = register_module("layer2", torch::nn::Linear(256, 256));
        layer3 = register_module("layer3", torch::nn::Linear(256, 128));
        layer4 = register_module("layer4", torch::nn::Linear(128, act_dim * 2));

        layer1->to(device);
        layer2->to(device);
        layer3->to(device);
        layer4->to(device);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in QNetworkImpl constructor: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor QNetworkImpl::forward(torch::Tensor obs) {
    try {
        auto activation1 = torch::relu(layer1(obs));
        auto activation2 = torch::relu(layer2(activation1));
        auto activation3 = torch::relu(layer3(activation2));
        return layer4(activation3);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in forward: " << e.what() << std::endl;
        throw;
    }
}

DQN::DQN(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, GraphWindowManager& graph_manager, string qnetwork_path, string target_network_path)
    : env(env), device(device), graph_manager(graph_manager), gen(42) {
    try {
        obs_dim = env.observation_space();
        act_dim = env.action_space();

        q_network = QNetwork(obs_dim, act_dim, device);
        target_network = QNetwork(obs_dim, act_dim, device);

        target_network->eval();
        target_network->to(device);

        if (!qnetwork_path.empty()) {
            cout << "Loading in " << qnetwork_path << "..." << endl;
            torch::load(q_network, qnetwork_path);
            cout << "Successfully loaded." << endl;
        }
        if (!target_network_path.empty()) {
            cout << "Loading in " << target_network_path << "..." << endl;
            torch::load(target_network, target_network_path);
            cout << "Successfully loaded." << endl;
        }
        if (qnetwork_path.empty() && target_network_path.empty()) {
            cout << "Training from scratch." << endl;
        }

        _init_hyperparameters(hyperparameters);

        optim = std::make_unique<torch::optim::Adam>(q_network->parameters(), torch::optim::AdamOptions(lr));

        logger["delta_t"] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        logger["t_so_far"] = 0;
        logger["ep_rets"] = vector<float>{};
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DQN constructor: " << e.what() << std::endl;
        throw;
    }
}


// MODIFIED: The learn loop is now structured around collecting rollouts.
void DQN::learn(int total_timesteps) {
    try {
        int t_so_far = 0;
        int network_updates = 0;

        torch::Tensor obs = env.reset();
        int ep_len = 0;
        float ep_ret = 0;

        while (t_so_far < total_timesteps) {
            // --- Rollout Collection Phase ---
            vector<torch::Tensor> roll_obs, roll_actions;
            vector<float> roll_rewards;
            vector<bool> roll_dones;

            for (int i = 0; i < rollout_length; ++i) {
                torch::Tensor action = get_action(obs);

                auto step_results = env.step(action.clone(), t_so_far);
                auto& [next_obs, reward, terminated, truncated] = step_results;
                bool done = terminated || truncated;

                // Store the experience
                roll_obs.push_back(obs);
                roll_actions.push_back(action);
                roll_rewards.push_back(reward);
                roll_dones.push_back(done);

                ep_ret += reward;
                ep_len += 1;
                t_so_far += 1;
                obs = next_obs;

                if (done || ep_len >= max_timesteps_per_episode) {
                    std::get<vector<float>>(logger["ep_rets"]).emplace_back(ep_ret);
                    obs = env.reset();
                    ep_ret = 0;
                    ep_len = 0;
                }
            }

            // --- Learning Phase ---
            update_q_network(roll_obs, roll_actions, roll_rewards, roll_dones, obs);
            network_updates++;

            // Update the target network periodically
            if (network_updates > 0 && network_updates % target_update_freq == 0) {
                cout << "Target Network updated." << endl;
                torch::NoGradGuard no_grad;
                for (const auto& pair : q_network->named_parameters()) {
                    target_network->named_parameters()[pair.key()].copy_(pair.value());
                }
            }

            // Logging
            if (t_so_far >= std::get<int>(logger["t_so_far"]) + log_freq) {
                logger["t_so_far"] = t_so_far;
                _log_train();
                if (!std::get<vector<float>>(logger["ep_rets"]).empty()) {
                    graph_manager.Graph("Rewards", std::get<vector<float>>(logger["ep_rets"]).back());
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in learn: " << e.what() << std::endl;
        throw;
    }
}

// MODIFIED: The update logic now processes a batch of rollout data.
void DQN::update_q_network(
    const std::vector<torch::Tensor>& roll_obs,
    const std::vector<torch::Tensor>& roll_actions,
    const std::vector<float>& roll_rewards,
    const std::vector<bool>& roll_dones,
    const torch::Tensor& last_obs) {
    try {
        // Convert rollout data to tensors
        auto states = torch::stack(roll_obs).to(device);
        auto actions = torch::stack(roll_actions).to(device);
        auto rewards = torch::tensor(roll_rewards, torch::kFloat).to(device).unsqueeze(-1);

        std::vector<float> dones_float;
        for (bool d : roll_dones) { dones_float.push_back(d ? 1.0f : 0.0f); }
        auto dones = torch::tensor(dones_float, torch::kFloat).to(device).unsqueeze(-1);


        std::cout << "states:\n" << states << std::endl;
        std::cout << "actions:\n" << actions << std::endl;
        std::cout << "rewards:\n" << rewards << std::endl;

        // --- Calculate Target Values ---
        torch::Tensor target_q_values;
        {
            torch::NoGradGuard no_grad;

            // CORRECTED: Build the next_states tensor more explicitly.
            // It consists of all observations from the rollout except the first one,
            // plus the final observation after the rollout ended.
            std::vector<torch::Tensor> next_obs_vec;
            next_obs_vec.insert(next_obs_vec.end(), roll_obs.begin() + 1, roll_obs.end());
            next_obs_vec.push_back(last_obs);
            auto next_states = torch::stack(next_obs_vec).to(device);

            // Get the target network's predicted mean action for the next states.
            auto target_output = target_network->forward(next_states);
            auto target_mean = torch::chunk(target_output, 2, -1)[0];

            // CORRECTED: Removed the erroneous slice that was causing the dimension mismatch.
            // The target_mean tensor now correctly has a size of 2048.
            target_q_values = rewards + gamma * (1.0 - dones) * target_mean;
        }

        // --- Calculate Loss ---
        // As noted before, this loss is an approximation for this network structure.
        // It encourages the policy's mean action to move toward the calculated target.
        auto current_output = q_network->forward(states);
        auto current_mean = torch::chunk(current_output, 2, -1)[0];

        // The tensors `current_mean` and `target_q_values` now both have a size of 2048,
        // so the loss calculation will succeed.
        torch::Tensor loss = torch::mse_loss(current_mean, target_q_values.detach());
        //std::cout << "loss:\n" << loss << std::endl;
        optim->zero_grad();
        loss.backward();
        optim->step();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in update_q_network: " << e.what() << std::endl;
    }
}

torch::Tensor DQN::get_action(const torch::Tensor& obs) {
    try {
        torch::NoGradGuard no_grad;
        auto network_output = q_network->forward(obs.to(device));
        auto chunks = torch::chunk(network_output, 2, -1);
        auto mean = chunks[0];
        auto log_std = chunks[1];
        auto stddev = torch::exp(log_std);

        NormalMultivariate dist(mean, stddev, device);
        return dist.sample();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DQN::get_action: " << e.what() << std::endl;
        return torch::zeros({ this->act_dim }, torch::TensorOptions().device(device));
    }
}

void DQN::_init_hyperparameters(const unordered_map<string, float>& hyperparameters) {
    try {
        lr = hyperparameters.count("lr") ? hyperparameters.at("lr") : 0.0005f;
        gamma = hyperparameters.count("gamma") ? hyperparameters.at("gamma") : 0.99f;
        // MODIFIED: Use rollout_length instead of replay_buffer_size
        rollout_length = hyperparameters.count("rollout_length") ? static_cast<int>(hyperparameters.at("rollout_length")) : 10;
        target_update_freq = hyperparameters.count("target_update_freq") ? static_cast<int>(hyperparameters.at("target_update_freq")) : 1;
        max_timesteps_per_episode = hyperparameters.count("max_timesteps_per_episode") ? static_cast<int>(hyperparameters.at("max_timesteps_per_episode")) : 500;
        log_freq = hyperparameters.count("log_freq") ? static_cast<int>(hyperparameters.at("log_freq")) : 2048;
        seed = hyperparameters.count("seed") ? optional<int>(static_cast<int>(hyperparameters.at("seed"))) : optional<int>(42);

        if (seed.has_value()) {
            torch::manual_seed(seed.value());
            cout << "Successfully set seed to " << seed.value() << endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in _init_hyperparameters: " << e.what() << std::endl;
        throw;
    }
}

void DQN::_log_train() {
    long long prev_delta_t = std::get<long long>(logger["delta_t"]);
    logger["delta_t"] = chrono::duration_cast<chrono::nanoseconds>(
        chrono::high_resolution_clock::now().time_since_epoch()).count();
    float delta_t_sec = (std::get<long long>(logger["delta_t"]) - prev_delta_t) / 1e9f;
    int t_so_far = std::get<int>(logger["t_so_far"]);
    vector<float>& ep_rets = std::get<vector<float>>(logger["ep_rets"]);

    float avg_ep_rews = 0.0f;
    if (!ep_rets.empty()) {
        int num_episodes = std::min(100, static_cast<int>(ep_rets.size()));
        avg_ep_rews = std::accumulate(ep_rets.end() - num_episodes, ep_rets.end(), 0.0f) / num_episodes;
    }

    stringstream delta_t_ss, avg_ep_rews_ss;
    delta_t_ss << fixed << setprecision(2) << delta_t_sec;
    avg_ep_rews_ss << fixed << setprecision(2) << avg_ep_rews;

    cout << endl;
    cout << "-------------------- Timestep #" << t_so_far << " --------------------" << endl;
    cout << "Average Episodic Return (last " << std::min(100, static_cast<int>(ep_rets.size())) << " eps): " << avg_ep_rews_ss.str() << endl;
    cout << "Timesteps So Far: " << t_so_far << endl;
    cout << "Iteration took: " << delta_t_ss.str() << " secs" << endl;
    cout << "------------------------------------------------------" << endl;
    cout << endl;
}


DQN_Eval::DQN_Eval(Env& env, torch::Device& device, string model_path)
    : env(env), device(device) {
    try {
        if (model_path.empty()) {
            cerr << "No model file specified. Exiting." << endl;
            exit(0);
        }
        obs_dim = env.observation_space();
        act_dim = env.action_space();

        q_network = QNetwork(obs_dim, act_dim, device);
        torch::load(q_network, model_path);
        q_network->eval();
        q_network->to(device);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in DQN_Eval constructor: " << e.what() << std::endl;
        throw;
    }
}

void DQN_Eval::eval_policy(bool render, float fixed_time_seconds) {
    try {
        int ep_num = 0;
        auto start_time = chrono::high_resolution_clock::now();
        auto end_time = start_time + chrono::milliseconds(static_cast<long long>(fixed_time_seconds * 1000));

        while (chrono::high_resolution_clock::now() < end_time) {
            auto obs_tensor = env.reset();
            bool done = false;
            float ep_ret = 0.0f;
            int ep_len = 0;

            while (!done && chrono::high_resolution_clock::now() < end_time) {
                torch::Tensor action = get_action(obs_tensor);

                auto step_results = env.step(action, ep_len);
                auto& [next_obs, rew, terminated, truncated] = step_results;

                obs_tensor = next_obs;
                ep_ret += rew;
                ep_len += 1;
                done = terminated || truncated;

                if (render) {
                    env.render();
                }
            }
            log_eval(static_cast<float>(ep_len), ep_ret, ep_num);
            ep_num++;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in eval_policy: " << e.what() << std::endl;
        throw;
    }
}

torch::Tensor DQN_Eval::get_action(const torch::Tensor& obs) {
    torch::NoGradGuard no_grad;
    auto network_output = q_network->forward(obs.to(device));
    auto mean = torch::chunk(network_output, 2, -1)[0];
    return mean;
}

void DQN_Eval::log_eval(float ep_len, float ep_ret, int ep_num) {
    ep_len = std::round(ep_len * 100.0f) / 100.0f;
    ep_ret = std::round(ep_ret * 100.0f) / 100.0f;

    std::cout << std::endl;
    std::cout << "-------------------- Episode #" << ep_num << " --------------------" << std::endl;
    std::cout << "Episodic Length: " << ep_len << std::endl;
    std::cout << "Episodic Return: " << ep_ret << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << std::endl;
    std::cout.flush();
}