#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>
#include <chrono>
#include <unordered_map>

#include "../Utils/b3Clock.h"
#include "env.h"

class NormalMultivariate {
    torch::Tensor mean, stddev, var, log_std;
    torch::Device& device;
public:
    NormalMultivariate(const torch::Tensor& mean, const torch::Tensor& std, torch::Device& device)
        : mean(mean), stddev(std), var(std* std), log_std(std.log()), device(device) {
    }

    torch::Tensor sample() {
        auto eps = torch::randn_like(mean).to(device);
        return this->mean + eps * this->stddev;
    }

    torch::Tensor log_prob(const torch::Tensor& value) {
        const double log_sqrt_2pi = 0.9189385332046727; // precomputed log(sqrt(2*pi))
        return -(value - this->mean) * (value - this->mean) / (2 * this->var) - this->log_std - log_sqrt_2pi;
    }
};

struct FeedForwardNNImpl : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr }, layer4{ nullptr };

    FeedForwardNNImpl(int in_dim, int out_dim, torch::Device& device);

    torch::Tensor forward(torch::Tensor obs);
};
TORCH_MODULE(FeedForwardNN);

class PPO {
public:

    PPO(Env& env, const std::unordered_map<std::string, float>& hyperparameters, torch::Device& device, std::string actor_model = "", std::string critic_model = "");

    void learn(int total_timesteps);

private:
    void _init_hyperparameters(const std::unordered_map<std::string, float>& hyperparameters);

    void _log_train();

    torch::Tensor compute_rtgs(const std::vector<std::vector<float>>& batch_rewards);

    std::pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor);

    std::pair<torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rollout_train();

    Env& env;
    int obs_dim;
    int act_dim;
    float lr;

    torch::Tensor std_dev;

    FeedForwardNN actor = nullptr;
    FeedForwardNN critic = nullptr;
    torch::Device device;

    std::unique_ptr<torch::optim::Adam> actor_optim;
    std::unique_ptr<torch::optim::Adam> critic_optim;

    using LoggerValue = std::variant<
        std::string,
        float,
        int,
        std::vector<float>,
        std::vector<int>,
        std::vector<torch::Tensor>,
        std::vector<std::vector<float>>,
        long long,
        torch::Tensor
    >;

    std::unordered_map<std::string, LoggerValue> logger;
    std::vector<int> batch_lengths;
    std::vector<float> batch_rewards;

    // Algorithm hyperparameters
    int timesteps_per_batch;
    int max_timesteps_per_episode;
    int n_updates_per_iteration;
    float gamma;
    float clip;

    // Miscellaneous parameters
    bool render;
    int render_every_i;
    int save_freq;
    std::optional<int> seed;
};

class PPO_Eval {
public:
    PPO_Eval(Env& env, torch::Device& device, std::string actor_model = "");

    void eval_policy(bool render = false, float fixedTimeStepS = 0.0);

private:

    std::pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor);

    void log_eval(float ep_len, float ep_ret, int ep_num);

    FeedForwardNN actor = nullptr;
    torch::Device device;
    Env& env;
    int obs_dim;
    int act_dim;
    torch::Tensor std_dev;
};
