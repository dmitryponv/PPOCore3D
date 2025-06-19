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

#include "env.h"



//#define DEBUG_TENSNORS
//#define LIMIT_ACTION_SPACE

void print_tensor_inline(const std::string& name, const torch::Tensor& t, int precision = 4, int max_elements = 10) {
#ifdef DEBUG_TENSNORS
    torch::Tensor flat = t.flatten().cpu();
    std::cout << name << "=tensor([";
    int64_t size = flat.size(0);
    std::cout << std::fixed << std::setprecision(precision);
    for (int64_t i = 0; i < std::min<int64_t>(size, max_elements / 2); ++i) {
        std::cout << flat[i].item<double>() << ", ";
    }
    if (size > max_elements) {
        std::cout << "...";
        for (int64_t i = size - max_elements / 2; i < size; ++i) {
            std::cout << ", " << flat[i].item<double>();
        }
    }
    std::cout << "])" << std::endl << std::endl;
#endif
}
#ifdef  LIMIT_ACTION_SPACE
std::pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor, FeedForwardNN& actor, torch::Tensor& cov_mat) {
    // Query the actor network for a mean action
    torch::Tensor mean = actor->forward(obs_tensor);

    // Create a distribution with the mean and covariance
    auto dist = MultivariateNormal(mean, cov_mat);

    // Sample an action (pre-squash)
    torch::Tensor raw_action = dist.sample();

    // Squash the action to [-1, 1] range
    torch::Tensor action_tensor = torch::tanh(raw_action);

    // Log prob of the unsquashed action
    torch::Tensor log_prob = dist.log_prob(raw_action);

    // Tanh correction: subtract log(det(Jacobian)) of the transformation
    // For tanh: log(1 - tanh(x)^2) = log(1 - action^2)
    torch::Tensor correction = torch::log(1 - action_tensor.pow(2) + 1e-6).sum(-1);

    // Adjusted log_prob
    log_prob = log_prob - correction;

    // Optional: print tensors
    print_tensor_inline("obs_tensor", obs_tensor);
    print_tensor_inline("mean", mean);
    print_tensor_inline("raw_action", raw_action);
    print_tensor_inline("action_tensor", action_tensor);
    print_tensor_inline("corrected_log_prob", log_prob);

    return { action_tensor, log_prob.detach() };
}
#else
std::pair<torch::Tensor, torch::Tensor> get_action(const torch::Tensor& obs_tensor, FeedForwardNN& actor, torch::Tensor& cov_mat) {
    // Query the actor network for a mean action
    torch::Tensor mean = actor->forward(obs_tensor);

    // Create a distribution with the mean action and std from the covariance matrix
    auto dist = MultivariateNormal(mean, cov_mat);

    // Sample an action from the distribution
    torch::Tensor action_tensor = dist.sample();// torch::tanh(dist.sample());

    // Compute the log probability with correction for tanh
    torch::Tensor log_prob = dist.log_prob(action_tensor);

    print_tensor_inline("obs_tensor", obs_tensor);
    print_tensor_inline("mean", mean);
    print_tensor_inline("action_tensor", action_tensor);
    print_tensor_inline("log_prob_tensor", log_prob);

    return { action_tensor, log_prob.detach() };
}
#endif //  LIMIT_ACTION_SPACE


class MultivariateNormal {
public:
    torch::Tensor loc;
    torch::Tensor _unbroadcasted_scale_tril;
    torch::Tensor scale_tril_;
    torch::Tensor covariance_matrix_;
    torch::Tensor precision_matrix_;
    std::vector<int64_t> batch_shape;
    std::vector<int64_t> event_shape;

    MultivariateNormal(const torch::Tensor& loc,
        const torch::optional<torch::Tensor>& covariance_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& precision_matrix = torch::nullopt,
        const torch::optional<torch::Tensor>& scale_tril = torch::nullopt);

    torch::Tensor scale_tril() const;

    torch::Tensor covariance_matrix() const;

    torch::Tensor precision_matrix() const;

    torch::Tensor mean() const;

    torch::Tensor mode() const;

    torch::Tensor variance() const;

    torch::Tensor sample(const std::vector<int64_t>& sample_shape = {}) const;

    torch::Tensor log_prob(const torch::Tensor& value) const;

    torch::Tensor entropy() const;

private:
    static torch::Tensor batch_mahalanobis(const torch::Tensor& L, const torch::Tensor& diff);

    static std::vector<int64_t> broadcast_shapes(std::vector<int64_t> a, std::vector<int64_t> b, int a_end = 0, int b_end = 0);
};

struct FeedForwardNNImpl : torch::nn::Module {
    torch::nn::Linear layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr };

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

    std::pair<torch::Tensor, torch::Tensor> evaluate(const torch::Tensor& batch_obs, const torch::Tensor& batch_acts);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rollout_train();

    Env& env;
    int obs_dim;
    int act_dim;
    float lr;

    torch::Tensor cov_var;
    torch::Tensor cov_mat;

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

    void eval_policy(bool render = false);

private:

    void log_eval(float ep_len, float ep_ret, int ep_num);

    FeedForwardNN actor = nullptr;
    torch::Device device;
    Env& env;
    int obs_dim;
    int act_dim;
    torch::Tensor cov_mat;
};
