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


// Abstract environment interface
class Env {
public:
    struct Space {
        std::vector<int> shape;
    };

    virtual ~Env() = default;

    // Reset the environment and return initial observation
    virtual std::pair<torch::Tensor, std::unordered_map<std::string, float>> reset() = 0;

    // Step the environment with given action returns: observation, reward, terminated flag, truncated flag, extra info
    virtual std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> step(const torch::Tensor& action) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
};


class AgentTargetEnv : public Env {
public:
    AgentTargetEnv(torch::Device& device);

    Space observation_space() const override;

    Space action_space() const override;

    std::pair<torch::Tensor, std::unordered_map<std::string, float>> reset() override;

    std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> step(const torch::Tensor& action) override;

    void render() override;

private:
    torch::Device& mDevice;
    float x_min = 0.0f, x_max = 10.0f;
    float y_min = 0.0f, y_max = 10.0f;
    float max_step = 0.5f;  // max movement per step in each axis

    std::vector<float> agent_pos;  // {x, y}
    std::vector<float> target_pos; // {x, y}

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist_x;
    std::uniform_real_distribution<float> dist_y;

    torch::Tensor get_observation() const;
};