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

#include "RobotSimulator.h"

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


class RobotEnv : public Env {
public:
    RobotEnv(torch::Device& device);

    Space observation_space() const override;

    Space action_space() const override;

    std::pair<torch::Tensor, std::unordered_map<std::string, float>> reset() override;

    std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> step(const torch::Tensor& action) override;

    void render() override;

private:
    torch::Device& mDevice;
    b3RobotSimulatorClientAPI* sim;
    int minitaurUid = -1;
    int numJoints = 0;
    std::vector<int> validTorqueJoints;
    torch::Tensor get_observation() const;
};