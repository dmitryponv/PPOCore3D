#pragma once

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
#include <algorithm>
// Include individual environment headers

// Abstract environment interface
class Env {
public:
    struct Space {
        std::vector<int> shape;
    };

    virtual ~Env() = default;

    // Reset a specific environment and return its observation
    virtual torch::Tensor reset(int index = -1) = 0;

    // Step all environments with a batch of actions
    // Removed std::unordered_map from the tuple
    virtual std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) = 0;

    virtual void render() = 0;

    virtual Space observation_space() const = 0;

    virtual Space action_space() const = 0;
    int GetGridCount() const { return (grid_size * grid_size); } // Made const

    int grid_size = 1;
    float grid_space = 20.0f;
};
