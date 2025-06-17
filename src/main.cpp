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

#include "PPO.h"

void train(
    Env& env,
    const std::unordered_map<std::string, float>& hyperparameters,
    torch::Device& device,
    const std::string& actor_model,
    const std::string& critic_model
) {
    std::cout << "Training" << std::endl;

    PPO model(env, hyperparameters, device, actor_model, critic_model);  // Construct PPO with environment and hyperparameters

    // Train PPO model for a large number of timesteps
    model.learn(200000000);
}

void eval(Env& env, torch::Device& device, const std::string& actor_model) {
    std::cout << "Testing " << actor_model << std::endl;

    PPO_Eval model(env, device, actor_model);

    model.eval_policy();
}

int main(int argc, char* argv[]) {
    // Hyperparameters for PPO (can be customized here)
    std::unordered_map<std::string, float> hyperparameters = {
        {"timesteps_per_batch", 2048},
        {"max_timesteps_per_episode", 200},
        {"gamma", 0.99},
        {"n_updates_per_iteration", 10},
        {"lr", 3e-4},
        {"clip", 0.2},
        {"render", 0},
        {"render_every_i", 10}
    };


    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    torch::Device device = torch::Device(torch::kCPU);

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. GPU will be used.\n";
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);

        if (err != cudaSuccess || deviceCount == 0) {
            std::cout << "Failed to get CUDA device count or no devices found.\n";
            return 1;
        }

        std::cout << "CUDA device count: " << deviceCount << "\n";

        cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, 0);
        if (err == cudaSuccess) {
            std::cout << "Current device name: " << deviceProp.name << "\n";
        }
        else {
            std::cout << "Failed to get device properties.\n";
        }

        //device = torch::Device(torch::kCUDA, 0);
    }
    else {
        std::cout << "CUDA is NOT available. CPU will be used.\n";
    }

    try {
        AgentTargetEnv env(device);
        if (true) {
            train(env, hyperparameters, device, "", "");
        }
        else {
            eval(env, device, "./models/ppo_actor.pt"); // only load the actor model
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred." << std::endl;
    }

    return 0;
}
