#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <torch/torch.h>
#include <iostream>
#ifdef CUDA
#include <cuda_runtime.h>
#endif
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
#include "DQN.h"
//#include "NEAT.h"
#include "PPO2.h"

#include "envs/AgentTarget3dEnv.h"
#include "envs/PendulumEnv.h"
#include "envs/Pendulum3dEnv.h"
#include "envs/Robot3dEnv.h"
#include "envs/Humanoid3dEnv.h"
#include "envs/HumanStand3dEnv.h"
#include "envs/BasketballEnv.h"
#include "envs/Basketball3dEnv.h"
#include "envs/LunarLanderEnv.h"
#include "envs/DoublePendulumEnv.h"
#include "envs/TriplePendulumEnv.h"


void train(
    Env& env,
    const std::unordered_map<std::string, float>& hyperparameters,
    torch::Device& device,
    GraphWindowManager& graph_manager,
    const std::string& actor_model,
    const std::string& critic_model
) {
    std::cout << "Training" << std::endl;

    PPO model(env, hyperparameters, device, graph_manager, actor_model, critic_model);  // Construct policy with environment and hyperparameters

    // Train PPO model for a large number of timesteps
    model.learn(2000000000);
}

void eval(Env& env, torch::Device& device, const std::string& actor_model, float fixedTimeStepS = 0.0) {
    std::cout << "Testing " << actor_model << std::endl;

    PPO_Eval model(env, device, actor_model);

    model.eval_policy(true, fixedTimeStepS);
}

void animate(Env& env, int anim_skip_steps = 1) {
    std::cout << "Animating with skip steps: " << anim_skip_steps << std::endl;
    env.EnableManipulator();
    // Call the environment's animate function
    env.animate();
}

void ShowConsole() {
    AllocConsole();
    FILE* fp;

    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
    freopen_s(&fp, "CONIN$", "r", stdin);

    SetConsoleTitleA("Debug Console");
    std::ios::sync_with_stdio();
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    ShowConsole();
    GraphWindowManager graph_manager(hInstance, nCmdShow);
    graph_manager.Init();
    // Hyperparameters for PPO (can be customized here)
    std::unordered_map<std::string, float> hyperparameters = {
        {"timesteps_per_batch", 50000},
        {"max_timesteps_per_episode", 25000},
        {"gamma", 0.9},
        {"n_updates_per_iteration", 10},
        {"lr", 3e-4},
        {"clip", 0.2},
    };


    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    torch::Device device = torch::Device(torch::kCPU);

#ifdef CUDA
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
#endif

    try {
        DoublePendulumEnv env(device);
        
        int mode = 0; // 0=train, 1=eval, 2=animate
        int anim_skip_steps = 1; // For animate mode

        float fixedTimeStepS = 1. / 5000.;
        switch (mode) {
            case 0: // train
                //train(env, hyperparameters, device, graph_manager, "./models/ppo_actor.pt", "./models/ppo_critic.pt");
                train(env, hyperparameters, device, graph_manager, "", "");
                break;
            case 2: // animate
                animate(env, anim_skip_steps);
                break;
            case 1: // eval
            default:                
                eval(env, device, "./models/ppo_actor.pt", fixedTimeStepS);
                break;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        throw;
    }

    return 0;
}
