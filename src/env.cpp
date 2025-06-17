#include "env.h"

AgentTargetEnv::AgentTargetEnv(torch::Device& device) : mDevice(device), dist_x(x_min, x_max), dist_y(y_min, y_max) {
    std::random_device rd;
    rng = std::mt19937(rd());
}

Env::Space AgentTargetEnv::observation_space() const{
    // Observations: agent_x, agent_y, target_x, target_y
    return Space{ {4} };
}

Env::Space AgentTargetEnv::action_space() const{
    // Actions: continuous 2 floats, each in [-1, 1]
    return Space{ {2} };
}

std::pair<torch::Tensor, std::unordered_map<std::string, float>> AgentTargetEnv::reset(){
    agent_pos = { dist_x(rng), dist_y(rng) };
    target_pos = { dist_x(rng), dist_y(rng) };
    return { get_observation(), {} };
}

std::tuple<torch::Tensor, float, bool, bool, std::unordered_map<std::string, float>> AgentTargetEnv::step(const torch::Tensor& action){
    // Clip action to [-1, 1]
    float dx = std::clamp(action[0].item<float>(), -1.0f, 1.0f) * max_step;
    float dy = std::clamp(action[1].item<float>(), -1.0f, 1.0f) * max_step;

    // Update agent position and clip to bounds
    agent_pos[0] = std::clamp(agent_pos[0] + dx, x_min, x_max);
    agent_pos[1] = std::clamp(agent_pos[1] + dy, y_min, y_max);

    // Distance to target
    float dist_x = agent_pos[0] - target_pos[0];
    float dist_y = agent_pos[1] - target_pos[1];
    float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

    // Reward and done conditions
    float reward = -0.01f * distance;
    bool done = false;

    if (distance < 1.0f) {
        reward += 1.0f;
        done = true;
    }

    return { get_observation(), reward, done, false, {} };
}

void AgentTargetEnv::render(){
    printf("Agent: (%.2f, %.2f), Target: (%.2f, %.2f)\n", agent_pos[0], agent_pos[1], target_pos[0], target_pos[1]);
}

torch::Tensor AgentTargetEnv::get_observation() const {
    return torch::tensor({ agent_pos[0], agent_pos[1], target_pos[0], target_pos[1] }).to(mDevice);
}