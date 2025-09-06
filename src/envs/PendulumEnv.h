#pragma once
#include "RobotSimulator.h"
#include "../env.h"

class PendulumEnv : public Env {
public:
    // Modified constructor to take grid_size and grid_space
    PendulumEnv(torch::Device& device, float gravity = 10.0f)
        : Env(device), // Call base class constructor
        g(gravity)
    {
        max_speed = 8.0f;
        max_torque = 2.0f;
        dt = 0.05f;
        m = 1.0f;
        l = 1.0f;

        obs_space.shape = { 3 };
        act_space.shape = { 1 };

        std::random_device rd;
        rng = std::mt19937(rd());
        dist = std::uniform_real_distribution<float>(-3.14f, 3.14f);
    }

    torch::Tensor reset() override {
        // Reset a specific environment
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        states = { theta, theta_dot };
        last_us = 0.0f;
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        float u = std::clamp(actions.item<float>(), -max_torque, max_torque);
        last_us = u;

        float theta = states[0];
        float theta_dot = states[1];

        float cost = angle_normalize(theta) * angle_normalize(theta)
            + 0.1f * theta_dot * theta_dot
            + 0.001f * u * u;

        float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * std::sin(theta) + 3.0f / (m * l * l) * u) * dt;
        new_theta_dot = std::clamp(new_theta_dot, -max_speed, max_speed);
        float new_theta = theta + new_theta_dot * dt;

        states = { new_theta, new_theta_dot };
        
        return { get_observation(), -cost, false, false };
    }

    void render() override {        
        if (!states.empty()) {
            std::cout << "Angle: " << states[0] << ", Angular velocity: " << states[1] << ", Torque: " << last_us << "\n";
        }
    }

    void animate() override {
        // Empty implementation
    }

    Space observation_space() const override {
        return obs_space;
    }

    Space action_space() const override {
        return act_space;
    }

    void EnableManipulator() override
    {

    }

private:
    float g, m, l, dt;
    float max_speed, max_torque;
    std::vector<float> states; // State for multiple pendulums
    float last_us = 0.0f; // Last applied torque for multiple pendulums
    Space obs_space, act_space;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    torch::Tensor get_observation() override {
        float theta = states[0];
        float theta_dot = states[1];
        return torch::tensor({ std::cos(theta), std::sin(theta), theta_dot }).to(mDevice);
    }

    float angle_normalize(float x) const {
        return std::fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }
};
