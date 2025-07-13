#pragma once
#include "RobotSimulator.h"
#include "env.h"

class PendulumEnv : public Env {
public:
    // Modified constructor to take grid_size and grid_space
    PendulumEnv(torch::Device& device, int grid_size = 1, float grid_space = 40.0f, const std::string& render_mode = "", float gravity = 10.0f)
        : Env(), // Call base class constructor
        mDevice(device),
        g(gravity),
        render_mode(render_mode)
    {
        this->grid_size = grid_size; // Initialize inherited member
        this->grid_space = grid_space; // Initialize inherited member

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

        states.resize(GetGridCount(), std::vector<float>(2)); // Use GetGridCount()
        last_us.resize(GetGridCount(), 0.0f); // Use GetGridCount()
    }

    torch::Tensor reset(int index = -1) override {
        // Only handles individual environment resets
        if (index < 0 || index >= states.size()) {
            return torch::empty({ 0 }); // Invalid index, return empty tensor
        }

        // Reset a specific environment
        float theta = dist(rng);
        float theta_dot = dist(rng) / 4.0f;
        states[index] = { theta, theta_dot };
        last_us[index] = 0.0f;
        return get_obs(index);
    }

    std::vector<std::tuple<torch::Tensor, float, bool, bool>> step(const std::vector<torch::Tensor>& actions) override {
        std::vector<std::tuple<torch::Tensor, float, bool, bool>> results;
        results.reserve(actions.size());

        for (size_t i = 0; i < actions.size(); ++i) {
            float u = std::clamp(actions[i].item<float>(), -max_torque, max_torque);
            last_us[i] = u;

            float theta = states[i][0];
            float theta_dot = states[i][1];

            float cost = angle_normalize(theta) * angle_normalize(theta)
                + 0.1f * theta_dot * theta_dot
                + 0.001f * u * u;

            float new_theta_dot = theta_dot + (3.0f * g / (2.0f * l) * std::sin(theta) + 3.0f / (m * l * l) * u) * dt;
            new_theta_dot = std::clamp(new_theta_dot, -max_speed, max_speed);
            float new_theta = theta + new_theta_dot * dt;

            states[i] = { new_theta, new_theta_dot };

            results.emplace_back(get_obs(i), -cost, false, false);
        }
        return results;
    }

    void render() override {
        if (render_mode == "human") {
            // Render only the first pendulum for simplicity, or modify to loop and print all.
            // For general rendering of multiple environments, this method would need a major overhaul
            // to display multiple simulation windows or a unified view.
            if (!states.empty()) {
                std::cout << "Angle: " << states[0][0] << ", Angular velocity: " << states[0][1] << ", Torque: " << last_us[0] << "\n";
            }
        }
    }

    void animate(int anim_skip_steps = 1) override {
        // Empty implementation
    }

    Space observation_space() const override {
        return obs_space;
    }

    Space action_space() const override {
        return act_space;
    }

private:
    torch::Device& mDevice;
    float g, m, l, dt;
    float max_speed, max_torque;
    std::string render_mode;
    std::vector<std::vector<float>> states; // State for multiple pendulums
    std::vector<float> last_us; // Last applied torque for multiple pendulums
    Space obs_space, act_space;

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    torch::Tensor get_obs(int index) const {
        float theta = states[index][0];
        float theta_dot = states[index][1];
        return torch::tensor({ std::cos(theta), std::sin(theta), theta_dot }).to(mDevice);
    }

    float angle_normalize(float x) const {
        return std::fmodf(x + M_PI, 2.0f * M_PI) - M_PI;
    }
};
