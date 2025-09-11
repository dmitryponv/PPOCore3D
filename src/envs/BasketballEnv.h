#include "../env.h"
#include <torch/torch.h>
#include <vector>
#include <stdexcept>
#include <string>

class Basketball1dEnv : public Env {
private:
    float position;
    float velocity;
    float mass;
    float timestep;
    float gravity;
    float target_position;
    float force_multiplier;

public:
    Basketball1dEnv(torch::Device& device)
        : Env(device),
        position(0.0f),
        velocity(0.0f),
        mass(0.1f),
        timestep(1.0f / 240.0f),
        gravity(0.0f),
        target_position(3.0f),
        force_multiplier(10.0f) {
    }

    torch::Tensor reset() override {
        position = 0.0f;
        velocity = 0.0f;
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        // Only apply force in one dimension (x)
        float force_x = actions[0].item<float>() * force_multiplier;
        float total_force = force_x + gravity * mass;
        float acceleration = total_force / mass;

        float old_pos = position;
        float old_vel = velocity;

        position = old_pos + old_vel * timestep + 0.5f * acceleration * timestep * timestep;
        velocity = old_vel + acceleration * timestep;

        float dist = position - target_position;
        float distance = std::abs(dist);

        // A reward that encourages moving closer to the target and penalizes large forces
        float reward = -distance - 0.001f * (force_x * force_x);

        bool done = distance > 10.0f; // The episode is done when the agent is close to the target.

        return { get_observation(), reward, done, false };
    }

    int observation_space() const override {
        return 2; // Position (x) and velocity (x)
    }

    int action_space() const override {
        return 1; // Force (x)
    }

    torch::Tensor get_observation() override {
        std::vector<float> obs;
        obs.push_back(position);
        obs.push_back(velocity);
        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    void render() override {}
    void animate() override {}
    void EnableManipulator() override {}
};

class Basketball2dEnv : public Env {
private:
    std::vector<float> position;
    std::vector<float> velocity;
    float mass;
    float timestep;
    std::vector<float> gravity;
    std::vector<float> target_position;
    float force_multiplier;

public:
    Basketball2dEnv(torch::Device& device)
        : Env(device),
        position({ 0.0f, 0.0f }),
        velocity({ 0.0f, 0.0f }),
        mass(0.1f),
        timestep(1.0f / 240.0f),
        gravity({ 0.0f, 0.0f }),
        target_position({ 3.0f, 3.0f }),
        force_multiplier(10.0f) {
    }

    torch::Tensor reset() override {
        position = { 0.0f, 0.0f };
        velocity = { 0.0f, 0.0f };
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        float force_x = actions[0].item<float>() * force_multiplier;
        float force_y = actions[1].item<float>() * force_multiplier;

        std::vector<float> total_force = {
            force_x + gravity[0] * mass,
            force_y + gravity[1] * mass
        };

        std::vector<float> acceleration = {
            total_force[0] / mass,
            total_force[1] / mass
        };

        std::vector<float> old_pos = position;
        std::vector<float> old_vel = velocity;

        position[0] = old_pos[0] + old_vel[0] * timestep + 0.5f * acceleration[0] * timestep * timestep;
        position[1] = old_pos[1] + old_vel[1] * timestep + 0.5f * acceleration[1] * timestep * timestep;

        velocity[0] = old_vel[0] + acceleration[0] * timestep;
        velocity[1] = old_vel[1] + acceleration[1] * timestep;

        float dist_x = position[0] - target_position[0];
        float dist_y = position[1] - target_position[1];
        float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y);

        float force_magnitude_sq = force_x * force_x + force_y * force_y;
        float reward = -distance - 0.001f * force_magnitude_sq;

        bool done = distance > 20.0f;

        return { get_observation(), reward, done, false };
    }

    int observation_space() const override {
        return 4; // Position (x, y) and velocity (x, y)
    }

    int action_space() const override {
        return 2; // Force (x, y)
    }

    torch::Tensor get_observation() override {
        std::vector<float> obs;
        obs.push_back(position[0]);
        obs.push_back(position[1]);
        obs.push_back(velocity[0]);
        obs.push_back(velocity[1]);
        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    int board_size = 20;
    void render() override {
        std::cout << "\033[2J\033[1;1H"; // ANSI escape codes to clear screen and set cursor to top-left

        // Calculate the center of the board
        int center_x = board_size / 2;
        int center_y = board_size / 2;

        // Scale position to fit the board, with origin (0,0) at the center
        // Assuming positions will be within a reasonable range (e.g., -5 to 5)
        int ball_x = static_cast<int>(position[0] + center_x);
        int ball_y = static_cast<int>(-position[1] + center_y); // Invert y-axis for console rendering

        // Ensure ball position is within bounds
        ball_x = std::max(0, std::min(board_size - 1, ball_x));
        ball_y = std::max(0, std::min(board_size - 1, ball_y));

        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                if (i == ball_y && j == ball_x) {
                    std::cout << "O";
                }
                else {
                    std::cout << ".";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "Position: (" << position[0] << ", " << position[1] << ")\n";
        std::cout << "Velocity: (" << velocity[0] << ", " << velocity[1] << ")\n";
        std::cout << "Distance: " << std::sqrt(std::pow(position[0] - target_position[0], 2) + std::pow(position[1] - target_position[1], 2)) << "\n";
        std::cout << std::flush;
    }

    void animate() override {}
    void EnableManipulator() override {}
};


class BasketballEnv : public Env {
private:
    std::vector<float> position;
    std::vector<float> velocity;
    float mass;
    float timestep;
    std::vector<float> gravity;
    std::vector<float> target_position;
    float force_multiplier;

public:
    BasketballEnv(torch::Device& device)
        : Env(device),
        position({ 0.0f, 0.0f, 0.0f }),
        velocity({ 0.0f, 0.0f, 0.0f }),
        mass(0.1f),
        timestep(1.0f / 240.0f),
        gravity({ 0.0f, 0.0f, 0.0f }),
        target_position({ 3.0f, 3.0f, 3.0f }),
        force_multiplier(10.0f) {
    }

    torch::Tensor reset() override {
        position = { 0.0f, 0.0f, 0.0f };
        velocity = { 0.0f, 0.0f, 0.0f };
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        float force_x = actions[0].item<float>() * force_multiplier;
        float force_y = actions[1].item<float>() * force_multiplier;
        float force_z = actions[2].item<float>() * force_multiplier;

        std::vector<float> total_force = {
            force_x + gravity[0] * mass,
            force_y + gravity[1] * mass,
            force_z + gravity[2] * mass
        };

        std::vector<float> acceleration = {
            total_force[0] / mass,
            total_force[1] / mass,
            total_force[2] / mass
        };

        std::vector<float> old_pos = position;
        std::vector<float> old_vel = velocity;

        position[0] = old_pos[0] + old_vel[0] * timestep + 0.5f * acceleration[0] * timestep * timestep;
        position[1] = old_pos[1] + old_vel[1] * timestep + 0.5f * acceleration[1] * timestep * timestep;
        position[2] = old_pos[2] + old_vel[2] * timestep + 0.5f * acceleration[2] * timestep * timestep;

        velocity[0] = old_vel[0] + acceleration[0] * timestep;
        velocity[1] = old_vel[1] + acceleration[1] * timestep;
        velocity[2] = old_vel[2] + acceleration[2] * timestep;

        float dist_x = position[0] - target_position[0];
        float dist_y = position[1] - target_position[1];
        float dist_z = position[2] - target_position[2];
        float distance = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

        float force_magnitude_sq = force_x * force_x + force_y * force_y + force_z * force_z;
        float reward = -(distance - 1) - 0.001f * (force_x * force_x);

        bool done = distance > 20.0f || (distance < 0.1f && std::abs(velocity[0]*velocity[1]*velocity[2]) < 0.01f);

        return { get_observation(), reward, done, false };
    }

    int observation_space() const override {
        return 6; // Position (x, y, z) and velocity (x, y, z)
    }

    int action_space() const override {
        return 3; // Force (x, y, z)
    }

    torch::Tensor get_observation() override {
        std::vector<float> obs;
        obs.push_back(position[0]);
        obs.push_back(position[1]);
        obs.push_back(position[2]);
        obs.push_back(velocity[0]);
        obs.push_back(velocity[1]);
        obs.push_back(velocity[2]);
        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    void render() override {}
    void animate() override {}
    void EnableManipulator() override {}
};