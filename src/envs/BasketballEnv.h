#include "../env.h"
#include <vector>
#include <stdexcept>
#include <string>

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
        position({ 0.0f, 0.0f, 1.0f }),
        velocity({ 0.0f, 0.0f, 0.0f }),
        mass(1.0f),
        timestep(1.0f / 240.0f),
        gravity({ 0.0f, 0.0f, 0.0f }),
        target_position({ 0.0f, 0.0f, 3.0f }),
        force_multiplier(10.0f) {
    }

    torch::Tensor reset() override {
        position = { 0.0f, 0.0f, 1.0f };
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
        float reward = -distance - 0.001f * force_magnitude_sq;

        bool done = distance > 10.0f;

        return { get_observation(), reward, done, false };
    }

    Space observation_space() const override {
        return Space{ {6} };
    }

    Space action_space() const override {
        return Space{ {3} };
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