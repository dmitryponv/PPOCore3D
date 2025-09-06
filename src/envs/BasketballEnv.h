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
        position({ 0.0f, 0.0f, 0.0f }),
        velocity({ 0.0f, 0.0f, 0.0f }),
        mass(0.1f),
        timestep(1.0f / 240.0f),
        gravity({ 0.0f, 0.0f, 0.0f }),
        target_position({ 3.0f, 0.0f, 0.0f }),
        force_multiplier(1000.0f) {
    }

    torch::Tensor reset() override {
        position = { 0.0f, 0.0f, 0.0f };
        velocity = { 0.0f, 0.0f, 0.0f };
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        float force_x = actions[0].item<float>() * force_multiplier;

        std::vector<float> total_force = {
            force_x + gravity[0] * mass,
        };

        std::vector<float> acceleration = {
            total_force[0] / mass,
        };

        std::vector<float> old_pos = position;
        std::vector<float> old_vel = velocity;

        position[0] = old_pos[0] + old_vel[0] * timestep + 0.5f * acceleration[0] * timestep * timestep;

        velocity[0] = old_vel[0] + acceleration[0] * timestep;

        float dist_x = position[0] - target_position[0];
        float distance = std::sqrt(dist_x * dist_x);

        //printf("distance: %f \r\n", distance);

        float force_magnitude_sq = force_x * force_x;
        float reward = -distance+0.1 - 0.001f * force_magnitude_sq;

        bool done = distance > 10.0f;

        return { get_observation(), reward, done, false };
    }

    Space observation_space() const override {
        return Space{ {2} };
    }

    Space action_space() const override {
        return Space{ {1} };
    }

    torch::Tensor get_observation() override {
        std::vector<float> obs;
        obs.push_back(position[0]);
        obs.push_back(velocity[0]);
        return torch::from_blob(obs.data(), { (int)obs.size() }).clone().to(mDevice);
    }

    void render() override {}
    void animate() override {}
    void EnableManipulator() override {}
};