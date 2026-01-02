#pragma once
#include "RobotSimulator.h"
#include "../env.h"

class LunarLanderEnv : public Env2D {
public:
    LunarLanderEnv(torch::Device& device)
        : Env2D(device),
          gravity(10.0f),
          dt(0.05f),
          main_engine_power(13.0f),
          side_engine_power(1.0f) // Weaker than main engine
    {
        // Random number generation
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    torch::Tensor reset() override {
        std::uniform_real_distribution<float> dist_pos(-0.5f, 0.5f);
        std::uniform_real_distribution<float> dist_vel(-1.0f, 1.0f);
        std::uniform_real_distribution<float> dist_angle(-0.2f, 0.2f);

        // State: [x, y, vx, vy, theta, theta_dot]
        // Start high up (y=10) with slight randomness
        state = {
            dist_pos(rng),       // x
            10.0f + dist_pos(rng), // y
            dist_vel(rng),       // vx
            dist_vel(rng),       // vy
            dist_angle(rng),     // theta
            dist_angle(rng)      // theta_dot
        };
        
        steps = 0;
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        steps++;
        
        // Actions: [0] = Vertical Thrust (Main), [1] = Horizontal Thrust (Side)
        // Clamp actions to range [-1, 1] then scale to power
        float act_main = std::clamp(actions[0].item<float>(), 0.0f, 1.0f); // Main engine usually 0 to 1
        float act_side = std::clamp(actions[1].item<float>(), -1.0f, 1.0f); 

        float force_y = act_main * main_engine_power;
        float force_x = act_side * side_engine_power;

        float theta = state[4];
        float theta_dot = state[5];

        // --- Physics Update ---
        
        // Rotation: Side thrusters apply torque (Force * distance_offset)
        // Assuming COM is slightly offset from thruster line
        float torque = -force_x * 0.5f; 
        float moment_of_inertia = 2.0f; // Arbitrary mass/shape constant
        float angular_acc = torque / moment_of_inertia;
        
        float new_theta_dot = theta_dot + angular_acc * dt;
        float new_theta = theta + new_theta_dot * dt;

        // Linear Forces transformed to Global Frame
        // Sin/Cos for rotation transformation
        float s = std::sin(theta);
        float c = std::cos(theta);

        // Force vector in global frame:
        // F_global_x = -F_local_y * sin(theta) + F_local_x * cos(theta)
        // F_global_y =  F_local_y * cos(theta) + F_local_x * sin(theta)
        float accel_x = (-force_y * s + force_x * c); // Mass = 1.0 assumed
        float accel_y = (force_y * c + force_x * s) - gravity;

        float new_vx = state[2] + accel_x * dt;
        float new_vy = state[3] + accel_y * dt;
        float new_x = state[0] + new_vx * dt;
        float new_y = state[1] + new_vy * dt;

        state = { new_x, new_y, new_vx, new_vy, new_theta, new_theta_dot };

        // --- Reward & Termination ---
        bool done = false;
        float reward = 0.0f;

        // Cost for using fuel
        reward -= 0.1f * std::abs(act_main); 
        reward -= 0.03f * std::abs(act_side);

        // Distance shaping reward
        float dist = std::sqrt(state[0]*state[0] + state[1]*state[1]);
        float vel_penalty = std::sqrt(state[2]*state[2] + state[3]*state[3]);
        float angle_penalty = std::abs(state[4]);
        
        // Heuristic reward per step
        reward += (last_shaping - (100.0f * dist + 100.0f * vel_penalty + 100.0f * angle_penalty));
        last_shaping = 100.0f * dist + 100.0f * vel_penalty + 100.0f * angle_penalty;

        // Terminal conditions
        if (state[1] <= 0.0f) { // Ground contact
            done = true;
            bool upright = std::abs(state[4]) < 0.2f;
            bool slow = std::abs(state[3]) < 2.0f; // Landing speed
            
            if (upright && slow) reward += 100.0f; // Successful land
            else reward -= 100.0f; // Crash
        } else if (std::abs(state[0]) > 10.0f || state[1] > 20.0f) {
            done = true;
            reward -= 100.0f; // Out of bounds
        }
        
        if (steps > 1000) done = true;

        return { get_observation(), reward, done, false };
    }

    void render() override {
        if (!state.empty()) {
            std::cout << "Lander [x,y]: " << state[0] << "," << state[1] 
                      << " Angle: " << state[4] << "\n";
        }
        Env2D::render();
    }

    // Required overrides
    void animate() override {}
    void EnableManipulator() override {}

    int observation_space() const override { return 8; } // x, y, vx, vy, theta, w, 2 dummy legs
    int action_space() const override { return 2; } // Main Engine, Side Engine

private:
    float gravity, dt;
    float main_engine_power, side_engine_power;
    std::vector<float> state;
    float last_shaping = 0.0f;
    int steps = 0;

    std::mt19937 rng;

    torch::Tensor get_observation() override {
        // Return 8 state vars to match standard envs usually
        // [x, y, vx, vy, theta, theta_dot, left_leg_gnd, right_leg_gnd]
        return torch::tensor({
            state[0], state[1], 
            state[2], state[3], 
            state[4], state[5], 
            0.0f, 0.0f // Dummy leg contact sensors
        }).to(mDevice);
    }
};