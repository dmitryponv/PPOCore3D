#pragma once
#include "RobotSimulator.h"
#include "../env.h"


class TriplePendulumEnv : public Env2D {
public:
    enum StateIdx {
        TH1 = 0, TH2, TH3, 
        DTH1, DTH2, DTH3,
        COUNT = 6
    };

    TriplePendulumEnv(torch::Device& device)
        : Env2D(device), g(9.81f), dt(0.02f), L1(1.0f), L2(1.0f), L3(1.0f), M1(1.0f), M2(1.0f), M3(1.0f) {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    torch::Tensor reset() override {
        state.assign(COUNT, 0.0f);
        std::uniform_real_distribution<float> d_theta(-0.1f, 0.1f);
        state[TH1] = d_theta(rng) + 3.14159f; // Start near bottom
        state[TH2] = d_theta(rng);
        state[TH3] = d_theta(rng);
        steps = 0;
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        steps++;
        float torque = std::clamp(actions[0].item<float>(), -20.0f, 20.0f);

        // Simple Euler integration for Hamiltonian/Lagrangian dynamics (simplified approximation)
        // In a real env, you'd solve the M(q)q'' + C(q,q') + G(q) = tau system
        float d2th1 = (-g * (2 * M1 + M2) * sin(state[TH1]) - M2 * g * sin(state[TH1] - 2 * state[TH2]) + torque) / (L1 * (2 * M1 + M2 - M2 * cos(2 * state[TH1] - 2 * state[TH2])));
        float d2th2 = (2 * sin(state[TH1] - state[TH2]) * (state[DTH1] * state[DTH1] * L1 * (M1 + M2) + g * (M1 + M2) * cos(state[TH1]) + state[DTH2] * state[DTH2] * L2 * M2 * cos(state[TH1] - state[TH2]))) / (L2 * (2 * M1 + M2 - M2 * cos(2 * state[TH1] - 2 * state[TH2])));
        float d2th3 = -g * sin(state[TH3]) / L3; 

        state[DTH1] += d2th1 * dt;
        state[DTH2] += d2th2 * dt;
        state[DTH3] += d2th3 * dt;
        state[TH1] += state[DTH1] * dt;
        state[TH2] += state[DTH2] * dt;
        state[TH3] += state[DTH3] * dt;

        float reward = cos(state[TH1]) + cos(state[TH2]) + cos(state[TH3]);
        bool done = steps >= 1000;

        return { get_observation(), reward, done, false };
    }

    void render() override {
        if (state.empty()) return;
        raylib::BeginDrawing();
        raylib::ClearBackground(raylib::BLACK);

        float sc = 100.0f;
        int cx = screenWidth / 2, cy = screenHeight / 2;

        float x1 = cx + L1 * sc * sin(state[TH1]);
        float y1 = cy + L1 * sc * cos(state[TH1]);
        float x2 = x1 + L2 * sc * sin(state[TH2]);
        float y2 = y1 + L2 * sc * cos(state[TH2]);
        float x3 = x2 + L3 * sc * sin(state[TH3]);
        float y3 = y2 + L3 * sc * cos(state[TH3]);

        raylib::DrawLineEx({(float)cx, (float)cy}, {x1, y1}, 5.0f, raylib::WHITE);
        raylib::DrawLineEx({x1, y1}, {x2, y2}, 5.0f, raylib::LIGHTGRAY);
        raylib::DrawLineEx({x2, y2}, {x3, y3}, 5.0f, raylib::GRAY);

        raylib::DrawCircleV({(float)cx, (float)cy}, 8, raylib::RED);
        raylib::DrawCircleV({x1, y1}, 12, raylib::BLUE);
        raylib::DrawCircleV({x2, y2}, 12, raylib::GREEN);
        raylib::DrawCircleV({x3, y3}, 12, raylib::ORANGE);

        raylib::EndDrawing();
    }

    int observation_space() const override { return COUNT; }
    int action_space() const override { return 1; }

private:
    float g, dt, L1, L2, L3, M1, M2, M3;
    std::vector<float> state;
    int steps;
    std::mt19937 rng;

    torch::Tensor get_observation() override { return torch::tensor(state).to(mDevice); }
};