#pragma once
#include "RobotSimulator.h"
#include "../env.h"

namespace raylib {
#define Rectangle _RayRectangle
#define CloseWindow _RayCloseWindow
#include "raylib.h"
#undef Rectangle
#undef CloseWindow
}

class DoublePendulumEnv : public Env2D {
public:
    enum StateIdx { TH1 = 0, TH2, DTH1, DTH2, COUNT = 4 };

    DoublePendulumEnv(torch::Device& device)
        : Env2D(device), g(9.81f), dt(0.01f), L1(1.0f), L2(1.0f), M1(0.1f), M2(0.1f) {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    torch::Tensor reset() override {
        state.assign(COUNT, 0.0f);
        std::uniform_real_distribution<float> d_theta(-0.1f, 0.1f);
        state[TH1] = 0.0f + d_theta(rng);
        state[TH2] = 0.0f + d_theta(rng);
        steps = 0;
        return get_observation();
    }

std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        steps++;
        float tau = std::clamp(actions[0].item<float>(), -15.0f, 15.0f);

        float a1 = state[TH1];
        float a2 = state[TH2];
        float v1 = state[DTH1];
        float v2 = state[DTH2];
        float da = a1 - a2;

        // Lagrangian Dynamics: M(q)q'' + C(q,q') + G(q) = Tau
        // Simplified solving for accelerations:
        
        float den1 = (2 * M1 + M2 - M2 * cos(2 * a1 - 2 * a2));
        float d2th1 = (-g * (2 * M1 + M2) * sin(a1) 
                       - M2 * g * sin(a1 - 2 * a2) 
                       - 2 * sin(a1 - a2) * M2 * (v2 * v2 * L2 + v1 * v1 * L1 * cos(a1 - a2)) 
                       + tau) // Motor at joint 1
                      / (L1 * den1);

        float den2 = (L2 / L1) * den1;
        float d2th2 = (2 * sin(a1 - a2) * (v1 * v1 * L1 * (M1 + M2) 
                       + g * (M1 + M2) * cos(a1) 
                       + v2 * v2 * L2 * M2 * cos(a1 - a2))) 
                      / den2;

        d2th1 = std::clamp(d2th1, -100.0f, 100.0f);
        d2th2 = std::clamp(d2th2, -100.0f, 100.0f);


        // Semi-implicit Euler
        state[DTH1] += d2th1 * dt;
        state[DTH2] += d2th2 * dt;
        state[TH1] += state[DTH1] * dt;
        state[TH2] += state[DTH2] * dt;

        float reward = -cos(state[TH1]) - cos(state[TH2]) - (0.01f * tau * tau) - (0.00001f * d2th1 * d2th2 * d2th1 * d2th2);
        bool done = steps >= 1000;

        return { get_observation(), reward, done, false };
    }

    void render() override {
        if (state.empty()) return;
        raylib::BeginDrawing();
        raylib::ClearBackground(raylib::BLACK);

        float sc = 150.0f;
        int cx = screenWidth / 2, cy = screenHeight / 2;

        // Inverted Y axis: Changed from (cy - L*cos) to (cy + L*cos)
        float x1 = cx + L1 * sc * sin(state[TH1]);
        float y1 = cy + L1 * sc * cos(state[TH1]);
        float x2 = x1 + L2 * sc * sin(state[TH2]);
        float y2 = y1 + L2 * sc * cos(state[TH2]);

        raylib::DrawLineEx({ (float)cx, (float)cy }, { x1, y1 }, 5.0f, raylib::RAYWHITE);
        raylib::DrawLineEx({ x1, y1 }, { x2, y2 }, 5.0f, raylib::LIGHTGRAY);
        raylib::DrawCircleV({ (float)cx, (float)cy }, 5, raylib::RED);
        raylib::DrawCircleV({ x1, y1 }, 10, raylib::BLUE);
        raylib::DrawCircleV({ x2, y2 }, 10, raylib::GREEN);

        raylib::EndDrawing();
    }

    int observation_space() const override { return COUNT; }
    int action_space() const override { return 1; }

private:
    float g, dt, L1, L2, M1, M2;
    std::vector<float> state;
    int steps;
    std::mt19937 rng;

    torch::Tensor get_observation() override { return torch::tensor(state).to(mDevice); }
};