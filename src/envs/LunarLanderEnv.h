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

class LunarLanderEnv : public Env2D {
public:
    enum StateIdx {
        X = 0, Y, VX, VY, THETA, THETA_DOT, T_X, T_Y,
        AST1_X, AST1_Y, AST2_X, AST2_Y, AST3_X, AST3_Y,
        COUNT = 14
    };

    LunarLanderEnv(torch::Device& device)
        : Env2D(device), gravity(10.0f), dt(0.05f),
        main_engine_power(13.0f), side_engine_power(1.0f) {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    torch::Tensor reset() override {
        state.assign(StateIdx::COUNT, 0.0f);
        std::uniform_real_distribution<float> d_x(MIN_X, MAX_X);
        std::uniform_real_distribution<float> d_y(MIN_Y, MAX_Y);

        state[X] = d_x(rng);
        state[Y] = d_y(rng);
        state[T_X] = d_x(rng);
        state[T_Y] = std::uniform_real_distribution<float>(2.0f, 20.0f)(rng);

        target_vel_x = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
        target_vel_y = std::uniform_real_distribution<float>(-0.5f, 0.5f)(rng);

        for (int i = AST1_X; i < COUNT; i += 2) {
            state[i] = d_x(rng);
            state[i + 1] = d_y(rng);
        }

        steps = 0;
        last_shaping = calculate_shaping();
        last_actions = { 0.0f, 0.0f };
        return get_observation();
    }

    std::tuple<torch::Tensor, float, bool, bool> step(const torch::Tensor& actions, int frame_index) override {
        steps++;
        float act_m = std::clamp(actions[0].item<float>(), 0.0f, 1.0f);
        float act_s = std::clamp(actions[1].item<float>(), -1.0f, 1.0f);
        last_actions = { act_m, act_s };

        state[T_X] += target_vel_x * dt;
        state[T_Y] += target_vel_y * dt;

        if (state[T_X] < MIN_X) { state[T_X] = MIN_X; target_vel_x *= -1.0f; }
        if (state[T_X] > MAX_X) { state[T_X] = MAX_X; target_vel_x *= -1.0f; }
        if (state[T_Y] < 2.0f)  { state[T_Y] = 2.0f;  target_vel_y *= -1.0f; }
        if (state[T_Y] > 25.0f) { state[T_Y] = 25.0f; target_vel_y *= -1.0f; }

        state[THETA_DOT] += (-act_s * side_engine_power * 0.25f) * dt;
        state[THETA] += state[THETA_DOT] * dt;
        float s = std::sin(state[THETA]), c = std::cos(state[THETA]);
        float ax = (-act_m * main_engine_power * s + act_s * side_engine_power * c);
        float ay = (act_m * main_engine_power * c + act_s * side_engine_power * s) - gravity;

        state[VX] += ax * dt; state[VY] += ay * dt;
        state[X] += state[VX] * dt; state[Y] += state[VY] * dt;

        float cur_shaping = calculate_shaping();
        float reward = (last_shaping - cur_shaping) - 0.01f;
        last_shaping = cur_shaping;

        bool done = false;
        for (int i = AST1_X; i < COUNT; i += 2) {
            float adist = std::sqrt(std::pow(state[X] - state[i], 2) + std::pow(state[Y] - state[i + 1], 2));
            if (adist < 1.2f) { reward -= 200.0f; done = true; }
        }

        float dist_t = std::sqrt(std::pow(state[X] - state[T_X], 2) + std::pow(state[Y] - state[T_Y], 2));
        if (dist_t < 1.0f) {
            done = true;
            reward += (std::sqrt(state[VX] * state[VX] + state[VY] * state[VY]) < 2.0f) ? 300.0f : -50.0f;
        }
        else if (state[Y] <= 0.0f || std::abs(state[X]) > 20.0f || state[Y] > 30.0f) {
            done = true; reward -= 150.0f;
        }

        if (steps > 800) done = true;
        return { get_observation(), reward, done, false };
    }

    void render() override {
        if (state.empty()) return;
        raylib::BeginDrawing();
        raylib::ClearBackground(raylib::BLACK);
        float sc = 25.0f;
        int cx = screenWidth / 2, cy = screenHeight - 50;
        raylib::Vector2 pos = { cx + state[X] * sc, cy - state[Y] * sc };

        for (int i = AST1_X; i < COUNT; i += 2) {
            raylib::Vector2 astPos = { cx + state[i] * sc, cy - state[i + 1] * sc };
            raylib::DrawPoly(astPos, (i % 3) + 5, 18.0f, (float)(i * 10), raylib::DARKGRAY);
            raylib::DrawPolyLinesEx(astPos, (i % 3) + 5, 18.0f, (float)(i * 10), 2.0f, raylib::BROWN);
        }

        float tx = cx + state[T_X] * sc, ty = cy - state[T_Y] * sc;
        raylib::DrawLineEx({ tx, ty }, { tx, ty - 40 }, 3.0f, raylib::RAYWHITE);
        raylib::DrawTriangle({ tx, ty - 40 }, { tx, ty - 20 }, { tx + 25, ty - 30 }, raylib::RED);

        if (last_actions[0] > 0.1f) {
            raylib::Vector2 fEnd = { pos.x + std::sin(state[THETA]) * 40, pos.y + std::cos(state[THETA]) * 40 };
            raylib::DrawLineEx(pos, fEnd, 4.0f, raylib::ORANGE);
        }

        raylib::DrawRectanglePro({ pos.x, pos.y, 22, 28 }, { 11, 14 }, state[THETA] * 57.3f, raylib::LIGHTGRAY);
        raylib::DrawLine(0, cy, screenWidth, cy, raylib::GREEN);
        raylib::EndDrawing();
    }

    int observation_space() const override { return StateIdx::COUNT; }
    int action_space() const override { return 2; }

private:
    const float MIN_X = -18.0f;
    const float MAX_X = 18.0f;
    const float MIN_Y = 5.0f;
    const float MAX_Y = 25.0f;

    float gravity, dt, main_engine_power, side_engine_power, last_shaping, target_vel_x, target_vel_y;
    std::vector<float> state, last_actions;
    int steps;
    std::mt19937 rng;

    float calculate_shaping() {
        float d = std::sqrt(std::pow(state[X] - state[T_X], 2) + std::pow(state[Y] - state[T_Y], 2));
        float v = std::sqrt(state[VX] * state[VX] + state[VY] * state[VY]);
        return (30.0f * d) + (10.0f * v) + (20.0f * std::abs(state[THETA]));
    }

    torch::Tensor get_observation() override { return torch::tensor(state).to(mDevice); }
};