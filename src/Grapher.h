#pragma once // Ensures this header is included only once during compilation
#define WIN32_LEAN_AND_MEAN // Excludes less common API elements to speed up compilation
#define NOMINMAX // Prevents Windows.h from defining min/max macros that conflict with std::min/std::max

#include <windows.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <map> // For managing multiple GraphWindow instances
#include <memory> // For std::unique_ptr
#include <atomic> // For std::atomic, or use volatile long for Interlocked* functions

// Global counter for active graph windows
// Using volatile LONG with InterlockedIncrement/Decrement for basic thread-safe counting
// __declspec(selectany) tells the linker to pick one definition if multiple exist,
// resolving LNK2005 errors when a global variable is defined in multiple compilation units.
__declspec(selectany) volatile LONG g_activeWindowCount = 0;

// Class to manage a single graph window and its drawing
class GraphWindow {
public:
    // Constructor: Registers the window class and creates the window
    GraphWindow(HINSTANCE hInstance, int initialWidth, int initialHeight, const std::string& title)
        : m_hInstance(hInstance),
        m_hWnd(NULL),
        m_windowWidth(initialWidth),
        m_windowHeight(initialHeight),
        m_padding(70.0f)
    {
        // Register the window class only once per application instance
        static bool classRegistered = false;
        if (!classRegistered) {
            WNDCLASS wc = {};
            wc.lpfnWndProc = StaticWndProc; // Static callback function
            wc.hInstance = hInstance;
            wc.lpszClassName = "GraphWindowClass"; // Use narrow string literal
            wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

            if (!RegisterClass(&wc)) {
                throw std::runtime_error("Window Class Registration Failed!");
            }
            classRegistered = true;
        }

        // Create the window
        m_hWnd = CreateWindowExA( // Use CreateWindowExA for narrow strings
            0,
            "GraphWindowClass", // Use narrow string literal
            title.c_str(),      // Use c_str() directly for narrow string title
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            m_windowWidth, m_windowHeight,
            NULL,
            NULL,
            hInstance,
            this // Pass 'this' pointer as lpParam to WM_CREATE
        );

        if (m_hWnd == NULL) {
            throw std::runtime_error("Window Creation Failed!");
        }
    }

    // Destructor: Decrements the global window count
    ~GraphWindow() {
        // If the HWND is still valid, ensure it's removed from global count
        // WM_DESTROY already decrements, but this covers cases where window isn't fully created
        // or if GraphWindow object is destroyed without WM_DESTROY (e.g., app force closes)
        if (m_hWnd && IsWindow(m_hWnd)) {
            // This case should ideally not happen if WM_DESTROY is processed correctly
            // but for robustness, ensures counter consistency.
            InterlockedDecrement(&g_activeWindowCount);
        }
    }

    // Displays the window
    void Show(int nCmdShow) {
        ShowWindow(m_hWnd, nCmdShow);
        UpdateWindow(m_hWnd);
    }

    // Plots new graph data
    // y: required vector of Y-values
    // x: optional vector of X-values; if empty, uses 0, 1, 2...
    void Graph(const std::string& windowTitle, const std::vector<float>& y, const std::vector<float>& x = {}) {
        m_graphData = y;
        m_graphXData = x;
        m_windowTitle = windowTitle; // Update window title

        if (m_graphXData.empty()) {
            m_graphXData.resize(m_graphData.size());
            for (size_t i = 0; i < m_graphData.size(); ++i) {
                m_graphXData[i] = static_cast<float>(i);
            }
        }

        // Recalculate min/max values based on new data
        if (!m_graphData.empty() && !m_graphXData.empty()) {
            m_graphMinX = *std::min_element(m_graphXData.begin(), m_graphXData.end());
            m_graphMaxX = *std::max_element(m_graphXData.begin(), m_graphXData.end());
            m_graphMinY = *std::min_element(m_graphData.begin(), m_graphData.end());
            m_graphMaxY = *std::max_element(m_graphData.begin(), m_graphData.end());

            // Add buffer to X-axis
            if (m_graphMaxX - m_graphMinX == 0) {
                m_graphMaxX = m_graphMinX + 1.0f;
                m_graphMinX = m_graphMinX - 1.0f;
            }
            else {
                float bufferX = (m_graphMaxX - m_graphMinX) * 0.05f; // 5% buffer
                m_graphMaxX += bufferX;
                m_graphMinX -= bufferX;
            }

            // Add buffer to Y-axis
            if (m_graphMaxY - m_graphMinY == 0) {
                m_graphMaxY = m_graphMinY + 1.0f;
                m_graphMinY = m_graphMinY - 1.0f;
            }
            else {
                float bufferY = (m_graphMaxY - m_graphMinY) * 0.1f; // 10% buffer
                m_graphMaxY += bufferY;
                m_graphMinY -= bufferY;
            }
        }
        else { // Handle empty data
            m_graphMinX = 0.0f; m_graphMaxX = 1.0f;
            m_graphMinY = 0.0f; m_graphMaxY = 1.0f;
        }

        // Set the new window title
        SetWindowTextA(m_hWnd, m_windowTitle.c_str()); // Use SetWindowTextA
        // Force redraw
        InvalidateRect(m_hWnd, NULL, TRUE);
    }

    // Public getter for HWND, used by WinMain to check window validity
    HWND GetHwnd() const { return m_hWnd; }

private:
    HINSTANCE m_hInstance;
    HWND m_hWnd;
    int m_windowWidth;
    int m_windowHeight;
    std::string m_windowTitle;

    std::vector<float> m_graphData;    // Y-values
    std::vector<float> m_graphXData;   // X-values
    float m_padding;
    float m_graphMinX;
    float m_graphMaxX;
    float m_graphMinY;
    float m_graphMaxY;

    void DrawGraph(HDC hdc, RECT clientRect) {
        float graphAreaX = m_padding;
        float graphAreaY = m_padding;
        float graphAreaWidth = static_cast<float>(clientRect.right - clientRect.left - 2 * m_padding);
        float graphAreaHeight = static_cast<float>(clientRect.bottom - clientRect.top - 2 * m_padding);

        if (graphAreaWidth <= 0 || graphAreaHeight <= 0 || m_graphData.empty() || m_graphXData.empty()) {
            return;
        }

        HPEN hPenAxis = CreatePen(PS_SOLID, 2, RGB(150, 150, 150));
        HPEN hOldPen = (HPEN)SelectObject(hdc, hPenAxis);

        MoveToEx(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY + graphAreaHeight), NULL);
        LineTo(hdc, static_cast<int>(graphAreaX + graphAreaWidth), static_cast<int>(graphAreaY + graphAreaHeight));

        MoveToEx(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY), NULL);
        LineTo(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY + graphAreaHeight));

        SelectObject(hdc, hOldPen);
        DeleteObject(hPenAxis);

        SetTextColor(hdc, RGB(255, 255, 255));
        SetBkMode(hdc, TRANSPARENT);

        TextOutA(hdc, static_cast<int>(graphAreaX), static_cast<int>(clientRect.bottom - m_padding / 2 - 20), m_windowTitle.c_str(), m_windowTitle.length());
        TextOutA(hdc, static_cast<int>(graphAreaX + graphAreaWidth / 2 - 50), static_cast<int>(graphAreaY - m_padding / 2), "X-Axis", 6);
        TextOutA(hdc, static_cast<int>(graphAreaX - m_padding / 2 + 10), static_cast<int>(graphAreaY + graphAreaHeight / 2 - 10), "Y-Axis", 6);

        HPEN hPenGraph = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
        SelectObject(hdc, hPenGraph);

        float firstXNorm = (m_graphXData[0] - m_graphMinX) / (m_graphMaxX - m_graphMinX);
        float firstYNorm = (m_graphData[0] - m_graphMinY) / (m_graphMaxY - m_graphMinY);
        MoveToEx(hdc,
            static_cast<int>(graphAreaX + firstXNorm * graphAreaWidth),
            static_cast<int>(graphAreaY + (1.0f - firstYNorm) * graphAreaHeight),
            NULL);

        for (size_t i = 0; i < m_graphData.size(); ++i) {
            float xNorm = (m_graphXData[i] - m_graphMinX) / (m_graphMaxX - m_graphMinX);
            float yNorm = (m_graphData[i] - m_graphMinY) / (m_graphMaxY - m_graphMinY);

            float plotX = graphAreaX + xNorm * graphAreaWidth;
            float plotY = graphAreaY + (1.0f - yNorm) * graphAreaHeight;

            LineTo(hdc, static_cast<int>(plotX), static_cast<int>(plotY));

            HBRUSH hBrushPoint = CreateSolidBrush(RGB(255, 0, 0));
            HBRUSH hOldBrush = (HBRUSH)SelectObject(hdc, hBrushPoint);

            Ellipse(hdc, static_cast<int>(plotX - 3), static_cast<int>(plotY - 3),
                static_cast<int>(plotX + 3), static_cast<int>(plotY + 3));

            SelectObject(hdc, hOldBrush);
            DeleteObject(hBrushPoint);
        }

        SelectObject(hdc, hOldPen);
        DeleteObject(hPenGraph);
    }


    // Static Window Procedure: Trampoline to the instance's member function
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        GraphWindow* pThis = nullptr;
        if (message == WM_NCCREATE) {
            LPCREATESTRUCT lpcs = reinterpret_cast<LPCREATESTRUCT>(lParam);
            pThis = static_cast<GraphWindow*>(lpcs->lpCreateParams);
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        }
        else {
            pThis = reinterpret_cast<GraphWindow*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        }

        if (pThis) {
            return pThis->WndProc(hWnd, message, wParam, lParam);
        }
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    // Instance Window Procedure: Handles messages for this GraphWindow instance
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
        case WM_CREATE:
            // Increment active window count when window is created
            InterlockedIncrement(&g_activeWindowCount);
            break;
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            RECT clientRect;
            GetClientRect(hWnd, &clientRect);
            HBRUSH hBrushBackground = CreateSolidBrush(RGB(50, 50, 50));
            FillRect(hdc, &clientRect, hBrushBackground);
            DeleteObject(hBrushBackground);
            DrawGraph(hdc, clientRect);
            EndPaint(hWnd, &ps);
        }
        break;
        case WM_SIZE:
            m_windowWidth = LOWORD(lParam);
            m_windowHeight = HIWORD(lParam);
            InvalidateRect(hWnd, NULL, TRUE);
            break;
        case WM_DESTROY:
            // Decrement active window count when window is destroyed
            InterlockedDecrement(&g_activeWindowCount);
            m_hWnd = NULL; // Clear HWND to indicate window is no longer valid
            // Do NOT PostQuitMessage here; WinMain's loop will manage termination
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        return 0;
    }
};


class GraphWindowManager {
private:
    std::vector<std::unique_ptr<GraphWindow>> graphWindows;
    std::thread messageLoopThread;
    std::atomic<bool> running{ false };
    HINSTANCE hInstance;
    int nCmdShow;

public:
    GraphWindowManager(HINSTANCE hInst, int cmdShow)
        : hInstance(hInst), nCmdShow(cmdShow) {
    }

    void Init() {
        //StartMessageLoop();
    }

    void Graph(const std::string& title, const std::vector<float>& yData, const std::vector<float>& xData = {}) {
        size_t index = EnsureWindow(title);
        if (index < graphWindows.size()) {
            if (xData.empty()) {
                graphWindows[index]->Graph(title, yData);
            }
            else {
                graphWindows[index]->Graph(title, yData, xData);
            }
        }
        MSG msg = {};
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                running = false;
                break;
            }
        }
    }

    void Graph(const std::string& title, float value) {
        static std::map<std::string, std::vector<float>> graphData;
        graphData[title].push_back(value);
        size_t index = EnsureWindow(title);
        if (index < graphWindows.size()) {
            graphWindows[index]->Graph(title, graphData[title]);
        }
        MSG msg = {};
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                running = false;
                break;
            }
        }
    }

    size_t EnsureWindow(const std::string& title) {
        static std::map<std::string, size_t> titleToIndex;
        auto it = titleToIndex.find(title);
        if (it != titleToIndex.end()) {
            return it->second;
        }

        int index = static_cast<int>(graphWindows.size());
        int width = 600 + index * 50;
        int height = 400 + index * 50;
        graphWindows.push_back(std::make_unique<GraphWindow>(hInstance, width, height, title.c_str()));
        graphWindows.back()->Show(nCmdShow);

        titleToIndex[title] = index;
        return index;
    }

    void CloseAll() {
        running = false;
        PostQuitMessage(0);
        if (messageLoopThread.joinable()) {
            messageLoopThread.join();
        }
    }

//private:
//    void StartMessageLoop() {
//        running = true;
//        messageLoopThread = std::thread([this]() {
//            MSG msg = {};
//            while (running && g_activeWindowCount > 0) {
//                while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
//                    TranslateMessage(&msg);
//                    DispatchMessage(&msg);
//                    if (msg.message == WM_QUIT) {
//                        running = false;
//                        break;
//                    }
//                }
//                if (running && g_activeWindowCount > 0) {
//                    Sleep(10);
//                }
//            }
//            });
//    }
};