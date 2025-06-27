// This code demonstrates how to open a window and draw a 2D line graph
// using the native Windows API (WinAPI) for window management and GDI (Graphics Device Interface) for drawing.
// It encapsulates the logic within a 'GraphWindow' class.
// It does NOT use SFML, Bullet 3D, FreeGLUT, OpenGL, GLFW, or GLAD.

// To compile this code, you will typically use a C++ compiler like MinGW-w64 or Visual Studio.
// For MinGW-w64 (g++):
// g++ main.cpp -o graph_app -lgdi32 -mwindows
//
// For Visual Studio: Create a new "Desktop App" project (C++) and paste this code.
// Ensure your project type is suitable for WinAPI (e.g., set subsystem to Windows, not Console).

#include <windows.h> // Core Windows API header
#include <vector>      // For std::vector to hold data
#include <string>      // For std::string for text
#include <algorithm>   // For std::min_element and std::max_element
#include <iostream>    // For std::cerr for error output
#include <stdexcept>   // For std::runtime_error

// Class to manage the window and graph drawing
class GraphWindow {
public:
    // Constructor: Registers the window class and creates the window
    GraphWindow(HINSTANCE hInstance, int initialWidth, int initialHeight, const std::string& title)
        : m_hInstance(hInstance),
        m_hWnd(NULL),
        m_windowWidth(initialWidth),
        m_windowHeight(initialHeight),
        m_padding(70.0f) // Initialize padding
    {
        // Set initial graph data, min/max X/Y will be calculated based on this
        m_graphData = {
            0.5f, 1.2f, 2.0f, 1.8f, 2.5f, 3.0f, 2.8f, 3.5f, 4.0f, 3.8f,
            4.5f, 5.0f, 4.8f, 5.5f, 6.0f, 5.8f, 6.5f, 7.0f, 6.8f, 7.5f,
            7.2f, 6.9f, 6.0f, 5.5f, 5.0f, 4.5f, 4.0f, 3.5f, 3.0f, 2.5f,
            2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f
        };

        // Calculate initial graph min/max X and Y based on data
        if (!m_graphData.empty()) {
            m_graphMinX = 0.0f;
            m_graphMaxX = static_cast<float>(m_graphData.size() - 1);
            m_graphMinY = *std::min_element(m_graphData.begin(), m_graphData.end());
            m_graphMaxY = *std::max_element(m_graphData.begin(), m_graphData.end());
            // Add buffer to Y-axis
            if (m_graphMaxY - m_graphMinY == 0) {
                m_graphMaxY = m_graphMinY + 1.0f;
                m_graphMinY = m_graphMinY - 1.0f;
            }
            else {
                float buffer = (m_graphMaxY - m_graphMinY) * 0.1f;
                m_graphMaxY += buffer;
                m_graphMinY -= buffer;
            }
        }
        else {
            m_graphMinX = 0.0f; m_graphMaxX = 1.0f;
            m_graphMinY = 0.0f; m_graphMaxY = 1.0f;
        }


        // Register the window class
        WNDCLASS wc = {};
        wc.lpfnWndProc = StaticWndProc; // Static callback function
        wc.hInstance = hInstance;
        wc.lpszClassName = "GraphWindowClass";
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); // Will be overwritten by custom drawing

        if (!RegisterClass(&wc)) {
            throw std::runtime_error("Window Registration Failed!");
        }

        // Create the window
        m_hWnd = CreateWindowEx(
            0,
            "GraphWindowClass",
            std::string(title.begin(), title.end()).c_str(), // Convert std::string to wide string (LPCWSTR)
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

    // Show the window
    void Show(int nCmdShow) {
        ShowWindow(m_hWnd, nCmdShow);
        UpdateWindow(m_hWnd);
    }

    // Runs the main message loop for the window
    int RunMessageLoop() {
        MSG msg;
        while (GetMessage(&msg, NULL, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        return (int)msg.wParam;
    }

private:
    HINSTANCE m_hInstance;
    HWND m_hWnd;
    int m_windowWidth;
    int m_windowHeight;

    std::vector<float> m_graphData;
    float m_padding;
    float m_graphMinX;
    float m_graphMaxX;
    float m_graphMinY;
    float m_graphMaxY;

    // Private member function to draw the graph using GDI
    void DrawGraph(HDC hdc, RECT clientRect) {
        // Recalculate graphMinY and graphMaxY based on current data
        // This recalculation can be optimized if data doesn't change often
        if (!m_graphData.empty()) {
            m_graphMinY = *std::min_element(m_graphData.begin(), m_graphData.end());
            m_graphMaxY = *std::max_element(m_graphData.begin(), m_graphData.end());

            // Add a little buffer for max/min Y to ensure points aren't exactly on the edge
            if (m_graphMaxY - m_graphMinY == 0) {
                m_graphMaxY = m_graphMinY + 1.0f;
                m_graphMinY = m_graphMinY - 1.0f;
            }
            else {
                float buffer = (m_graphMaxY - m_graphMinY) * 0.1f;
                m_graphMaxY += buffer;
                m_graphMinY -= buffer;
            }
        }
        else {
            m_graphMinY = 0.0f;
            m_graphMaxY = 1.0f;
        }

        // Define the graph drawing area within the window (pixels)
        float graphAreaX = m_padding;
        float graphAreaY = m_padding;
        float graphAreaWidth = static_cast<float>(clientRect.right - clientRect.left - 2 * m_padding);
        float graphAreaHeight = static_cast<float>(clientRect.bottom - clientRect.top - 2 * m_padding);

        // Ensure dimensions are positive
        if (graphAreaWidth <= 0 || graphAreaHeight <= 0) {
            std::cerr << "Warning: Graph area is too small due to window size or padding." << std::endl;
            return;
        }

        // --- Draw Axes ---
        HPEN hPenAxis = CreatePen(PS_SOLID, 2, RGB(150, 150, 150)); // Grey pen for axes, 2 pixels thick
        HPEN hOldPen = (HPEN)SelectObject(hdc, hPenAxis); // Select the pen into the DC

        // X-axis
        MoveToEx(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY), NULL);
        LineTo(hdc, static_cast<int>(graphAreaX + graphAreaWidth), static_cast<int>(graphAreaY));

        // Y-axis
        MoveToEx(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY), NULL);
        LineTo(hdc, static_cast<int>(graphAreaX), static_cast<int>(graphAreaY + graphAreaHeight));

        // Deselect and delete the axis pen
        SelectObject(hdc, hOldPen);
        DeleteObject(hPenAxis);

        // --- Draw Graph Title and Labels (using TextOut) ---
        SetTextColor(hdc, RGB(255, 255, 255)); // White color for text
        SetBkMode(hdc, TRANSPARENT); // Make text background transparent

        // Graph Title
        std::string title = "Episode Rewards Over Time";
        TextOutA(hdc, static_cast<int>(graphAreaX), static_cast<int>(clientRect.bottom - m_padding / 2 - 20), title.c_str(), title.length());

        // X-axis Label
        std::string xAxisLabel = "Data Point Index";
        TextOutA(hdc, static_cast<int>(graphAreaX + graphAreaWidth / 2 - 50), static_cast<int>(graphAreaY - m_padding / 2), xAxisLabel.c_str(), xAxisLabel.length());

        // Y-axis Label
        std::string yAxisLabel = "Value";
        TextOutA(hdc, static_cast<int>(graphAreaX - m_padding / 2 + 10), static_cast<int>(graphAreaY + graphAreaHeight / 2 - 10), yAxisLabel.c_str(), yAxisLabel.length());


        // --- Draw Data Points and Lines ---
        if (!m_graphData.empty()) {
            HPEN hPenGraph = CreatePen(PS_SOLID, 2, RGB(0, 255, 0)); // Green pen for graph line, 2 pixels thick
            SelectObject(hdc, hPenGraph); // Select the graph pen

            // Start drawing the line strip
            MoveToEx(hdc,
                static_cast<int>(graphAreaX + ((0.0f - m_graphMinX) / (m_graphMaxX - m_graphMinX)) * graphAreaWidth),
                static_cast<int>(graphAreaY + ((m_graphData[0] - m_graphMinY) / (m_graphMaxY - m_graphMinY)) * graphAreaHeight),
                NULL);

            for (size_t i = 0; i < m_graphData.size(); ++i) {
                float xNorm = (static_cast<float>(i) - m_graphMinX) / (m_graphMaxX - m_graphMinX);
                float yNorm = (m_graphData[i] - m_graphMinY) / (m_graphMaxY - m_graphMinY);

                float plotX = graphAreaX + xNorm * graphAreaWidth;
                float plotY = graphAreaY + yNorm * graphAreaHeight;

                // Lines connect from previous point to current
                LineTo(hdc, static_cast<int>(plotX), static_cast<int>(plotY));

                // Draw individual points (simple ellipses for points)
                HBRUSH hBrushPoint = CreateSolidBrush(RGB(255, 0, 0)); // Red brush for points
                HBRUSH hOldBrush = (HBRUSH)SelectObject(hdc, hBrushPoint);

                Ellipse(hdc, static_cast<int>(plotX - 3), static_cast<int>(plotY - 3),
                    static_cast<int>(plotX + 3), static_cast<int>(plotY + 3));

                SelectObject(hdc, hOldBrush);
                DeleteObject(hBrushPoint);
            }

            // Deselect and delete the graph pen
            SelectObject(hdc, hOldPen); // Restore old pen before deleting current
            DeleteObject(hPenGraph);
        }
    }

    // Static Window Procedure: Acts as a trampoline to the instance's member function
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        GraphWindow* pThis = nullptr;

        if (message == WM_NCCREATE) {
            // Get the CREATESTRUCT pointer, which contains the 'this' pointer
            LPCREATESTRUCT lpcs = reinterpret_cast<LPCREATESTRUCT>(lParam);
            pThis = static_cast<GraphWindow*>(lpcs->lpCreateParams);

            // Store the 'this' pointer in the window's user data
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        }
        else {
            // Retrieve the 'this' pointer from the window's user data
            pThis = reinterpret_cast<GraphWindow*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        }

        if (pThis) {
            // Call the instance's non-static message handler
            return pThis->WndProc(hWnd, message, wParam, lParam);
        }

        // If 'this' pointer not set yet or not found, fall back to default processing
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    // Instance Window Procedure: Handles messages for this specific GraphWindow instance
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        switch (message) {
        case WM_CREATE:
            // m_hWnd is already set in the constructor for this instance
            break;

        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps); // Get a device context for painting

            RECT clientRect;
            GetClientRect(hWnd, &clientRect); // Get client rectangle for drawing area

            // Fill background with dark grey
            HBRUSH hBrushBackground = CreateSolidBrush(RGB(50, 50, 50));
            FillRect(hdc, &clientRect, hBrushBackground);
            DeleteObject(hBrushBackground);

            // Call the member function to draw the graph
            DrawGraph(hdc, clientRect);

            EndPaint(hWnd, &ps); // Release the device context
        }
        break;

        case WM_SIZE:
            // Update member window dimensions on resize
            m_windowWidth = LOWORD(lParam);
            m_windowHeight = HIWORD(lParam);
            // Invalidate the window to force a repaint with new dimensions
            InvalidateRect(hWnd, NULL, TRUE);
            break;

        case WM_DESTROY:
            PostQuitMessage(0); // Post a quit message to terminate the application
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam); // Default message processing
        }
        return 0;
    }
};
