#pragma once // Ensures this header is included only once during compilation
#define WIN32_LEAN_AND_MEAN // Excludes less common API elements to speed up compilation
#define NOMINMAX // Prevents Windows.h from defining min/max macros that conflict with std::min/std::max

#include <windows.h>
#include <commctrl.h> // Required for trackbar and up-down controls
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <functional> // For std::function

// Link with comctl32.lib for common controls
#pragma comment(lib, "comctl32.lib")

// Unique identifiers for controls to distinguish them in WM_COMMAND/WM_HSCROLL
#define IDC_TRACKBAR_JOINT_BASE 1000
#define IDC_UPDOWN_FRAME 2000
#define IDC_EDIT_FRAME 2001

class Manipulator {
public:
    // Callback function type for when joint angles or frame number change
    using OnManipulatorChangeCallback = std::function<void(const std::vector<int>& jointAngles, int frameNumber)>;

    // Constructor: Creates the manipulator window and its controls
    Manipulator(HINSTANCE hInstance, int initialX, int initialY, int initialWidth, int initialHeight,
        const std::string& title, int numberOfJoints)
        : m_hInstance(hInstance),
        m_hWnd(NULL),
        m_windowWidth(initialWidth),
        m_windowHeight(initialHeight),
        m_title(title),
        m_numberOfJoints(numberOfJoints),
        m_frameNumber(0)
    {
        // Initialize joint angles to 0
        m_jointAngles.resize(m_numberOfJoints, 0);
        m_jointMinMax.resize(m_numberOfJoints, { 0, 360 }); // Default range for joints

        // Initialize Common Controls library (needed for trackbars and up-down controls)
        // This should ideally be called once per application.
        static bool commonControlsInitialized = false;
        if (!commonControlsInitialized) {
            INITCOMMONCONTROLSEX icc;
            icc.dwSize = sizeof(icc);
            icc.dwICC = ICC_BAR_CLASSES | ICC_UPDOWN_CLASS; // Load trackbar and up-down classes
            InitCommonControlsEx(&icc);
            commonControlsInitialized = true;
        }

        // Register the window class for the Manipulator
        static bool classRegistered = false;
        if (!classRegistered) {
            WNDCLASS wc = {};
            wc.lpfnWndProc = StaticWndProc;
            wc.hInstance = hInstance;
            wc.lpszClassName = "ManipulatorWindowClass";
            wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
            wc.hCursor = LoadCursor(NULL, IDC_ARROW);

            if (!RegisterClass(&wc)) {
                throw std::runtime_error("Manipulator Window Class Registration Failed!");
            }
            classRegistered = true;
        }

        // Create the Manipulator window
        m_hWnd = CreateWindowExA(
            0,
            "ManipulatorWindowClass",
            m_title.c_str(),
            WS_OVERLAPPEDWINDOW | WS_VSCROLL, // Add vertical scroll for many joints
            initialX, initialY,
            m_windowWidth, m_windowHeight,
            NULL,
            NULL,
            hInstance,
            this // Pass 'this' pointer as lpParam to WM_CREATE
        );

        if (m_hWnd == NULL) {
            throw std::runtime_error("Manipulator Window Creation Failed!");
        }

        // Create controls (trackbars and up-down)
        CreateControls();
    }

    // Destructor
    ~Manipulator() {
        if (m_hWnd && IsWindow(m_hWnd)) {
            DestroyWindow(m_hWnd); // Ensures the window is properly destroyed
        }
    }

    // Displays the window
    void Show(int nCmdShow) {
        ShowWindow(m_hWnd, nCmdShow);
        UpdateWindow(m_hWnd);
    }

    // Sets the range for a specific joint trackbar
    void SetJointRange(int jointIndex, int min, int max) {
        if (jointIndex >= 0 && jointIndex < m_numberOfJoints) {
            m_jointMinMax[jointIndex] = { min, max };
            if (m_trackbars[jointIndex]) {
                SendMessage(m_trackbars[jointIndex], TBM_SETRANGEMIN, TRUE, min);
                SendMessage(m_trackbars[jointIndex], TBM_SETRANGEMAX, TRUE, max);
                // Ensure current value is within new range
                if (m_jointAngles[jointIndex] < min) m_jointAngles[jointIndex] = min;
                if (m_jointAngles[jointIndex] > max) m_jointAngles[jointIndex] = max;
                SendMessage(m_trackbars[jointIndex], TBM_SETPOS, TRUE, m_jointAngles[jointIndex]);
            }
        }
    }

    // Sets the range for the frame number up-down control
    void SetFrameRange(int min, int max) {
        m_frameMin = min;
        m_frameMax = max;
        if (m_upDownControl) {
            SendMessage(m_upDownControl, UDM_SETRANGE, 0, MAKELPARAM(max, min)); // High value in low word, low value in high word
            // Ensure current frame number is within new range
            if (m_frameNumber < min) m_frameNumber = min;
            if (m_frameNumber > max) m_frameNumber = max;
            SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
        }
    }

    // Sets the callback function to be invoked on value changes
    void SetOnManipulatorChangeCallback(OnManipulatorChangeCallback callback) {
        m_callback = callback;
    }

    // Get current joint angles
    const std::vector<int>& GetJointAngles() const {
        return m_jointAngles;
    }

    // Get current frame number
    int GetFrameNumber() const {
        return m_frameNumber;
    }

private:
    HINSTANCE m_hInstance;
    HWND m_hWnd;
    int m_windowWidth;
    int m_windowHeight;
    std::string m_title;
    int m_numberOfJoints;

    std::vector<HWND> m_trackbars;
    std::vector<HWND> m_jointLabels;
    std::vector<int> m_jointAngles;
    std::vector<std::pair<int, int>> m_jointMinMax; // min, max for each joint

    HWND m_upDownControl;
    HWND m_editControl;
    HWND m_frameLabel;
    int m_frameNumber;
    int m_frameMin = 0;
    int m_frameMax = 100; // Default frame range

    OnManipulatorChangeCallback m_callback;

    void CreateControls() {
        // Trackbars for joints
        int yPos = 20;
        int labelWidth = 80;
        int trackbarWidth = m_windowWidth - 120; // Adjust based on window size and padding
        int controlHeight = 25;
        int padding = 10;

        for (int i = 0; i < m_numberOfJoints; ++i) {
            // Joint label
            HWND hLabel = CreateWindowExA(
                0,
                "STATIC",
                ("Joint " + std::to_string(i + 1) + ":").c_str(),
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                padding, yPos, labelWidth, controlHeight,
                m_hWnd,
                NULL,
                m_hInstance,
                NULL
            );
            m_jointLabels.push_back(hLabel);

            // Trackbar
            HWND hTrackbar = CreateWindowExA(
                0,
                TRACKBAR_CLASSA, // Use TRACKBAR_CLASSA for narrow strings
                "",
                WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_TOOLTIPS | TBS_HORZ,
                padding + labelWidth, yPos, trackbarWidth, controlHeight,
                m_hWnd,
                (HMENU)(IDC_TRACKBAR_JOINT_BASE + i), // Unique ID for each trackbar
                m_hInstance,
                NULL
            );
            m_trackbars.push_back(hTrackbar);

            SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(m_jointMinMax[i].first, m_jointMinMax[i].second));
            SendMessage(hTrackbar, TBM_SETPAGESIZE, 0, 10); // Page size
            SendMessage(hTrackbar, TBM_SETTICFREQ, 10, 0); // Tick frequency
            SendMessage(hTrackbar, TBM_SETPOS, TRUE, m_jointAngles[i]); // Set initial position

            yPos += controlHeight + padding;
        }

        // Frame Number Up-Down control
        yPos += padding; // Extra padding before frame controls

        // Frame label
        m_frameLabel = CreateWindowExA(
            0,
            "STATIC",
            "Frame Number:",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            padding, yPos, labelWidth + 30, controlHeight,
            m_hWnd,
            NULL,
            m_hInstance,
            NULL
        );

        // Edit control for frame number (for direct input)
        m_editControl = CreateWindowExA(
            WS_EX_CLIENTEDGE, // Sunken edge for edit control
            "EDIT",
            std::to_string(m_frameNumber).c_str(),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER | ES_AUTOHSCROLL,
            padding + labelWidth + 30, yPos, 50, controlHeight,
            m_hWnd,
            (HMENU)IDC_EDIT_FRAME,
            m_hInstance,
            NULL
        );

        // Up-down control (spinner)
        m_upDownControl = CreateWindowExA(
            0,
            UPDOWN_CLASSA, // Use UPDOWN_CLASSA for narrow strings
            "",
            WS_CHILD | WS_VISIBLE | UDS_SETBUDDYINT | UDS_ALIGNRIGHT | UDS_AUTOBUDDY | UDS_ARROWKEYS | UDS_NOTHOUSANDS,
            0, 0, 0, 0, // Size and position are set by auto-buddy
            m_hWnd,
            (HMENU)IDC_UPDOWN_FRAME,
            m_hInstance,
            NULL
        );
        SendMessage(m_upDownControl, UDM_SETBUDDY, (WPARAM)m_editControl, 0); // Link to edit control
        SendMessage(m_upDownControl, UDM_SETRANGE, 0, MAKELPARAM(m_frameMax, m_frameMin)); // Max in high-word, min in low-word
        SendMessage(m_upDownControl, UDM_SETPOS, 0, MAKELPARAM(m_frameNumber, 0)); // Set initial position

        // Adjust window height if controls exceed initial height
        int requiredHeight = yPos + controlHeight + padding + 50; // Add some buffer
        if (requiredHeight > m_windowHeight) {
            SetWindowPos(m_hWnd, NULL, 0, 0, m_windowWidth, requiredHeight, SWP_NOMOVE | SWP_NOZORDER);
            m_windowHeight = requiredHeight;
        }
    }

    // Static Window Procedure: Trampoline to the instance's member function
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        Manipulator* pThis = nullptr;
        if (message == WM_NCCREATE) {
            LPCREATESTRUCT lpcs = reinterpret_cast<LPCREATESTRUCT>(lParam);
            pThis = static_cast<Manipulator*>(lpcs->lpCreateParams);
            SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
        }
        else {
            pThis = reinterpret_cast<Manipulator*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
        }

        if (pThis) {
            return pThis->WndProc(hWnd, message, wParam, lParam);
        }
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    // Instance Window Procedure: Handles messages for this Manipulator instance
    LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
        // g_activeWindowCount is assumed to be defined elsewhere (e.g., in a shared header)
        // For this specific file only, we'd need to declare it as extern or define it here if not shared.
        // Assuming it's part of a larger project and defined in a common global context.
        extern volatile LONG g_activeWindowCount;

        switch (message) {
        case WM_CREATE:
            InterlockedIncrement(&g_activeWindowCount);
            break;

        case WM_HSCROLL: // Message from trackbar
        {
            HWND hTrackbar = (HWND)lParam;
            int id = GetDlgCtrlID(hTrackbar);

            if (id >= IDC_TRACKBAR_JOINT_BASE && id < IDC_TRACKBAR_JOINT_BASE + m_numberOfJoints) {
                int jointIndex = id - IDC_TRACKBAR_JOINT_BASE;
                int pos = SendMessage(hTrackbar, TBM_GETPOS, 0, 0);
                m_jointAngles[jointIndex] = pos;

                // Update label with current value
                std::string labelText = "Joint " + std::to_string(jointIndex + 1) + ": " + std::to_string(pos);
                SetWindowTextA(m_jointLabels[jointIndex], labelText.c_str());

                if (m_callback) {
                    m_callback(m_jointAngles, m_frameNumber);
                }
            }
        }
        break;

        case WM_COMMAND: // Messages from controls (like up-down)
        {
            UINT id = LOWORD(wParam);
            UINT code = HIWORD(wParam);

            if (id == IDC_EDIT_FRAME && code == EN_CHANGE) {
                // This handles direct text input into the edit control
                char buffer[256];
                GetWindowTextA(m_editControl, buffer, sizeof(buffer));
                try {
                    int newFrame = std::stoi(buffer);
                    // Clamp to valid range
                    if (newFrame < m_frameMin) newFrame = m_frameMin;
                    if (newFrame > m_frameMax) newFrame = m_frameMax;

                    if (newFrame != m_frameNumber) {
                        m_frameNumber = newFrame;
                        if (m_callback) {
                            m_callback(m_jointAngles, m_frameNumber);
                        }
                    }
                }
                catch (const std::invalid_argument& e) {
                    // Handle invalid input (e.g., non-numeric) - maybe reset to old value
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
                }
                catch (const std::out_of_range& e) {
                    // Handle out of range input
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
                }
            }
            else if (id == IDC_UPDOWN_FRAME && code == UDN_DELTAPOS) {
                LPNMUPDOWN lpnmud = (LPNMUPDOWN)lParam;
                int newFrame = lpnmud->iPos + lpnmud->iDelta;

                // Clamp to valid range
                if (newFrame < m_frameMin) newFrame = m_frameMin;
                if (newFrame > m_frameMax) newFrame = m_frameMax;

                if (newFrame != m_frameNumber) {
                    m_frameNumber = newFrame;
                    // Update the buddy edit control
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());

                    if (m_callback) {
                        m_callback(m_jointAngles, m_frameNumber);
                    }
                }
                // Prevent default processing for UDN_DELTAPOS if you've handled it
                // The return value indicates whether the parent processed the notification
                return TRUE;
            }
        }
        break;

        case WM_SIZE:
        {
            m_windowWidth = LOWORD(lParam);
            m_windowHeight = HIWORD(lParam);
            // Reposition/resize controls here if layout needs to be dynamic
            // This would involve iterating through m_trackbars, m_jointLabels, etc.
            // and calling SetWindowPos for each. For simplicity, we'll assume fixed layout.
        }
        break;

        case WM_DESTROY:
            InterlockedDecrement(&g_activeWindowCount);
            m_hWnd = NULL;
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        return 0;
    }
};