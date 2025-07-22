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
#define IDC_TRACKBAR_POS_X 3000
#define IDC_TRACKBAR_POS_Y 3001
#define IDC_TRACKBAR_POS_Z 3002
#define IDC_TRACKBAR_ROT_X 3003
#define IDC_TRACKBAR_ROT_Y 3004
#define IDC_TRACKBAR_ROT_Z 3005

#define IDC_TRACKBAR_JOINT_BASE 1000
#define IDC_UPDOWN_FRAME 2000
#define IDC_EDIT_FRAME 2001

#define IDC_BUTTON_SAVE 4000
#define IDC_BUTTON_LOAD 4001

// Enum to specify the type of action in the callback
enum class ManipulatorAction {
    SLIDER_CHANGE,
    SAVE,
    LOAD
};

class Manipulator {
public:
    // Callback function type for when position, rotation, joint angles, frame number, or buttons change
    using OnManipulatorChangeCallback = std::function<void(ManipulatorAction action,
        const std::vector<int>& positionXYZ,
        const std::vector<int>& rotationXYZ,
        const std::vector<int>& jointAngles,
        int frameNumber)>;

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
        // Initialize position and rotation values
        m_positionValues.resize(3, 0); // X, Y, Z
        m_rotationValues.resize(3, 0); // X, Y, Z (Euler angles or similar)

        // Default ranges for position and rotation
        // For 0.1 increment, scale by 10. So -1.0 to +1.0 becomes -10 to +10.
        m_positionMinMax.resize(3, { -10, 10 }); // Default: -1.0 to +1.0 (scaled by 10)
        m_rotationMinMax.resize(3, { -180, 180 }); // Example: -180 to 180 degrees

        // Initialize joint angles to 0
        m_jointAngles.resize(m_numberOfJoints, 0);
        m_jointMinMax.resize(m_numberOfJoints, { 0, 360 }); // Default range for joints

        // Initialize Common Controls library (needed for trackbars and up-down controls)
        static bool commonControlsInitialized = false;
        if (!commonControlsInitialized) {
            INITCOMMONCONTROLSEX icc;
            icc.dwSize = sizeof(icc);
            icc.dwICC = ICC_BAR_CLASSES | ICC_UPDOWN_CLASS;
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
            WS_OVERLAPPEDWINDOW | WS_VSCROLL,
            initialX, initialY,
            m_windowWidth, initialHeight, // Use initialHeight here, adjust later
            NULL,
            NULL,
            hInstance,
            this
        );

        if (m_hWnd == NULL) {
            throw std::runtime_error("Manipulator Window Creation Failed!");
        }

        CreateControls();
    }

    // Destructor
    ~Manipulator() {
        if (m_hWnd && IsWindow(m_hWnd)) {
            DestroyWindow(m_hWnd);
        }
    }

    // Displays the window
    void Show(int nCmdShow) {
        ShowWindow(m_hWnd, nCmdShow);
        UpdateWindow(m_hWnd);
    }

    // Sets the range for a specific position trackbar (0=X, 1=Y, 2=Z)
    // Values are scaled by 10 internally for 0.1 increments
    void SetPositionRange(int index, double min, double max) {
        if (index >= 0 && index < 3) {
            int scaledMin = static_cast<int>(min * 10.0);
            int scaledMax = static_cast<int>(max * 10.0);
            m_positionMinMax[index] = { scaledMin, scaledMax };
            if (m_positionSliders[index]) {
                SendMessage(m_positionSliders[index], TBM_SETRANGEMIN, TRUE, scaledMin);
                SendMessage(m_positionSliders[index], TBM_SETRANGEMAX, TRUE, scaledMax);
                // Ensure current value is within new range (scaled)
                if (m_positionValues[index] < scaledMin) m_positionValues[index] = scaledMin;
                if (m_positionValues[index] > scaledMax) m_positionValues[index] = scaledMax;
                SendMessage(m_positionSliders[index], TBM_SETPOS, TRUE, m_positionValues[index]);
            }
        }
    }

    // Sets the range for a specific rotation trackbar (0=X, 1=Y, 2=Z)
    void SetRotationRange(int index, int min, int max) {
        if (index >= 0 && index < 3) {
            m_rotationMinMax[index] = { min, max };
            if (m_rotationSliders[index]) {
                SendMessage(m_rotationSliders[index], TBM_SETRANGEMIN, TRUE, min);
                SendMessage(m_rotationSliders[index], TBM_SETRANGEMAX, TRUE, max);
                if (m_rotationValues[index] < min) m_rotationValues[index] = min;
                if (m_rotationValues[index] > max) m_rotationValues[index] = max;
                SendMessage(m_rotationSliders[index], TBM_SETPOS, TRUE, m_rotationValues[index]);
            }
        }
    }

    // Sets the range for a specific joint trackbar
    void SetJointRange(int jointIndex, int min, int max) {
        if (jointIndex >= 0 && jointIndex < m_numberOfJoints) {
            m_jointMinMax[jointIndex] = { min, max };
            if (m_trackbars.size() > jointIndex && m_trackbars[jointIndex]) {
                SendMessage(m_trackbars[jointIndex], TBM_SETRANGEMIN, TRUE, min);
                SendMessage(m_trackbars[jointIndex], TBM_SETRANGEMAX, TRUE, max);
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
            SendMessage(m_upDownControl, UDM_SETRANGE, 0, MAKELPARAM(max, min));
            if (m_frameNumber < min) m_frameNumber = min;
            if (m_frameNumber > max) m_frameNumber = max;
            SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
        }
    }

    // Sets the callback function to be invoked on value changes or button clicks
    void SetOnManipulatorChangeCallback(OnManipulatorChangeCallback callback) {
        m_callback = callback;
    }

    // Get current position values (scaled integers)
    const std::vector<int>& GetPositionXYZRaw() const { return m_positionValues; }
    // Get current position values (converted to double)
    std::vector<double> GetPositionXYZ() const {
        std::vector<double> pos(3);
        for (int i = 0; i < 3; ++i) pos[i] = static_cast<double>(m_positionValues[i]) / 10.0;
        return pos;
    }
    // Get current rotation values
    const std::vector<int>& GetRotationXYZ() const { return m_rotationValues; }
    // Get current joint angles
    const std::vector<int>& GetJointAngles() const { return m_jointAngles; }
    // Get current frame number
    int GetFrameNumber() const { return m_frameNumber; }

    // --- New Public Setters to Update UI from External Code (e.g., Load) ---
    void SetCurrentPosition(const std::vector<double>& pos) {
        if (pos.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                m_positionValues[i] = static_cast<int>(pos[i] * 10.0);
                if (m_positionSliders[i]) {
                    SendMessage(m_positionSliders[i], TBM_SETPOS, TRUE, m_positionValues[i]);
                    const char* labels[] = { "Pos X:", "Pos Y:", "Pos Z:" };
                    std::string labelText = std::string(labels[i]) + " " + std::to_string(pos[i]);
                    SetWindowTextA(m_positionLabels[i], labelText.c_str());
                }
            }
        }
    }

    void SetCurrentRotation(const std::vector<int>& rot) {
        if (rot.size() == 3) {
            for (int i = 0; i < 3; ++i) {
                m_rotationValues[i] = rot[i];
                if (m_rotationSliders[i]) {
                    SendMessage(m_rotationSliders[i], TBM_SETPOS, TRUE, m_rotationValues[i]);
                    const char* labels[] = { "Rot X:", "Rot Y:", "Rot Z:" };
                    std::string labelText = std::string(labels[i]) + " " + std::to_string(rot[i]);
                    SetWindowTextA(m_rotationLabels[i], labelText.c_str());
                }
            }
        }
    }

    void SetCurrentJointAngles(const std::vector<int>& angles) {
        if (angles.size() == m_numberOfJoints) {
            for (int i = 0; i < m_numberOfJoints; ++i) {
                m_jointAngles[i] = angles[i];
                if (m_trackbars.size() > i && m_trackbars[i]) {
                    SendMessage(m_trackbars[i], TBM_SETPOS, TRUE, m_jointAngles[i]);
                    std::string labelText = "Joint " + std::to_string(i + 1) + ": " + std::to_string(angles[i]);
                    SetWindowTextA(m_jointLabels[i], labelText.c_str());
                }
            }
        }
    }

    void SetCurrentFrameNumber(int frame) {
        m_frameNumber = frame;
        if (m_editControl) {
            SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
            // Also update up-down control's position
            if (m_upDownControl) {
                SendMessage(m_upDownControl, UDM_SETPOS, 0, MAKELPARAM(m_frameNumber, 0));
            }
        }
    }
    // --- End New Public Setters ---

private:
    HINSTANCE m_hInstance;
    HWND m_hWnd;
    int m_windowWidth;
    int m_windowHeight;
    std::string m_title;
    int m_numberOfJoints;

    // Base position/rotation controls
    std::vector<HWND> m_positionSliders;
    std::vector<HWND> m_positionLabels;
    std::vector<int> m_positionValues; // Stored as scaled integers
    std::vector<std::pair<int, int>> m_positionMinMax; // Stored as scaled integers

    std::vector<HWND> m_rotationSliders;
    std::vector<HWND> m_rotationLabels;
    std::vector<int> m_rotationValues;
    std::vector<std::pair<int, int>> m_rotationMinMax;

    // Joint controls
    std::vector<HWND> m_trackbars;
    std::vector<HWND> m_jointLabels;
    std::vector<int> m_jointAngles;
    std::vector<std::pair<int, int>> m_jointMinMax;

    // Frame controls
    HWND m_upDownControl;
    HWND m_editControl;
    HWND m_frameLabel;
    int m_frameNumber;
    int m_frameMin = 0;
    int m_frameMax = 100;

    // Save/Load buttons
    HWND m_saveButton;
    HWND m_loadButton;

    OnManipulatorChangeCallback m_callback;

    void CreateControls() {
        int yPos = 20; // Starting Y position
        int labelWidth = 60; // Increased for "Pos X:" and "Rot X:"
        int jointLabelWidth = 100; // Wider for "Joint XX:"
        int trackbarWidth = m_windowWidth - 70;
        int controlHeight = 25;
        int padding = 10;
        int columnWidth = (m_windowWidth - 2 * padding) / 3; // Divide client area into 3 columns

        // --- Position (XYZ) Sliders ---
        const char* posLabels[] = { "Pos X:", "Pos Y:", "Pos Z:" };
        for (int i = 0; i < 3; ++i) {
            int xOffset = padding + i * columnWidth;

            HWND hLabel = CreateWindowExA(
                0, "STATIC", posLabels[i],
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                xOffset, yPos, labelWidth, controlHeight,
                m_hWnd, NULL, m_hInstance, NULL
            );
            m_positionLabels.push_back(hLabel);

            HWND hTrackbar = CreateWindowExA(
                0, TRACKBAR_CLASSA, "",
                WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_TOOLTIPS | TBS_HORZ,
                xOffset, yPos + controlHeight, columnWidth - padding, controlHeight,
                m_hWnd, (HMENU)(IDC_TRACKBAR_POS_X + i), m_hInstance, NULL
            );
            m_positionSliders.push_back(hTrackbar);

            SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(m_positionMinMax[i].first, m_positionMinMax[i].second));
            SendMessage(hTrackbar, TBM_SETPOS, TRUE, m_positionValues[i]);
            SendMessage(hTrackbar, TBM_SETTICFREQ, 10, 0); // Tick every 10 units (for 0.1 increment)
            SendMessage(hTrackbar, TBM_SETPAGESIZE, 0, 1); // Page size 1 unit (0.1 actual)
        }
        yPos += (controlHeight * 2) + padding;

        // --- Rotation (XYZ) Sliders ---
        const char* rotLabels[] = { "Rot X:", "Rot Y:", "Rot Z:" };
        for (int i = 0; i < 3; ++i) {
            int xOffset = padding + i * columnWidth;

            HWND hLabel = CreateWindowExA(
                0, "STATIC", rotLabels[i],
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                xOffset, yPos, labelWidth, controlHeight,
                m_hWnd, NULL, m_hInstance, NULL
            );
            m_rotationLabels.push_back(hLabel);

            HWND hTrackbar = CreateWindowExA(
                0, TRACKBAR_CLASSA, "",
                WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_TOOLTIPS | TBS_HORZ,
                xOffset, yPos + controlHeight, columnWidth - padding, controlHeight,
                m_hWnd, (HMENU)(IDC_TRACKBAR_ROT_X + i), m_hInstance, NULL
            );
            m_rotationSliders.push_back(hTrackbar);

            SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(m_rotationMinMax[i].first, m_rotationMinMax[i].second));
            SendMessage(hTrackbar, TBM_SETPOS, TRUE, m_rotationValues[i]);
            SendMessage(hTrackbar, TBM_SETTICFREQ, 10, 0); // Example: Tick every 10 degrees
            SendMessage(hTrackbar, TBM_SETPAGESIZE, 0, 10);
        }
        yPos += (controlHeight * 2) + padding;

        // --- Joint Sliders ---
        CreateWindowExA(
            0, "STATIC", "Joint Controls:",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            padding, yPos + padding, m_windowWidth - 2 * padding, controlHeight,
            m_hWnd, NULL, m_hInstance, NULL
        );
        yPos += controlHeight + 2 * padding;

        for (int i = 0; i < m_numberOfJoints; ++i) {
            HWND hLabel = CreateWindowExA(
                0, "STATIC", ("Joint " + std::to_string(i + 1) + ":").c_str(),
                WS_CHILD | WS_VISIBLE | SS_LEFT,
                padding, yPos, jointLabelWidth, controlHeight, // Use wider label for joints
                m_hWnd, NULL, m_hInstance, NULL
            );
            m_jointLabels.push_back(hLabel);

            HWND hTrackbar = CreateWindowExA(
                0, TRACKBAR_CLASSA, "",
                WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_TOOLTIPS | TBS_HORZ,
                padding + jointLabelWidth, yPos, trackbarWidth - (jointLabelWidth - labelWidth), controlHeight,
                m_hWnd, (HMENU)(IDC_TRACKBAR_JOINT_BASE + i), m_hInstance, NULL
            );
            m_trackbars.push_back(hTrackbar);

            SendMessage(hTrackbar, TBM_SETRANGE, TRUE, MAKELPARAM(m_jointMinMax[i].first, m_jointMinMax[i].second));
            SendMessage(hTrackbar, TBM_SETPAGESIZE, 0, 10);
            SendMessage(hTrackbar, TBM_SETTICFREQ, 10, 0);
            SendMessage(hTrackbar, TBM_SETPOS, TRUE, m_jointAngles[i]);

            yPos += controlHeight + padding;
        }

        // --- Frame Number Up-Down control ---
        yPos += padding;

        m_frameLabel = CreateWindowExA(
            0, "STATIC", "Frame Number:",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            padding, yPos, labelWidth + 30, controlHeight,
            m_hWnd, NULL, m_hInstance, NULL
        );

        m_editControl = CreateWindowExA(
            WS_EX_CLIENTEDGE, "EDIT", std::to_string(m_frameNumber).c_str(),
            WS_CHILD | WS_VISIBLE | WS_BORDER | ES_NUMBER | ES_AUTOHSCROLL,
            padding + labelWidth + 30, yPos, 50, controlHeight,
            m_hWnd, (HMENU)IDC_EDIT_FRAME, m_hInstance, NULL
        );

        m_upDownControl = CreateWindowExA(
            0, UPDOWN_CLASSA, "",
            WS_CHILD | WS_VISIBLE | UDS_SETBUDDYINT | UDS_ALIGNRIGHT | UDS_AUTOBUDDY | UDS_ARROWKEYS | UDS_NOTHOUSANDS,
            0, 0, 0, 0,
            m_hWnd, (HMENU)IDC_UPDOWN_FRAME, m_hInstance, NULL
        );
        SendMessage(m_upDownControl, UDM_SETBUDDY, (WPARAM)m_editControl, 0);
        SendMessage(m_upDownControl, UDM_SETRANGE, 0, MAKELPARAM(m_frameMax, m_frameMin));
        SendMessage(m_upDownControl, UDM_SETPOS, 0, MAKELPARAM(m_frameNumber, 0));

        yPos += controlHeight + padding;

        // --- Save and Load Buttons ---
        int buttonWidth = 80;
        int buttonHeight = 30;
        int buttonSpacing = 20;
        int buttonsX = (m_windowWidth - (2 * buttonWidth + buttonSpacing)) / 2; // Center buttons

        m_saveButton = CreateWindowExA(
            0, "BUTTON", "Save",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            buttonsX, yPos + padding, buttonWidth, buttonHeight,
            m_hWnd, (HMENU)IDC_BUTTON_SAVE, m_hInstance, NULL
        );

        m_loadButton = CreateWindowExA(
            0, "BUTTON", "Load",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            buttonsX + buttonWidth + buttonSpacing, yPos + padding, buttonWidth, buttonHeight,
            m_hWnd, (HMENU)IDC_BUTTON_LOAD, m_hInstance, NULL
        );

        yPos += buttonHeight + 2 * padding; // Move yPos past buttons

        // Adjust window height if controls exceed initial height
        int requiredHeight = yPos + 50; // Add some buffer at the bottom
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
        extern volatile LONG g_activeWindowCount; // Assuming this is defined globally

        switch (message) {
        case WM_CREATE:
            InterlockedIncrement(&g_activeWindowCount);
            break;

        case WM_HSCROLL: // Message from trackbar
        {
            HWND hTrackbar = (HWND)lParam;
            int id = GetDlgCtrlID(hTrackbar);
            int pos = SendMessage(hTrackbar, TBM_GETPOS, 0, 0);

            // Handle Position Sliders
            if (id >= IDC_TRACKBAR_POS_X && id <= IDC_TRACKBAR_POS_Z) {
                int index = id - IDC_TRACKBAR_POS_X;
                m_positionValues[index] = pos;
                const char* labels[] = { "Pos X:", "Pos Y:", "Pos Z:" };
                // Display scaled value in label
                std::string labelText = std::string(labels[index]) + " " + std::to_string(static_cast<double>(pos) / 10.0);
                SetWindowTextA(m_positionLabels[index], labelText.c_str());
                if (m_callback) {
                    m_callback(ManipulatorAction::SLIDER_CHANGE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                }
            }
            // Handle Rotation Sliders
            else if (id >= IDC_TRACKBAR_ROT_X && id <= IDC_TRACKBAR_ROT_Z) {
                int index = id - IDC_TRACKBAR_ROT_X;
                m_rotationValues[index] = pos;
                const char* labels[] = { "Rot X:", "Rot Y:", "Rot Z:" };
                std::string labelText = std::string(labels[index]) + " " + std::to_string(pos);
                SetWindowTextA(m_rotationLabels[index], labelText.c_str());
                if (m_callback) {
                    m_callback(ManipulatorAction::SLIDER_CHANGE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                }
            }
            // Handle Joint Sliders
            else if (id >= IDC_TRACKBAR_JOINT_BASE && id < IDC_TRACKBAR_JOINT_BASE + m_numberOfJoints) {
                int jointIndex = id - IDC_TRACKBAR_JOINT_BASE;
                m_jointAngles[jointIndex] = pos;
                std::string labelText = "Joint " + std::to_string(jointIndex + 1) + ": " + std::to_string(pos);
                SetWindowTextA(m_jointLabels[jointIndex], labelText.c_str());
                if (m_callback) {
                    m_callback(ManipulatorAction::SLIDER_CHANGE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                }
            }
        }
        break;

        case WM_COMMAND: // Messages from controls (like up-down, buttons)
        {
            UINT id = LOWORD(wParam);
            UINT code = HIWORD(wParam);

            if (id == IDC_EDIT_FRAME && code == EN_CHANGE) {
                char buffer[256];
                GetWindowTextA(m_editControl, buffer, sizeof(buffer));
                try {
                    int newFrame = std::stoi(buffer);
                    if (newFrame < m_frameMin) newFrame = m_frameMin;
                    if (newFrame > m_frameMax) newFrame = m_frameMax;

                    if (newFrame != m_frameNumber) {
                        m_frameNumber = newFrame;
                        if (m_callback) {
                            m_callback(ManipulatorAction::SLIDER_CHANGE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                        }
                    }
                }
                catch (const std::invalid_argument& e) {
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
                }
                catch (const std::out_of_range& e) {
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
                }
            }
            else if (id == IDC_UPDOWN_FRAME && code == UDN_DELTAPOS) {
                LPNMUPDOWN lpnmud = (LPNMUPDOWN)lParam;
                int newFrame = lpnmud->iPos + lpnmud->iDelta;

                if (newFrame < m_frameMin) newFrame = m_frameMin;
                if (newFrame > m_frameMax) newFrame = m_frameMax;

                if (newFrame != m_frameNumber) {
                    m_frameNumber = newFrame;
                    SetWindowTextA(m_editControl, std::to_string(m_frameNumber).c_str());
                    if (m_callback) {
                        m_callback(ManipulatorAction::SLIDER_CHANGE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                    }
                }
                return TRUE;
            }
            else if (id == IDC_BUTTON_SAVE) {
                if (m_callback) {
                    m_callback(ManipulatorAction::SAVE, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                }
            }
            else if (id == IDC_BUTTON_LOAD) {
                if (m_callback) {
                    m_callback(ManipulatorAction::LOAD, m_positionValues, m_rotationValues, m_jointAngles, m_frameNumber);
                }
            }
        }
        break;

        case WM_SIZE:
        {
            m_windowWidth = LOWORD(lParam);
            m_windowHeight = HIWORD(lParam);
            // For dynamic resizing, you would re-layout controls here.
            // For simplicity in this example, controls are laid out once.
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