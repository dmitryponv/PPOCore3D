
#define B3_USE_ROBOTSIM_GUI

#ifdef B3_USE_ROBOTSIM_GUI
#include "b3RobotSimulatorClientAPI.h"
#else
#include "b3RobotSimulatorClientAPI_NoGUI.h"
#endif

#include "../Utils/b3Clock.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>
#define ASSERT_EQ(a, b) assert((a) == (b));
#include "MinitaurSetup.h"


class RobotSimulator
{
	public:
		int run_simulator();
};
