cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(PPOCore3D)

#############################################
############## BUILD OPTIONS ################
#############################################
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)
set(CAFFE2_USE_CUDSS 1)
set(CAFFE2_USE_CUFILE 1)

#############################################
############### TORCH SETUP #################
#############################################
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_d/share/cmake/Torch")
else()
    set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/share/cmake/Torch")
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/torch.cmake")

############################################
############# EDIT AND CONTINUE ############
############################################
if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /ZI")
endif()

#############################################
############## SOURCE FILES #################
#############################################
set(RobotSimulator_SRCS
    include/minitaur/RobotSimulator.cpp
    include/minitaur/b3RobotSimulatorClientAPI.cpp
    include/minitaur/MinitaurSetup.cpp
    include/bullet/ExampleBrowser/InProcessExampleBrowser.cpp
    include/bullet/SharedMemory/GraphicsServerExample.cpp
    include/bullet/SharedMemory/GraphicsClientExample.cpp
    include/bullet/SharedMemory/RemoteGUIHelper.cpp
    include/bullet/SharedMemory/RemoteGUIHelperTCP.cpp
    include/bullet/SharedMemory/PhysicsServerExample.cpp
    include/bullet/SharedMemory/PhysicsServerExampleBullet2.cpp
    include/bullet/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp
)

add_executable(PPOCore3D WIN32 src/main.cpp src/ppo.cpp ${RobotSimulator_SRCS})
set_property(TARGET PPOCore3D PROPERTY CXX_STANDARD 17)
target_link_libraries(PPOCore3D "${TORCH_LIBRARIES}")

#############################################
############## INCLUDE PATHS ################
#############################################
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/minitaur
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet/SharedMemory
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet/ThirdPartyLibs
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet/ThirdPartyLibs/enet/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bullet/ThirdPartyLibs/clsocket/src
)

#############################################
######## FORCE USE OF DYNAMIC CRT ###########
#############################################
foreach(flag_var CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG)
    if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
    endif()
    if(${flag_var} MATCHES "/MTd")
        string(REGEX REPLACE "/MTd" "/MDd" ${flag_var} "${${flag_var}}")
    endif()
endforeach()

#############################################
################ DEFINITIONS ################
#############################################
add_definitions(-DGLEW_STATIC)
if(BUILD_CLSOCKET)
    add_definitions(-DBT_ENABLE_CLSOCKET)
    add_definitions(${OSDEF})
endif()

#############################################
############ LIBRARY DIRECTORIES ############
#############################################
set(LIB_DIR_DEBUG "${CMAKE_CURRENT_LIST_DIR}/lib/Debug")
set(LIB_DIR_RELEASE "${CMAKE_CURRENT_LIST_DIR}/lib/Release")

target_link_directories(PPOCore3D
    PRIVATE
    $<$<CONFIG:Debug>:${LIB_DIR_DEBUG}>
    $<$<CONFIG:Release>:${LIB_DIR_RELEASE}>
)

#############################################
############## BULLET LIBS ##################
#############################################
set(BULLET_LIBS_DEBUG
    BulletRobotics_DEBUG
    BulletExampleBrowserLib_DEBUG
    BulletFileLoader_DEBUG
    BulletWorldImporter_DEBUG
    BulletSoftBody_DEBUG
    BulletDynamics_DEBUG
    BulletCollision_DEBUG
    BulletInverseDynamicsUtils_DEBUG
    BulletInverseDynamics_DEBUG
    LinearMath_DEBUG
    OpenGLWindow_DEBUG
    gwen_DEBUG
    Bullet3Common_DEBUG
    Winmm
)

set(BULLET_LIBS_RELEASE
    BulletRobotics
    BulletExampleBrowserLib
    BulletFileLoader
    BulletWorldImporter
    BulletSoftBody
    BulletDynamics
    BulletCollision
    BulletInverseDynamicsUtils
    BulletInverseDynamics
    LinearMath
    OpenGLWindow
    gwen
    Bullet3Common
    Winmm
)

target_link_libraries(PPOCore3D
    $<$<CONFIG:Debug>:${BULLET_LIBS_DEBUG}>
    $<$<CONFIG:Release>:${BULLET_LIBS_RELEASE}>
    ws2_32
    opengl32
)

#############################################
############ COPY TORCH DLLs ################
#############################################
set(LIBTORCH_DLL_DIR_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_d/lib")
set(LIBTORCH_DLL_DIR_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_r/lib")

file(GLOB TORCH_DLLS_DEBUG "${LIBTORCH_DLL_DIR_DEBUG}/*.dll" "${LIB_DIR_DEBUG}/*.dll")
file(GLOB TORCH_DLLS_RELEASE "${LIBTORCH_DLL_DIR_RELEASE}/*.dll" "${LIB_DIR_RELEASE}/*.dll")

set(DEST_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/build/Debug")
set(DEST_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/build/Release")

file(MAKE_DIRECTORY "${DEST_DEBUG}")
file(MAKE_DIRECTORY "${DEST_RELEASE}")

foreach(dll ${TORCH_DLLS_DEBUG})
    message(STATUS "Copying if newer ${dll} to ${DEST_DEBUG}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${DEST_DEBUG}")
endforeach()

foreach(dll ${TORCH_DLLS_RELEASE})
    message(STATUS "Copying if newer ${dll} to ${DEST_RELEASE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${DEST_RELEASE}")
endforeach()
