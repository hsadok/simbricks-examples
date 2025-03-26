cmake_minimum_required(VERSION 3.5)
project(EnsoVta)

# C++ STD version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# C STD version
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Wconversion -Wsign-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

link_libraries(enso)
link_libraries(pthread)

if(NOT DEFINED ENV{TVM_HOME})
    message(FATAL_ERROR "Environment variable TVM_HOME is not set!")
endif()

find_library(TVM_RUNTIME_LIB tvm_runtime PATHS "$ENV{TVM_HOME}/build/")
if(NOT TVM_RUNTIME_LIB)
    message(FATAL_ERROR "Required library 'tvm_runtime' not found!")
endif()
link_libraries(${TVM_RUNTIME_LIB})

find_library(TVM_VTA_LIB vta PATHS "$ENV{TVM_HOME}/build/")
if(NOT TVM_VTA_LIB)
    message(FATAL_ERROR "Required library 'vta' not found!")
endif()
link_libraries(${TVM_VTA_LIB})

find_library(TVM_RUNNER_LIB tvm_runner PATHS "$ENV{TVM_HOME}/build/apps/cpp_rtvm")
if(NOT TVM_RUNNER_LIB)
    message(FATAL_ERROR "Required library 'tvm_runner' not found!")
endif()
link_libraries(${TVM_RUNNER_LIB})

add_executable(enso_vta_server enso_vta_server.cc)
add_executable(enso_vta_client enso_vta_client.cc)

target_include_directories(
  enso_vta_server
  SYSTEM PUBLIC $ENV{TVM_HOME}/include
  SYSTEM PUBLIC $ENV{TVM_HOME}/src
  SYSTEM PUBLIC $ENV{TVM_HOME}/apps/cpp_rtvm
  SYSTEM PUBLIC $ENV{TVM_HOME}/3rdparty/dlpack/include
  SYSTEM PUBLIC $ENV{TVM_HOME}/3rdparty/dmlc-core/include
)

target_include_directories(
  enso_vta_client
  SYSTEM PUBLIC $ENV{TVM_HOME}/include
  SYSTEM PUBLIC $ENV{TVM_HOME}/src
  SYSTEM PUBLIC $ENV{TVM_HOME}/apps/cpp_rtvm
  SYSTEM PUBLIC $ENV{TVM_HOME}/3rdparty/dlpack/include
  SYSTEM PUBLIC $ENV{TVM_HOME}/3rdparty/dmlc-core/include
)

cmake_policy(SET CMP0069 NEW) # suppress cmake warning about IPO

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)
if(result)
  set_property(TARGET enso_vta_server PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()
