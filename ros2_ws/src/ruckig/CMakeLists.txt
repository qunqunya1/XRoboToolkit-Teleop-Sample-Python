cmake_minimum_required(VERSION 3.15)

project(ruckig LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
_ament_cmake_python_register_environment_hook()

add_library(
  ruckig
  src/ruckig/brake.cpp
  src/ruckig/position_first_step1.cpp
  src/ruckig/position_first_step2.cpp
  src/ruckig/position_second_step1.cpp
  src/ruckig/position_second_step2.cpp
  src/ruckig/position_third_step1.cpp
  src/ruckig/position_third_step2.cpp
  src/ruckig/velocity_second_step1.cpp
  src/ruckig/velocity_second_step2.cpp
  src/ruckig/velocity_third_step1.cpp
  src/ruckig/velocity_third_step2.cpp)

target_compile_features(ruckig PUBLIC cxx_std_17)

target_include_directories(
  ruckig
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

add_library(ruckig::ruckig ALIAS ruckig)

if(MSVC)
  target_compile_definitions(ruckig PUBLIC _USE_MATH_DEFINES)
else()
  target_compile_options(ruckig PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

pybind11_add_module(
  ruckig_python
  MODULE
  src/ruckig/python.cpp)

target_link_libraries(ruckig_python PRIVATE ruckig)
target_compile_features(ruckig_python PRIVATE cxx_std_17)
set_target_properties(
  ruckig_python
  PROPERTIES OUTPUT_NAME ruckig
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")

install(
  TARGETS ruckig
  EXPORT ruckig_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include)

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.hpp")

install(
  TARGETS ruckig_python
  LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}")

ament_export_targets(ruckig_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(ament_cmake)
ament_export_dependencies(ament_cmake_python)
ament_export_include_directories(include)
ament_export_libraries(ruckig)

ament_package()
