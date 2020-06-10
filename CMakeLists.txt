cmake_minimum_required (VERSION 3.8)

project ("FaceTracking")

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

# Add source to this project's executable.
add_executable (FaceTracking "FaceTracking.cpp")

target_link_libraries (FaceTracking ${OpenCV_LIBS} dlib::dlib)