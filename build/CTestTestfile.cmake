# CMake generated Testfile for 
# Source directory: /home/suman/agentic_coding/orthoseg
# Build directory: /home/suman/agentic_coding/orthoseg/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(segmentation "/home/suman/agentic_coding/orthoseg/build/test_seg")
set_tests_properties(segmentation PROPERTIES  _BACKTRACE_TRIPLES "/home/suman/agentic_coding/orthoseg/CMakeLists.txt;29;add_test;/home/suman/agentic_coding/orthoseg/CMakeLists.txt;0;")
add_test(document "/home/suman/agentic_coding/orthoseg/build/test_document")
set_tests_properties(document PROPERTIES  _BACKTRACE_TRIPLES "/home/suman/agentic_coding/orthoseg/CMakeLists.txt;33;add_test;/home/suman/agentic_coding/orthoseg/CMakeLists.txt;0;")
add_test(ui "/home/suman/agentic_coding/orthoseg/build/test_ui")
set_tests_properties(ui PROPERTIES  ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT "30" _BACKTRACE_TRIPLES "/home/suman/agentic_coding/orthoseg/CMakeLists.txt;42;add_test;/home/suman/agentic_coding/orthoseg/CMakeLists.txt;0;")
