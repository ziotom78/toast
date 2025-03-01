
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>


int toast::test_runner(int argc, char ** argv) {
    ::testing::GTEST_FLAG(filter) = std::string("TOAST*");
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
