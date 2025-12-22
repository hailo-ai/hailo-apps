/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#pragma once
#include <map>
namespace common
{
    static std::map<uint8_t, std::string> dota_fifteen = {
        {0, "plane"},
        {1, "ship"},
        {2, "storage-tank"},
        {3, "baseball-diamond"},
        {4, "tennis-court"},
        {5, "basketball-court"},
        {6, "ground-track-field"},
        {7, "harbor"},
        {8, "bridge"},
        {9, "large-vehicle"},
        {10, "small-vehicle"},
        {11, "helicopter"},
        {12, "roundabout"},
        {13, "soccer-ball-field"},
        {14, "swimming-pool"}};
}
