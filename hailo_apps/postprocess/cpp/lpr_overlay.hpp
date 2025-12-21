/**
* Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
* Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
**/
#include <gst/video/video.h>
#include <opencv2/opencv.hpp>
#include "hailo_objects.hpp"

G_BEGIN_DECLS
void filter(HailoROIPtr roi, GstVideoFrame *frame);
G_END_DECLS