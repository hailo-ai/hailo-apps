# Writing Your Own Post-Process for Hailo Apps

So you want to add a new network to hailo-apps that isn't supported yet? You've come to the right place! This guide will walk you through creating your own post-process. Don't worry—it's easier than it sounds.
This guide describes how to create a post-process function written in C++ that can be used in the HailoApps framework. This is done using the 'hailofilter' element.
**Note** For python post-process, you can add your post-process function as a python callback function.

## Getting Started
The post-process function should read the outputs of the HailoNet element, which are tensors attached to the HailoROI object. The post-process function should remove the tensors from the HailoROI object and replace them with the post-process results. i.e. HailoDetection, HailoClassification, etc.

### Where to Put Your Files

First things first - navigate to this directory:
```
hailo_apps/postprocess/cpp
```

All your new post-processing magic will happen in the `postprocesses/` folder.

### Creating Your Header File

Let's start by creating a header file called `my_post.hpp`. Think of this as your blueprint - it tells the system what your post-process can do.

Here's what you need to include at the top:

```cpp
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
```

**What do these do?**
- `hailo_objects.hpp` - Contains all the classes for handling different types of data (tensors, detections, classifications, etc.)
- `hailo_common.hpp` - Provides helpful utility functions for working with these classes

Now add your function prototype. Your complete header file should look like this:

```cpp
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"

__BEGIN_DECLS
void filter(HailoROIPtr roi);
__END_DECLS
```

That's it! The `hailofilter` element only expects one thing from you - a `filter` function that takes a `HailoROIPtr` parameter. This pointer gives you access to all the data for each image that passes through.

### Writing Your First Filter

Now let's create the actual implementation. Make a new file called `my_post.cpp` and start with these includes:

```cpp
#include <iostream>
#include "my_post.hpp"
```

For now, let's create a simple filter that just prints a message:

```cpp
// Your first post-processing function!
void filter(HailoROIPtr roi)
{
    std::cout << "My first postprocess!" << std::endl;
}
```

Congratulations! You now have a working post-process. It doesn't do much yet, but it's a start.

## Building and Testing Your Code

### Setting Up the Build System

Hailo uses Meson (a build system that's fast and easy to use) to compile everything. You'll need to tell Meson about your new post-process.

Find the `meson.build` file in `libs/postprocesses` and add an entry for your post-process. Here's an example of what it should look like:

```cpp
################################################
# MY CUSTOM POST PROCESS
################################################

my_post_sources = [
    'my_post.cpp'
]
shared_library('my_post',
    my_post_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)
```

### Compiling Your Post-Process

To compile your new post-process, run:
```bash
hailo-compile-postprocess
```
or use:
```bash
./compile_postprocess.sh
```

### Testing It Out

Ready to see your post-process in action? Run this test pipeline:

```bash
gst-launch-1.0 videotestsrc ! hailofilter so-path=/usr/local/hailo/resources/so/libmy_post.so ! fakesink
```

You should see "My first postprocess!" printed to your console. Pretty cool, right?

## Working with Real Data

### Accessing Tensor Information

Printing messages is fun, but let's do something more useful. Replace your print statement with code that actually examines the network's output:

```cpp
void filter(HailoROIPtr roi)
{
    // Get all the output tensors from the network
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();

    // Let's look at the first tensor
    HailoTensorPtr first_tensor = tensors[0];
    std::cout << "Tensor: " << first_tensor->name();
    std::cout << " has width: " << first_tensor->shape()[0];
    std::cout << " height: " << first_tensor->shape()[1];
    std::cout << " channels: " << first_tensor->shape()[2] << std::endl;
}
```

**What's happening here?**
- `roi->get_tensors()` gives you all output tensors as a vector
- `roi->get_tensor("name")` lets you get a specific tensor by name
- Each tensor contains metadata like dimensions and quantization parameters
- The actual tensor data is accessible via `tensor->data()`

**Pro tip:** The raw data comes as `uint8_t`, so you'll need to dequantize it to `float` for full precision using `tensor->fix_scale(uint8_t num)`.

### Creating Detection Objects

Now let's create some detection boxes! Replace your tensor examination code with:

```cpp
void filter(HailoROIPtr roi)
{
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();

    // Create some demo detections
    std::vector<HailoDetection> detections = demo_detection_objects();

    // Add them to the frame
    hailo_common::add_detections(roi, detections);
}
```

Now add this helper function to create demo detections:

```cpp
std::vector<HailoDetection> demo_detection_objects()
{
    std::vector<HailoDetection> objects;

    // Create two detection boxes
    HailoDetection first_detection = HailoDetection(
        HailoBBox(0.2, 0.2, 0.2, 0.2), // x, y, width, height (as percentages)
        "person",                        // label
        0.99                            // confidence
    );

    HailoDetection second_detection = HailoDetection(
        HailoBBox(0.6, 0.6, 0.2, 0.2),
        "person",
        0.89
    );

    objects.push_back(first_detection);
    objects.push_back(second_detection);

    return objects;
}
```

**Important note:** The bounding box coordinates are percentages of the image size (0.0 to 1.0), not pixel values. This ensures your detections remain accurate even when the image is resized.

## Seeing Your Results

### Adding Visual Output

Your post-process is now creating detection boxes, but you can't see them yet. To make them visible, add the `hailooverlay` element to your pipeline:

```bash
gst-launch-1.0 filesrc location=your_video.mp4 ! decodebin ! videoscale ! \
video/x-raw, pixel-aspect-ratio=1/1 ! videoconvert ! queue ! \
hailonet hef-path=your_model.hef is-active=true ! queue ! \
hailofilter so-path=/usr/local/hailo/resources/so/libmy_post.so qos=false ! \
queue ! hailooverlay ! videoconvert ! \
fpsdisplaysink video-sink=ximagesink sync=true text-overlay=false
```

Now you should see your video with two detection boxes labeled "person" with their confidence scores!

## Advanced Features

### Multiple Functions in One Post-Process

Sometimes you need different variations of the same post-process for different networks. Instead of creating separate `.so` files, you can include multiple functions in one file.

Just declare them in your header:

```cpp
__BEGIN_DECLS
void filter(HailoROIPtr roi);                    // Default function
void yolov5(HailoROIPtr roi, void *params);      // YOLOv5 specific
void yolov8(HailoROIPtr roi, void *params);      // YOLOv8 specific
__END_DECLS
```

Then tell the `hailofilter` which function to use:

```bash
hailofilter function-name=yolov5 so-path=libmy_post.so
```

## Complete Example

Here's what your final `my_post.cpp` should look like:

```cpp
#include <iostream>
#include "my_post.hpp"

std::vector<HailoDetection> demo_detection_objects()
{
    std::vector<HailoDetection> objects;

    HailoDetection first_detection = HailoDetection(
        HailoBBox(0.2, 0.2, 0.2, 0.2), "person", 0.99
    );

    HailoDetection second_detection = HailoDetection(
        HailoBBox(0.6, 0.6, 0.2, 0.2), "person", 0.89
    );

    objects.push_back(first_detection);
    objects.push_back(second_detection);

    return objects;
}

void filter(HailoROIPtr roi)
{
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();
    std::vector<HailoDetection> detections = demo_detection_objects();
    hailo_common::add_detections(roi, detections);
}
```