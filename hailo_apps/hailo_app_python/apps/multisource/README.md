# Multisource Application

#### Run the multisource example:
```bash
hailo-multisource
```
To close the application, press `Ctrl+C`.

This GStreamer pipeline demonstrates object detection on multiple camera streams over USB cameras / RTSP protocol / Files, etc.

All the streams are processed in parallel through the decode and scale phases, and enter the Hailo device frame by frame.

Afterwards, the post-process and drawing phases add the classified object and bounding boxes to each frame. The last step is to match each frame back to its respective stream and output all of them to the display.

In addition to the common callback function object that exists in other applications, In the multisource pipeline, there are stream-specific callback functions.

A detailed diagram of an example pipeline sourced from two USB cameras: ![Pipeline Example](../../../../doc/images/multisource_pipeline.png)
File can be found here: `hailo-apps/doc/images/multisource_pipeline.png`

The pipeline is using [`HailoRoundRobin`](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_roundrobin.rst) & [`HailoStreamRouter`](https://github.com/hailo-ai/tappas/blob/master/docs/elements/hailo_stream_router.rst) elements.

#### IMPORTANT: 
The input is provided via `--source` arguments, and the sources are separated with a comma (see below examples). This is in contrast to other applications where the argument --input is used.

#### Usage limitations:

At the end of the day, it's a simple balance between:

*   Platform resources: Raspberry Pi vs x86 etc.
*   Number and type of sources: On Raspberry Pi up to three sources are optimal
*   Frame rate of the sources: 15 FPS is recommended via `--frame-rate` argument
*   Resolution of the sources: There are dedicated `--width` & `--height` arguments defaulted to 640 x 640 pixels

#### Input examples:

```bash
python multisource.py --sources source_1,...,source_n
```

Currently all sources must be from the same type.
For example:

```bash
python multisource.py --sources source_1,...,source_n
python multisource.py --sources /dev/video0,/dev/video2
python multisource.py --sources ~/hailo-apps/resources/videos/example.mp4,~/hailo-apps/resources/videos/example.mp4
```

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)

For additional options, execute:
```bash
hailo-multisource --help
```