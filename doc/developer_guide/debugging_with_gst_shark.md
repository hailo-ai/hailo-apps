# Debugging GStreamer Pipelines with GstShark

GstShark is a powerful profiling and debugging tool for GStreamer pipelines. It provides "tracers" that extract data from the pipeline in real-time, allowing you to monitor queue levels, latency, framerate, and CPU usage per element.

## 1. Installation

### Prerequisites

**System Requirements:**
*   GStreamer 1.0 installed and working
*   Build tools and development libraries

Install the required build tools and GStreamer development libraries.

**On Ubuntu / Debian / Raspberry Pi OS:**

```bash
sudo apt-get update
sudo apt-get install -y \
    git \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    autoconf \
    automake \
    libtool \
    graphviz \
    pkg-config \
    gtk-doc-tools
```

### Build and Install

We will build `gst-shark` from source.

1.  **Clone the repository:**
    ```bash
    # Clone to a convenient location (e.g., your home directory)
    cd ~
    git clone https://github.com/RidgeRun/gst-shark.git
    cd gst-shark
    ```

    **Note:** The `gst-shark` directory is needed later for generating plots. A good default location is `~/gst-shark` (your home directory).

2.  **Configure:**
    You must specify the correct library directory.

    *   **For x86_64 PCs (Standard Ubuntu):**
        ```bash
        ./autogen.sh --prefix=/usr/ --libdir=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/
        ```

    *   **For Raspberry Pi 5 (aarch64 / Debian Trixie):**
        ```bash
        ./autogen.sh --prefix=/usr/ --libdir=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/
        ```

    *   *Tip: If you are unsure of your architecture, run `uname -m`. `x86_64` uses the first command, `aarch64` uses the second.*

3.  **Compile and Install:**
    ```bash
    make
    sudo make install
    ```

### Verify Installation

Check if GStreamer can see the new tracers:

```bash
gst-inspect-1.0 sharktracers
```

You should see a list of tracers including `interlatency`, `proctime`, `framerate`, `queuelevel`, etc.

---

## 2. Real-Time Debugging (Console Output)

You can print trace information directly to the console to debug bottlenecks live.

### Monitoring Queue Levels

This is critical for diagnosing "stuck" pipelines or jitter.

**Example with Hailo applications:**
```bash
# Using a CLI command (e.g., simple detection)
GST_DEBUG="GST_TRACER:7" GST_TRACERS="queuelevel" hailo-detect-simple

# Or using Python script directly
GST_DEBUG="GST_TRACER:7" GST_TRACERS="queuelevel" python3 hailo_apps/python/pipeline_apps/detection_simple/detection_simple.py --input usb
```

**What to look for:**
*   **Bottleneck:** Queue level hits max (e.g., `30/30`) and stays there. The element *reading* from this queue is too slow.
*   **Starvation:** Queue level is always 0. The element *feeding* this queue is too slow.

### Other Useful Tracers

You can use multiple tracers by separating them with semicolons:

```bash
# Single tracer examples:
GST_TRACERS="interlatency" hailo-detect-simple    # Latency Spikes
GST_TRACERS="framerate" hailo-detect-simple        # Throughput
GST_TRACERS="proctime" hailo-detect-simple         # Processing Time

# Multiple tracers:
GST_DEBUG="GST_TRACER:7" GST_TRACERS="queuelevel;framerate;proctime" hailo-detect-simple
```

---

## 3. Generating Reports

GstShark can record pipeline data to a folder and generate reports in two formats:
*   **PDF Reports**: Static, printable charts suitable for documentation and sharing
*   **HTML/Interactive Reports**: Dynamic, interactive visualizations viewable in web browsers

Both formats use the same trace data, so you can generate both from a single recording session.

## 3.1 Generating PDF Reports

PDF reports provide static, printable charts that are excellent for documentation and sharing.

### Step 1: Install Plotting Dependencies

The plotting scripts require Octave, Gnuplot, Ghostscript, and Babeltrace2 (for reading CTF trace data).

```bash
sudo apt-get install -y octave gnuplot ghostscript babeltrace2
```

**Note:** `babeltrace2` is required to convert the CTF (Common Trace Format) trace data into a readable format for plotting.

### Step 2: Record the Trace Data (Same for Both PDF and HTML)

The trace data recording process is the same for both PDF and HTML reports:

Run your application with `GST_SHARK_LOCATION` set. This tells GstShark where to save the data.

**Example with Hailo applications:**
```bash
# Create a directory for the trace logs (default: /tmp/trace_data)
mkdir -p /tmp/trace_data

# Run the app with desired tracers (using CLI command)
GST_DEBUG="GST_TRACER:7" \
GST_TRACERS="proctime;interlatency;framerate;scheduletime;cpuusage;queuelevel" \
GST_SHARK_LOCATION=/tmp/trace_data \
hailo-detect-simple

# Or using Python script directly
GST_DEBUG="GST_TRACER:7" \
GST_TRACERS="proctime;interlatency;framerate;scheduletime;cpuusage;queuelevel" \
GST_SHARK_LOCATION=/tmp/trace_data \
python3 hailo_apps/python/pipeline_apps/detection_simple/detection_simple.py --input usb
```

**Notes:**
*   The default trace directory is `/tmp/trace_data` (you can use any directory path).
*   Run the app for at least 10-20 seconds to gather meaningful data.
*   Stop the app (Ctrl+C).
*   You will see a folder `/tmp/trace_data` containing a CTF (Common Trace Format) metadata file and binary logs.
*   Use an absolute path for `GST_SHARK_LOCATION` to avoid confusion about where the data is saved.

### Step 3: Generate the PDF Charts

Use the `gstshark-plot` script located in the `gst-shark` source directory you cloned earlier.

```bash
# Navigate to the directory where you cloned gst-shark (default: ~/gst-shark)
cd ~/gst-shark

# Run the plotting script with the absolute path to your trace_data directory
# The script will generate PDF charts in the trace_data directory
# Default example using /tmp/trace_data:
./scripts/graphics/gstshark-plot /tmp/trace_data --savefig pdf

# Or if you used a different directory:
./scripts/graphics/gstshark-plot /absolute/path/to/trace_data --savefig pdf
```

**Notes:**
*   The script must be run from the `gst-shark` directory (e.g., `~/gst-shark`).
*   Provide the **absolute path** to your `trace_data` directory.
*   Use `--savefig pdf` to save the charts as PDF files (default format).
*   The script generates a single `tracer.pdf` file containing all plots appended together.
*   The PDF is initially created in the `gst-shark` directory, but you can copy it to your trace_data directory:
    ```bash
    cp ~/gst-shark/tracer.pdf /tmp/trace_data/
    ```
*   The PDF contains multiple pages with charts for each tracer: queuelevel, framerate, cpuusage, proctime, interlatency, scheduling, bitrate, and buffer.

### Step 4: Analyze the PDF Results

Open the generated PDF (`tracer.pdf`) which contains multiple pages with charts for each tracer:

*   **Queue Level**: Shows the buffer count in every queue over time. Look for lines that plateau at the top (full) or bottom (empty).
*   **Frame Rate**: Shows the FPS at various points. Look for drops.
*   **CPU Usage**: Shows CPU load per element.
*   **Processing Time**: Shows how long each element takes to process a buffer. This helps identify the slowest specific element (e.g., is it the inference engine, the post-process, or the display?).
*   **Interlatency**: Shows latency between pipeline elements.
*   **Scheduling**: Shows scheduling time for elements.
*   **Bitrate**: Shows bitrate over time.
*   **Buffer**: Shows buffer processing time.

---

## 3.2 Generating Interactive HTML Reports

HTML reports provide interactive, web-based visualizations that allow you to zoom, pan, filter, and explore the trace data dynamically. This is ideal for detailed analysis and presentations.

### Step 1: Install HTML Report Dependencies

For interactive HTML reports, you only need `babeltrace2` which supports conversion to Chrome tracing format:

```bash
sudo apt-get install -y babeltrace2
```

**Note:** `babeltrace2` provides format conversion capabilities for CTF traces to interactive formats.

### Step 2: Record the Trace Data

Use the same trace data recording process as described in Section 3.1, Step 2. The trace data can be used for both PDF and HTML reports.

### Step 3: Convert to Chrome Tracing Format

Convert the CTF trace data to Chrome tracing JSON format using `babeltrace2`:

```bash
# First, check available output formats (sink components)
babeltrace2 list-plugins

# Convert CTF trace to text format (babeltrace2 may not have Chrome tracing sink by default)
# Output as text which can then be converted to Chrome tracing format
babeltrace2 convert /tmp/trace_data -c sink.text.pretty > /tmp/trace_data/trace.txt
```

**Note:** Standard `babeltrace2` installations may not include a Chrome tracing sink. Available sinks typically include:
- `sink.text.pretty` - Pretty-print text format (default)
- `sink.text.details` - Detailed text output
- `sink.ctf.fs` - Write CTF traces to file system

**Alternative approaches for Chrome tracing format:**

1. **Use Trace Compass** (recommended): Trace Compass can directly open CTF traces and provides interactive visualization without conversion.

2. **Convert text output**: Use the text output from `babeltrace2` and convert it to Chrome tracing JSON format using custom scripts or tools.

3. **Check for additional plugins**: Some distributions or custom builds may include Chrome tracing sinks. Check `babeltrace2 list-plugins` for available sinks in your installation.

### Step 4: View Interactive Reports

**Recommended: Using Trace Compass**

Trace Compass is the recommended tool for interactive CTF trace visualization as it works directly with CTF traces without conversion:

1.  Install Trace Compass: Download from [https://www.eclipse.org/tracecompass/](https://www.eclipse.org/tracecompass/)
2.  Open Trace Compass
3.  File → Open Trace → Select `/tmp/trace_data` directory
4.  Explore the trace data with advanced filtering and analysis features

**Interactive Features:**
*   **Zoom**: Use mouse wheel or trackpad to zoom in/out on specific time ranges
*   **Pan**: Click and drag to navigate through the timeline
*   **Filter**: Search for specific elements or events
*   **Details**: Click on any event to see detailed information
*   **Measure**: Select time ranges to measure durations between events
*   **Advanced Analysis**: Multiple views and analysis tools for performance profiling

**Alternative: Chrome Tracing (if conversion available)**

If you have successfully converted the trace to Chrome tracing JSON format, you can view it in Chrome's built-in trace viewer:

1.  Open Google Chrome browser
2.  Navigate to `chrome://tracing`
3.  Click "Load" button
4.  Select the generated `trace.json` file
5.  Explore the interactive timeline

**Note:** Chrome tracing format conversion may require additional plugins or custom conversion scripts, as standard `babeltrace2` installations may not include Chrome tracing sinks. Trace Compass is recommended as it works directly with CTF traces.

---

## 3.3 Comparison: PDF vs HTML Reports

| Feature           | PDF Reports                              | HTML/Interactive Reports                             |
| ----------------- | ---------------------------------------- | ---------------------------------------------------- |
| **Format**        | Static PDF document                      | Interactive JSON/HTML                                |
| **Viewing**       | PDF viewer (any platform)                | Chrome browser (`chrome://tracing`) or Trace Compass |
| **Interactivity** | None (static pages)                      | Full (zoom, pan, filter, search)                     |
| **Sharing**       | Easy (single file)                       | Easy (single JSON file)                              |
| **Printing**      | Excellent (formatted pages)              | Limited (screenshots)                                |
| **Documentation** | Ideal for reports and presentations      | Ideal for detailed analysis                          |
| **File Size**     | Small (compressed PDF)                   | Medium (JSON format)                                 |
| **Best For**      | Documentation, sharing, offline analysis | Interactive exploration, debugging, presentations    |

**Recommendation:**
*   Use **PDF reports** when you need printable documentation, formal reports, or offline analysis
*   Use **HTML/Interactive reports** when you need to explore data in detail, zoom into specific time ranges, or present findings interactively

Both formats can be generated from the same trace data, so you can create both for comprehensive analysis.

---

## 4. External Documentation

For deeper details on interpreting every metric, refer to the official documentation:

*   **GstShark Wiki:** [https://developer.ridgerun.com/wiki/index.php?title=GstShark](https://developer.ridgerun.com/wiki/index.php?title=GstShark)
*   **GstShark GitHub:** [https://github.com/RidgeRun/gst-shark](https://github.com/RidgeRun/gst-shark)
