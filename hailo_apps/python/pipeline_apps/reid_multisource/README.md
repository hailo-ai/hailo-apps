# REID Multisource Application

> ⚠️ **Beta:** This application is currently in beta. Features and APIs may change.

#### Run the REID example:
```bash
hailo-reid
```
To close the application, press `Ctrl+C`.

This is an advanced application, utilizing concepts from two other applications:

*   Multisource
*   Face recognition

Therefore, it is recommended to first gain a deep understanding of those applications.

The overall task is to track people (faces) across multiple cameras (input sources / streams). So that a specific person who first appeared on stream X will be classified and recognized as the same person when, and if, they appear on stream Y.

The pipeline is rather complex, combining elements from multisource and face recognition pipelines.

Similar to the face recognition application, dedicated networks assign unique embedding for each face, which is stored in LanceDB. Every detected face is first queried against the database for recognition.

The available arguments are similar to the multisource application and include `--sources`, `--width` & `--height`, while `--frame-rate` is hardcoded to be 15 for optimization of performance. 

Please note that the DB is populated persistently on each run, and will remember the faces. For a fresh start, simply delete the database files.

### All pipeline commands support these common arguments:

[Common arguments](../../../../doc/user_guide/running_applications.md#command-line-argument-reference)

For additional options, execute:
```bash
hailo-reid --help
```