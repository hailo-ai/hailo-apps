
################################################
# YOLO HAILORTPP POST SOURCES
################################################

yolo_hailortpp_sources = [
    'yolo_hailortpp.cpp'
]
shared_library('yolo_hailortpp_postprocess',
    yolo_hailortpp_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)
################################################
# REMOVE LABELS POST SOURCES
################################################

remove_labels_sources = [
    'remove_labels.cpp',
]
shared_library('remove_labels',
    remove_labels_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# yolov8pose_postprocess SOURCES
################################################

yolov8pose_postprocess_sources = [
    'yolov8pose_postprocess.cpp',
]
shared_library('yolov8pose_postprocess',
    yolov8pose_postprocess_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# YOLOV5SEG SOURCES
################################################
yolov5seg_post_sources = [
    'yolov5seg.cpp',
]

shared_library('yolov5seg_postprocess',
    yolov5seg_post_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# YOLOV8SEG SOURCES
################################################
yolov8seg_post_sources = [
    'yolov8seg.cpp',
]

shared_library('yolov8seg_postprocess',
    yolov8seg_post_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)


################################################
# depth_postprocess SOURCES
################################################

depth_postprocess_sources = [
    'depth_estimation.cpp',
]
shared_library('depth_postprocess',
    depth_postprocess_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# FACE ALIGN SOURCES
################################################
face_align_post_sources = [
    'face_align.cpp',
]

# Use static OpenCV if available, otherwise fall back to TAPPAS dynamic OpenCV
face_align_deps = [postprocess_dep]
if opencv_dep.found()
    face_align_deps += opencv_dep
endif
if zlib_dep.found()
    face_align_deps += zlib_dep
endif

shared_library('vms_face_align',
    face_align_post_sources,
    dependencies : face_align_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# SCRFD SOURCES
################################################
scrfd_post_sources = [
    'scrfd.cpp',
]

shared_library('scrfd',
    scrfd_post_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# ARCFACE SOURCES
################################################
arcface_post_sources = [
    'arcface.cpp',
]

shared_library('face_recognition_post',
    arcface_post_sources,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# VMS CROPPERS SOURCES
################################################
vms_croppers_sources = [
    'vms_croppers.cpp',
]

# Use static OpenCV if available
vms_croppers_deps = [postprocess_dep]
if opencv_dep.found()
    vms_croppers_deps += opencv_dep
endif
if zlib_dep.found()
    vms_croppers_deps += zlib_dep
endif

shared_library('vms_croppers',
    vms_croppers_sources,
    dependencies : vms_croppers_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# REPVGG REID POST SOURCES
################################################

repvgg_reid_sources = [
    'repvgg_reid.cpp'
]
shared_library('repvgg_reid_postprocess',
    repvgg_reid_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)
################################################
# ALL DETECTIONS CROPPER POST SOURCES
################################################

all_detections_cropper_sources = [
    'all_detections_cropper.cpp'
]

# Use static OpenCV if available
all_detections_cropper_deps = [postprocess_dep]
if opencv_dep.found()
    all_detections_cropper_deps += opencv_dep
endif

shared_library('all_detections_cropper_postprocess',
    all_detections_cropper_sources,
    include_directories : rapidjson_inc,
    dependencies : all_detections_cropper_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)
################################################
# TILING MOBILENET SSD POST SOURCES
################################################

mobilenet_ssd_sources = [
    'mobilenet_ssd.cpp'
]
shared_library('mobilenet_ssd_postprocess',
    mobilenet_ssd_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# CLIP SOURCES
################################################
clip_sources = [
    'clip.cpp',
]
shared_library('clip_postprocess',
    clip_sources,
    include_directories : rapidjson_inc,
    dependencies : postprocess_dep,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)    

################################################
# CLIP CROPPERS SOURCES
################################################
clip_croppers_sources = [
    'clip_croppers.cpp',
]

# Use static OpenCV if available
clip_croppers_deps = [postprocess_dep]
if opencv_dep.found()
    clip_croppers_deps += opencv_dep
endif

shared_library('clip_croppers_postprocess',
    clip_croppers_sources,
    include_directories : rapidjson_inc,
    dependencies : clip_croppers_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)

################################################
# OCR POST SOURCES
################################################

ocr_postprocess_sources = [
    'ocr_postprocess.cpp'
]

# Use static OpenCV if available
ocr_postprocess_deps = [postprocess_dep]
if opencv_dep.found()
    ocr_postprocess_deps += opencv_dep
endif
if zlib_dep.found()
    ocr_postprocess_deps += zlib_dep
endif

shared_library('ocr_postprocess',
    ocr_postprocess_sources,
    include_directories : rapidjson_inc,
    dependencies : ocr_postprocess_deps,
    gnu_symbol_visibility : 'default',
    install: true,
    install_dir: '/usr/local/hailo/resources/so',
)