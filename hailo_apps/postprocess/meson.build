project('hailo-apps', 'c', 'cpp',
        version : '25-12',
        default_options : [ 'warning_level=1',
                            'buildtype=release',
                            'c_std=c11', 'cpp_std=c++17',
                            'c_args=-Wno-psabi',
                            'cpp_args=-Wno-psabi -Wno-class-memaccess -Wno-deprecated-declarations']
       )

postprocess_dep = dependency('hailo-apps-core', version : '>=3.30.0', required : false)

if not postprocess_dep.found()
    postprocess_dep = dependency('hailo-tappas-core', version : '>=3.30.0', required : false)
endif

if not postprocess_dep.found()
    postprocess_dep = dependency('hailo_tappas', version : '>=3.30.0', required : false)
endif

zlib_dep = dependency('zlib', required : false)

# Try to find OpenCV4 for static linking
# Option 1: Prefer static OpenCV (if available)
# Set static: true to prefer static libraries
opencv_dep_static = dependency('opencv4', required : false, static : true)

# Option 2: If static OpenCV is not available, fall back to dynamic
# This ensures we can still build even if static libraries aren't installed
if opencv_dep_static.found()
    opencv_dep = opencv_dep_static
else
    opencv_dep = dependency('opencv4', required : false)
    if opencv_dep.found()
        message('⚠️  Static OpenCV not found, using dynamic OpenCV from TAPPAS')
    endif
endif

# Option 3: If you want to force static linking and fail if not available, use:
# opencv_dep = dependency('opencv4', required : true, static : true)

gst_dep = dependency('gstreamer-1.0', required : false)
gst_base_dep = dependency('gstreamer-base-1.0', required : false)
gst_video_dep = dependency('gstreamer-video-1.0', required : false)

# Resolve the system GStreamer plugin directory (same location as tappas-core plugins)
gst_plugins_dir = '/usr/lib/gstreamer-1.0'
if gst_dep.found()
    gst_plugins_dir_result = run_command('pkg-config', '--variable=pluginsdir', 'gstreamer-1.0', check: false)
    if gst_plugins_dir_result.returncode() == 0
        gst_plugins_dir = gst_plugins_dir_result.stdout().strip()
    endif
endif
message('GStreamer plugin install dir: @0@'.format(gst_plugins_dir))

if postprocess_dep.found()
    rapidjson_inc_dir = postprocess_dep.get_variable(pkgconfig: 'rapidjson_includedir', default_value: '')
    rapidjson_inc = include_directories(rapidjson_inc_dir, is_system: true)

    # Get the version to determine static vs dynamic linking strategy
    pkg_name = 'hailo-apps-core'
    pkg_version_result = run_command('pkg-config', '--modversion', pkg_name, check: false)
    if pkg_version_result.returncode() != 0
        pkg_name = 'hailo-tappas-core'
        pkg_version_result = run_command('pkg-config', '--modversion', pkg_name, check: false)
    endif
    if pkg_version_result.returncode() != 0
        pkg_name = 'hailo_tappas'
        pkg_version_result = run_command('pkg-config', '--modversion', pkg_name, check: false)
    endif
    
    tappas_version = pkg_version_result.returncode() == 0 ? pkg_version_result.stdout().strip() : 'unknown'
    message('Found TAPPAS core version: @0@'.format(tappas_version))
    
    # Determine if we should use static libraries (5.2.0+) or dynamic (5.1.0 and earlier)
    use_static = false
    if tappas_version != 'unknown'
        version_parts = tappas_version.split('.')
        if version_parts.length() >= 2
            major = version_parts[0].to_int()
            minor = version_parts[1].to_int()
            # Use static for 5.2.0+, dynamic for 5.1.0 and earlier
            use_static = (major > 5) or (major == 5 and minor >= 2)
        endif
    endif
    
    # Check if libhailo_opencv_utils.so exists (it may be missing on community installs)
    cc = meson.get_compiler('c')
    libhailo_opencv_utils_exists = cc.find_library('hailo_opencv_utils', required: false).found()
    
    if not libhailo_opencv_utils_exists
        message('⚠️  libhailo_opencv_utils.so not found, filtering it out from linker command')
        # Extract libraries from pkg-config and filter out libhailo_opencv_utils.so
        # Use --static flag if we want static libraries for 5.2.0+
        if use_static
            message('Using static libraries for tappas-core @0@'.format(tappas_version))
            pkg_libs_result = run_command('pkg-config', '--libs', '--static', pkg_name, check: false)
        else
            message('Using dynamic libraries for tappas-core @0@'.format(tappas_version))
            pkg_libs_result = run_command('pkg-config', '--libs', pkg_name, check: false)
        endif
        
        if pkg_libs_result.returncode() == 0
            pkg_libs_str = pkg_libs_result.stdout().strip()
            # Split by spaces, handling multiple spaces
            pkg_libs_list = pkg_libs_str.split()
            filtered_libs = []
            foreach lib : pkg_libs_list
                if not lib.contains('libhailo_opencv_utils')
                    filtered_libs += lib
                endif
            endforeach
            
            # Create a custom dependency with filtered libraries
            # Preserve the original dependency's properties (include dirs, compile args) but override link_args
            # This way we keep all the include directories from the original dependency
            postprocess_dep = declare_dependency(
                dependencies: postprocess_dep,
                link_args: filtered_libs
            )
        else
            message('⚠️  Could not extract libraries from pkg-config, using original dependency')
        endif
    elif use_static
        # Library exists, but we want static linking for 5.2.0+
        message('Using static libraries for tappas-core @0@ (libhailo_opencv_utils.so found)'.format(tappas_version))
        # Try to get static libraries
        pkg_libs_result = run_command('pkg-config', '--libs', '--static', pkg_name, check: false)
        if pkg_libs_result.returncode() == 0
            pkg_libs_str = pkg_libs_result.stdout().strip()
            pkg_libs_list = pkg_libs_str.split()
            
            # Preserve the original dependency's properties but override link_args with static libs
            postprocess_dep = declare_dependency(
                dependencies: postprocess_dep,
                link_args: pkg_libs_list
            )
        endif
    endif

    # TAPPAS 5.3.0 removed common:: utility functions (get_xtensor, softmax_2D,
    # vector_normalization, iou_calc, etc.) from libhailo_opencv_utils.so.
    # The implementations are now shipped as source files alongside the headers.
    # Build them into a static convenience library for downstream targets.
    fs = import('fs')
    common_impl_dir = '/usr/include/hailo/tappas/common'
    common_tensors_cpp = common_impl_dir / 'tensors.cpp'

    if fs.is_file(common_tensors_cpp)
        hailo_common_compat_lib = static_library('hailo_common_compat',
            [common_impl_dir / 'tensors.cpp',
             common_impl_dir / 'math.cpp',
             common_impl_dir / 'nms.cpp'],
            dependencies : postprocess_dep,
        )
        hailo_common_compat_dep = declare_dependency(link_with : hailo_common_compat_lib)
        message('Built hailo_common_compat from TAPPAS source files (5.3.0+ compat)')
    else
        # TAPPAS < 5.3.0 — symbols are in libhailo_opencv_utils.so, no extra lib needed
        hailo_common_compat_dep = declare_dependency()
        message('common:: sources not found — symbols in libhailo_opencv_utils.so (< 5.3.0)')
    endif

    subdir('cpp')
else
    message('⚠️  No Hailo TAPPAS dependencies found — skipping C++ postprocess build.')
endif
