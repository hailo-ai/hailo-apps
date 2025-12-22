project('hailo-apps', 'c', 'cpp',
        version : '25-12',
        default_options : [ 'warning_level=1',
                            'buildtype=release',
                            'c_std=c11', 'cpp_std=c++17',
                            'c_args=-Wno-psabi',
                            'cpp_args=-Wno-psabi -Wno-class-memaccess -Wno-deprecated-declarations']
       )

postprocess_dep = dependency('hailo-tappas-core', version : '>=3.30.0', required : false)

if not postprocess_dep.found()
    postprocess_dep = dependency('hailo_tappas', version : '>=3.30.0', required : false)
endif

if postprocess_dep.found()
    rapidjson_inc_dir = postprocess_dep.get_variable(pkgconfig: 'rapidjson_includedir', default_value: '')
    rapidjson_inc = include_directories(rapidjson_inc_dir, is_system: true)

    subdir('cpp')
else
    message('⚠️  No Hailo TAPPAS dependencies found — skipping C++ postprocess build.')
endif
