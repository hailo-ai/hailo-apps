#!/usr/bin/env python3
"""
Configuration Verification Script

This script verifies that all configuration files can be loaded correctly
and are consistent with each other. Uses the unified config_manager.

Usage:
    python tests/verify_configs.py

For more comprehensive testing, use:
    python -m hailo_apps.config.config_manager --dry-run
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hailo_apps.config import config_manager
from hailo_apps.config.config_manager import ConfigPaths, ConfigError


def verify_configs() -> bool:
    """Verify all configuration files can be loaded and are consistent."""
    print("=" * 80)
    print("VERIFYING CONFIGURATION FILES")
    print("=" * 80)

    errors = []
    warnings = []

    # 1. Verify config file paths exist
    print(f"\n1. Checking configuration file paths...")
    config_files = [
        ("Main Config (config.yaml)", ConfigPaths.main_config()),
        ("Resources Config (resources_config.yaml)", ConfigPaths.resources_config()),
        ("Test Definition Config", ConfigPaths.test_definition_config()),
        ("Test Control Config", ConfigPaths.test_control_config()),
    ]

    for name, path in config_files:
        if path.exists():
            print(f"   ✓ {name}: {path}")
        else:
            if "Test Control" in name:
                print(f"   ⚠️ {name}: {path} (optional, not found)")
                warnings.append(f"{name} not found (optional)")
            else:
                print(f"   ❌ {name}: {path} (NOT FOUND)")
                errors.append(f"{name} not found at {path}")

    # 2. Load and verify main config
    print(f"\n2. Loading main configuration (config.yaml)...")
    try:
        main_config = config_manager.get_main_config()
        print(f"   ✓ Loaded successfully")
        print(f"   Keys: {list(main_config.keys())}")

        # Check valid versions
        valid_versions = main_config.get("valid_versions", {})
        print(f"   Valid HailoRT versions: {valid_versions.get('hailort', [])}")
        print(f"   Valid architectures: {valid_versions.get('hailo_arch', [])}")
    except ConfigError as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Main config error: {e}")

    # 3. Load and verify resources config
    print(f"\n3. Loading resources configuration (resources_config.yaml)...")
    try:
        apps = config_manager.get_available_apps()
        print(f"   ✓ Loaded successfully")
        print(f"   Available apps: {len(apps)}")
        print(f"   Apps: {', '.join(apps[:8])}{'...' if len(apps) > 8 else ''}")
        print(f"   Videos: {len(config_manager.get_videos())}")
        print(f"   Images: {len(config_manager.get_images())}")
    except ConfigError as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Resources config error: {e}")

    # 4. Load and verify test definition config
    print(f"\n4. Loading test definition configuration...")
    try:
        defined_apps = config_manager.get_defined_apps()
        test_suites = config_manager.get_all_test_suites()
        test_combinations = config_manager.get_all_test_run_combinations()

        print(f"   ✓ Loaded successfully")
        print(f"   Defined apps: {len(defined_apps)}")
        print(f"   Test suites: {len(test_suites)}")
        print(f"   Test combinations: {', '.join(test_combinations)}")
    except ConfigError as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Test definition config error: {e}")

    # 5. Load and verify test control config
    print(f"\n5. Loading test control configuration...")
    try:
        control_config = config_manager.get_test_control_config()
        if control_config:
            print(f"   ✓ Loaded successfully")
            print(f"   Keys: {list(control_config.keys())}")

            # Check test combinations
            test_combos = control_config.get("test_combinations", {})
            print(f"   Test combinations: {list(test_combos.keys())}")
        else:
            print(f"   ⚠️ Test control config not found (optional)")
            warnings.append("Test control config not found")
    except ConfigError as e:
        print(f"   ⚠️ Warning: {e}")
        warnings.append(f"Test control config: {e}")

    # 6. Cross-validation
    print(f"\n6. Cross-validating configurations...")

    # Check if test combinations in control exist in definition
    try:
        control_config = config_manager.get_test_control_config()
        if control_config:
            control_combos = set(control_config.get("test_combinations", {}).keys())
            definition_combos = set(config_manager.get_all_test_run_combinations())

            missing_combos = control_combos - definition_combos
            if missing_combos:
                msg = f"Control combinations not in definition: {missing_combos}"
                print(f"   ⚠️ {msg}")
                warnings.append(msg)
            else:
                print(f"   ✓ All control combinations found in definition")
    except Exception as e:
        warnings.append(f"Cross-validation error: {e}")

    # Check if custom test apps exist in definition
    try:
        custom_apps = set(config_manager.get_custom_test_apps().keys())
        defined_apps_set = set(config_manager.get_defined_apps())

        missing_apps = custom_apps - defined_apps_set
        if missing_apps:
            msg = f"Custom test apps not in definition: {missing_apps}"
            print(f"   ⚠️ {msg}")
            warnings.append(msg)
        else:
            print(f"   ✓ All custom test apps found in definition")
    except Exception as e:
        warnings.append(f"App validation error: {e}")

    # 7. Verify test suite references
    print(f"\n7. Verifying test suite references...")
    try:
        all_suites = set(config_manager.get_all_test_suites())
        referenced_suites = set()

        for app_name in config_manager.get_defined_apps():
            app_def = config_manager.get_app_definition(app_name)
            if app_def:
                referenced_suites.update(app_def.default_test_suites)
                referenced_suites.update(app_def.extra_test_suites)

        missing_suites = referenced_suites - all_suites
        if missing_suites:
            msg = f"Referenced test suites not found: {missing_suites}"
            print(f"   ❌ {msg}")
            errors.append(msg)
        else:
            print(f"   ✓ All referenced test suites exist")
            print(f"   Total referenced suites: {len(referenced_suites)}")
    except Exception as e:
        errors.append(f"Test suite validation error: {e}")

    # 8. Check app consistency between resources and definitions
    print(f"\n8. Checking app consistency...")
    try:
        resource_apps = set(config_manager.get_available_apps())
        definition_apps = set(config_manager.get_defined_apps())

        in_resources_only = resource_apps - definition_apps
        in_definition_only = definition_apps - resource_apps

        if in_resources_only:
            msg = f"Apps in resources but not definitions: {in_resources_only}"
            print(f"   ⚠️ {msg}")
            warnings.append(msg)

        if in_definition_only:
            msg = f"Apps in definitions but not resources: {in_definition_only}"
            print(f"   ⚠️ {msg}")
            warnings.append(msg)

        if not in_resources_only and not in_definition_only:
            print(f"   ✓ All apps consistent between configs")
    except Exception as e:
        warnings.append(f"App consistency error: {e}")

    # Summary
    print("\n" + "=" * 80)
    if errors:
        print(f"❌ VERIFICATION FAILED - {len(errors)} error(s)")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ ALL CONFIGURATION FILES VERIFIED SUCCESSFULLY")

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   • {warning}")

    print("=" * 80)
    return len(errors) == 0


if __name__ == "__main__":
    success = verify_configs()
    sys.exit(0 if success else 1)
