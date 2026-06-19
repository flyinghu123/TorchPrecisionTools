# -*- coding: utf-8 -*-

import csv
import os
import shutil
import sysconfig

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

PTH_FILENAME = "torch_precision_tools.pth"
PTH_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), PTH_FILENAME)


def _install_pth(target_dir):
    dst = os.path.join(target_dir, PTH_FILENAME)
    if os.path.exists(dst):
        return
    shutil.copy2(PTH_SRC, dst)
    print(f"  -> Installed {PTH_FILENAME} to {target_dir}")


def _update_record(dist_info_dir, pth_path):
    record_path = os.path.join(dist_info_dir, "RECORD")
    if not os.path.exists(record_path):
        return
    with open(record_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    norm = os.path.normcase(pth_path)
    if any(os.path.normcase(r[0]) == norm for r in rows):
        return
    rows.append([pth_path, "", ""])
    with open(record_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  -> Registered {PTH_FILENAME} in RECORD for cleanup")


def _find_dist_info(site_packages):
    if not site_packages or not os.path.isdir(site_packages):
        return None
    for entry in os.listdir(site_packages):
        if entry.startswith("test_tools") and (
            entry.endswith(".dist-info") or entry.endswith(".egg-info")
        ):
            return os.path.join(site_packages, entry)
    return None


def _post_install(target_dir):
    _install_pth(target_dir)
    dist_info = _find_dist_info(target_dir)
    if dist_info:
        _update_record(dist_info, os.path.join(target_dir, PTH_FILENAME))


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        if hasattr(self, "install_lib") and self.install_lib:
            _post_install(self.install_lib)


setup(
    cmdclass={
        "install": PostInstallCommand,
    },
    data_files=[
        (sysconfig.get_paths()["purelib"], [PTH_FILENAME]),
    ],
)
