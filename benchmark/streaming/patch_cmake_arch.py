#!/usr/bin/env python3
"""
Патч для FindCUDA/select_compute_arch.cmake (CMake 3.28): регулярное выражение
номера архитектуры допускает только однозначный major (`[0-9]\\.[0-9]`), поэтому
Blackwell «12.0» / «12.1» отвергается как «Unknown CUDA Architecture Name».
Разрешаем `[0-9]+`. CTranslate2 использует именно этот устаревший модуль
(find_package(CUDA) + cuda_select_nvcc_arch_flags), новый CMAKE_CUDA_ARCHITECTURES
он игнорирует.
"""
import glob
import sys

files = glob.glob("/usr/share/cmake-*/Modules/FindCUDA/select_compute_arch.cmake")
if not files:
    sys.exit("select_compute_arch.cmake not found")
old = '"^([0-9]\\\\.[0-9](\\\\([0-9]\\\\.[0-9]\\\\))?)$"'
new = '"^([0-9]+\\\\.[0-9](\\\\([0-9]+\\\\.[0-9]\\\\))?)$"'
for p in files:
    s = open(p).read()
    if old not in s:
        sys.exit(f"pattern not found in {p}")
    open(p, "w").write(s.replace(old, new))
    print("patched", p)
