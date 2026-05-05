
add_requires("vcpkg::openblas")

-- make a openblas project
target("openblas_bench")
    set_languages("c++17")
    set_kind("binary")

    add_files("main_blas.cpp")
    -- add_vectorexts("avx2", "fma")

    add_packages("vcpkg::openblas")

    -- add flag
    add_options("-O3")
    add_options("-mavx2")
    add_options("-mfma")

    if is_plat("windows") then
        add_syslinks("user32", "gdi32")
    end
