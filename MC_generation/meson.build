project('MC_simulations', 'cpp', default_options:'optimization=3')
add_global_arguments('-DDISABLE_INMC_PARALLEL', language:'cpp')
add_global_arguments('-O3', language:'cpp')

mpi = dependency('openmp')

incdir = include_directories('include/')
dependencies = [mpi]
MC = library('MC', ['src/MC/quantum.cpp'], include_directories: incdir, install:true, dependencies: dependencies)

qho = executable('qho', 'src/qho.cpp', link_with:[MC], include_directories: incdir, dependencies: dependencies)
qho_autocorr = executable('qho_autocorr', 'src/qho_autocorr.cpp', link_with:[MC], include_directories: incdir, dependencies: dependencies)
deltatH_qho = executable('deltatH_qho', 'src/deltatH_qho.cpp', link_with:[MC], include_directories: incdir, dependencies: dependencies)
deltatH_qho_autocorr = executable('deltatH_qho_autocorr', 'src/deltatH_qho_autocorr.cpp', link_with:[MC], include_directories: incdir, dependencies: dependencies)
