# parallel driver.jl
using Distributed, LinearAlgebra
BLAS.set_num_threads(1)
println("ncores=",nprocs())
if nprocs() == 1
  addprocs(3) 
end

@everywhere begin
  using LinearAlgebra
  BLAS.set_num_threads(1)

  import LinearAlgebra, OpenBLAS32_jll
LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
 # import Pkg
  #Pkg.activate($(Base.current_project()))
end

@everywhere include("pivot-growth.jl")

function do_it_on_ncores(n, ncores, howmany; T=Real )
    println("--------------------")
    println("ncores=", ncores) 
    println("howmany=", howmany)
    tm = @elapsed z = findmax_models(howmany, n=n; T=T)
    
    return z, tm
end

rationalize(x) = convert(BigInt, x)

howmany = 10000

data = do_it_on_ncores(k,nprocs(), howmany, T=Real)   # n, ncores, howmany
#data[1][2]

println()
println("Time in minutes: ",data[2] / 60)
data

A = data[1][2];
Float64(maximum(pivc(genp(A))[2])), Float64(genp(A)[k,k,k])
