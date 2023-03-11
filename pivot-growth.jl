using JuMP
import Ipopt
# using FoldsPreview
using Dates
using FLoops
using LinearAlgebra

#
# KEY ALGORITHMIC SUBROUTINES:
#


# converts a matrix into completely pivoted (CP) form
function geperm(a)
    n = size(a, 1)
    b = copy(a)
    for k = n:-1:2
        i, j = findmax(abs.(a))[2].I
        a[[i 1],:] = a[[1 i],:]
        a[:,[1 j]] = a[:,[j 1]]
        b[[n - k + i n - k + 1],:] = b[[n - k + 1 n - k + i],:]
        b[:,[n - k + 1 n - k + j]] = b[:,[n - k + j n - k + 1]]
        a = a[2:k,2:k] .- (a[2:k,1:1] ./ a[1,1]) * a[1:1,2:k]
    end
    return(b)
end


# performs Gaussian elimination without pivoting
function genp(a)
    n = size(a, 1)
    thestart = zeros( eltype(a), n, n, n)
    thestart[:,:,1] = a
    for k = n:-1:2
        a = a[2:k,2:k] .- (a[2:k,1:1] ./ a[1,1]) * a[1:1,2:k]
        thestart[ (n - k + 2):n,  (n - k + 2):n, n - k + 2 ] .= a
    end
    thestart
end


# converts an almost CP matrix into a CP matrix (without pivoting)
function convert_to_cp(a)
    n = size(a,1)
    b = genp(a)
    for k = (n-1):-1:1
        del = max((maximum(abs.(b[k,k+1:n,k]))/abs(b[k,k,k]))^2, (maximum(abs.(b[k+1:n,k,k]))/abs(b[k,k,k]))^2, maximum(abs.(b[k+1:n,k+1:n,k]))/abs(b[k,k,k]) )
        if del > 1
            delsqrt = Rational{BigInt}(nextfloat(del^(1/2)))
            if k == 1
                b[k,k:n,k] *= delsqrt
                b[k:n,k,k] *= delsqrt
            else
                for i = 1:(k-1)
                    b[k,k,i] += (delsqrt^2-1)*b[k,k,k]
                    b[k,k+1:n,i] += (delsqrt-1)*b[k,k+1:n,k]
                    b[k+1:n,k,i] += (delsqrt-1)*b[k+1:n,k,k]
                end
            end
        end
    end
    b[:,:,1]
end


# converts an almost RP matrix into a RP matrix (without pivoting)
function gerp(a)
    n = size(a,1)
    b = genp(a)
    for k = (n-1):-1:1
        del = max( (maximum(abs.(b[k,k+1:n,k]))/abs(b[k,k,k]))^2 , (maximum(abs.(b[k+1:n,k,k]))/abs(b[k,k,k]))^2 )
        if del > 1
            delsqrt = Rational{BigInt}(nextfloat(del^(1/2)))
            if k == 1
                b[k,k:n,k] *= delsqrt
                b[k:n,k,k] *= delsqrt
            else
                for i = 1:(k-1)
                    b[k,k,i] += (delsqrt^2-1)*b[k,k,k]
                    b[k,k+1:n,i] += (delsqrt-1)*b[k,k+1:n,k]
                    b[k+1:n,k,i] += (delsqrt-1)*b[k+1:n,k,k]
                end
            end
        end
    end
    b[:,:,1]
end


# outputs the pivots of a matrix and how close it is to being CP
function pivc(a)
    n = size(a,1)
    p = zeros(eltype(a[1,1,1]),n)
    t = copy(p)
    for i in 1:n
        p[i] = abs(a[i,i,i])
        t[i] = maximum(abs.(a[i:n,i:n,i]))/abs(a[i,i,i])
    end
    p,t
end


# outputs the pivots of a matrix and how close it is to being RP
function pivr(a)
    n = size(a,1)
    p = zeros(eltype(a[1,1,1]),n)
    t = copy(p)
    for i in 1:n
        p[i] = abs(a[i,i,i])
        t[i] = max(maximum(abs.(a[i:n,i,i]))/abs(a[i,i,i]),maximum(abs.(a[i,i:n,i]))/abs(a[i,i,i]))
    end
    p,t
end

# maximize pivot growth for real matrix with complete pivoting
function run_model(n)
    model = Model(Ipopt.Optimizer)
    indices = [ (i, j, k) for k = 1:n for i = k:n for j = k:n]

    startmatrix = randn(n, n)
    thestart = genp(geperm(startmatrix))
    thestart = genp(thestart[:,:,1] * Diagonal(sign.(thestart[k,k,k] for k = 1:n)))
    thestart ./= thestart[1,1,1]

    @variable(model, x[i=indices], start = thestart[i[1],i[2],i[3]]  ) # random starts
    for k in 1:(n - 1), i in (k + 1):n, j in (k + 1:n)
        @NLconstraint(model,
        x[(i, j, k + 1)] - x[(i, j, k)]  +  x[(i, k, k)] * x[(k, j, k)] / x[(k, k, k)] == 0 )
    end
    for k = 1:n
        @constraint(model, x[(k, k, k)] ≥ 0)
    end
    @constraint(model,   x[(1, 1, 1)] == 1)
    for i in 1:n, j in 1:n
        @constraint(model, -1 ≤ x[(i, j, 1)] ≤ 1)
    end
    for k in 2:n - 1, i in k:n, j in k:n
        @constraint(model,  x[(i, j, k)] ≤ x[(k, k, k)])
        @constraint(model, -x[(k, k, k)] ≤ x[(i, j, k)] )
    end
    w = n
    @objective(model, Max, x[(w, w, w)])
    set_optimizer_attribute(model, "max_iter", 500)
    set_silent(model)
    optimize!(model)
    A = reshape((value.(x)).data[1:n^2], n, n)
    B = convert_to_cp(Rational{BigInt}.(A))
    B = B/B[1,1]
    val = genp(B)[n,n,n]
    # return the optimization and the argmax
    val,B,MOI.FEASIBLE_POINT
end

"""
    run_model_new(n)

Maximize pivot growth for real matrix with complete pivoting.
"""
function run_model_new(n)
    start_matrix = randn(n, n)
    thestart = genp(geperm(start_matrix))
    thestart = genp(thestart[:,:,1] * Diagonal(sign.(thestart[k,k,k] for k = 1:n)))
    thestart ./= thestart[1,1,1]
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 500)
    set_silent(model)
    indices = [(i, j, k) for k in 1:n for i in k:n for j in k:n]
    @variable(model, x[(i,j,k) in indices], start = thestart[i,j,k])
    for k in 1:n
        set_lower_bound(x[(k,k,k)], 0)
    end
    for i in 1:n, j in 1:n
        set_lower_bound(x[(i,j,1)], -1)
        set_upper_bound(x[(i,j,1)], 1)
    end
    fix(x[(1, 1, 1)], 1; force = true)
    @constraint(
        model,
        [k=1:(n-1), i=(k+1):n, j=(k+1:n)],
        x[(k,k,k)] * (x[(i,j,k+1)] - x[(i,j,k)]) + x[(i,k,k)] * x[(k,j,k)] == 0,
    )
    for k in 2:n - 1, i in k:n, j in k:n
        @constraint(model, x[(i,j,k)] <= x[(k,k,k)])
        @constraint(model, -x[(k,k,k)] <= x[(i,j,k)])
    end
    @objective(model, Max, x[(n,n,n)])
    optimize!(model)
    A = reshape(Array(value.(x))[1:n^2], n, n)
    B = convert_to_cp(Rational{BigInt}.(A))
    B = B / B[1, 1]
    val = genp(B)[n, n, n]
    # return the optimization and the argmax
    return val, B, primal_status(model)
end

# maximize pivot growth for real matrix with rook pivoting
function run_rook_model(n)
    model = Model(Ipopt.Optimizer)
    indices = [ (i, j, k) for k = 1:2 for i = 1:n for j = 1:n]
    # Create a start matrix
    startmatrix = randn(n, n)
    startGE = genp(geperm(startmatrix))
    startGE = genp(startGE[:,:,1] * Diagonal(sign.(startGE[k,k,k] for k = 1:n)))
    startGE ./= startGE[1,1,1]
    thestart = zeros(n,n,2)
    thestart[:,:,1] = startGE[:,:,1]
    for k = 1:n
        thestart[k:n,k,2]=startGE[k:n,k,k]
        thestart[k,k:n,2]=startGE[k,k:n,k]
    end
    @variable(model, x[i=indices], start = thestart[i[1],i[2],i[3]]  ) # random starts
    for k = 1:n
        @constraint(model, x[(k, k, 2)] ≥ 0)
    end
    for i in 1:n, j in 1:n
        @constraint(model, -1 ≤ x[(i, j, 1)] ≤ 1)
    end
    for k in 1:n - 1, i in k:n
        @constraint(model,  x[(i, k, 2)] ≤ x[(k, k, 2)])
        @constraint(model, -x[(k, k, 2)] ≤ x[(i, k, 2)] )
    end
    for k in 1:n - 1, j in k:n
        @constraint(model,  x[(k, j, 2)] ≤ x[(k, k, 2)])
        @constraint(model, -x[(k, k, 2)] ≤ x[(k, j, 2)] )
    end

    for i in 1:n
        @constraint(model, x[(i, 1, 1)] - x[(i, 1, 2)] == 0)
    end
    for j in 2:n
        @constraint(model, x[(1, j, 1)] - x[(1, j, 2)] == 0)
    end

    for i in 2:n, j in i:n
        @NLconstraint(model,
        x[(i, j, 2)] - x[(i, j, 1)]  +  sum(x[(i, l, 2)] * x[(l, j, 2)] / x[(l, l, 2)] for l in 1:(i-1)) == 0 )
    end

    for j in 2:(n-1), i in (j+1):n
        @NLconstraint(model,
        x[(i, j, 2)] - x[(i, j, 1)]  +  sum(x[(i, l, 2)] * x[(l, j, 2)] / x[(l, l, 2)] for l in 1:(j-1)) == 0 )
    end
    w = n
    @objective(model, Max, x[(w, w, 2)])
    set_optimizer_attribute(model, "max_iter", 500)
    set_silent(model)
    optimize!(model)
    # return the optimization and the argmax
    objective_value(model), reshape((value.(x)).data[1:n^2], n, n), primal_status(model)
end

"""
    run_rook_model_new(n)

Maximize pivot growth for real matrix with rook pivoting.
"""
function run_rook_model_new(n)
    start_matrix = randn(n, n)
    startGE = genp(geperm(start_matrix))
    startGE = genp(startGE[:,:,1] * Diagonal(sign.(startGE[k,k,k] for k in 1:n)))
    startGE ./= startGE[1,1,1]
    the_start = zeros(n,n,2)
    the_start[:,:,1] = startGE[:,:,1]
    for k in 1:n
        the_start[k:n,k,2] = startGE[k:n,k,k]
        the_start[k,k:n,2] = startGE[k,k:n,k]
    end
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 500)
    set_silent(model)
    indices = [(i, j, k) for k in 1:2 for i in 1:n for j in 1:n]
    @variable(model, x[(i,j,k) in indices], start = the_start[i,j,k])
    for k in 1:n
        set_lower_bound(x[(k,k,2)], 0)
    end
    for i in 1:n, j in 1:n
        set_lower_bound(x[(i,j,1)], -1)
        set_upper_bound(x[(i,j,1)], 1)
    end
    for k in 1:n - 1, i in k:n
        @constraint(model, x[(i,k,2)] <= x[(k,k,2)])
        @constraint(model, -x[(k,k,2)] <= x[(i,k,2)])
    end
    for k in 1:n - 1, j in k:n
        @constraint(model, x[(k,j,2)] <= x[(k,k,2)])
        @constraint(model, -x[(k,k,2)] <= x[(k,j,2)])
    end
    for i in 1:n
        @constraint(model, x[(i,1,1)] == x[(i,1,2)])
    end
    for j in 2:n
        @constraint(model, x[(1,j,1)] == x[(1,j,2)])
    end
    for i in 2:n, j in i:n
        @NLconstraint(
            model,
            x[(i,j,2)] - x[(i,j,1)] + sum(x[(i,l,2)] * x[(l,j,2)] / x[(l,l,2)] for l in 1:(i-1)) == 0
        )
    end
    for j in 2:(n-1), i in (j+1):n
        @NLconstraint(
            model,
            x[(i,j,2)] - x[(i,j,1)] + sum(x[(i,l,2)] * x[(l,j,2)] / x[(l,l,2)] for l in 1:(j-1)) == 0,
        )
    end
    @objective(model, Max, x[(n,n,2)])
    optimize!(model)
    # return the optimization and the argmax
    B = reshape(Array(value.(x))[1:n^2], n, n)
    return objective_value(model), B, primal_status(model)
end


#
# MAIN FUNCTION:
#
# runs multiple iterations of an optimization algorithm, outputs best result


function findmax_models(
    howmany;  # total number of models to be generated (across all workers)
    n,
    ncores=typemax(Int),  # (maximum) number of core to be used
    th=IntervalThrottle(5), # preview for each 10 seconds by default
    T=T
)
    prev_maxobj = Ref{Any}(nothing)
    function on_preview((acc, (nsols, nfeasibles)))

        @info "nfeasibles/nsols = $nfeasibles/$nsols"
        acc isa Tuple || return
        (maxobj, best) = acc
        if prev_maxobj[] === nothing
            flag1 = true
        else
            flag1 = maxobj - prev_maxobj[] > .0001  # or just `> 0`
        end
        prev_maxobj[] = maxobj
        if flag1
            show(stdout, "text/plain", best)
            println()
            @info "Preview" now() maxobj
            show(stdout, "text/plain", best)
        else
            @info "No Change"   # now()
        end
    end
# function on_preview(((maxobj, best),))
#     @info "Preview" now() maxobj best
# end
    ex = DistributedEx(basesize=cld(howmany, min(nworkers(), ncores)))
    with_preview(on_preview, ex, th) do previewer
        maxobj = best = nothing
        @floop ex for run in previewer(1:howmany)
           # println("run=",run)

            obj, val, status  = T==Real ? run_model(n) : run_complex_model(n)
           # if status ==  MOI.FEASIBLE_POINT
                @reduce() do (maxobj; obj), (best; val)
                    if maxobj < obj
                        maxobj = obj
                        best = val
                    end
                end
           # end
            @reduce(
                nsols += 1,
                nfeasibles += (status == MOI.FEASIBLE_POINT),
            )
        end
        return (; maxobj, best, nsols, nfeasibles)
    end
end

import Random
import Statistics

function conf_interval(x)
    μ = Statistics.mean(x)
    err = 1.96 * Statistics.std(x) / sqrt(length(x))
    return "$μ ± $err"
end

function benchmark_model(; n, replications)
    @info "Benchmarking run_model"
    run_model(n)
    run_model_new(n)
    model_times, model_new_times = Float64[], Float64[]
    for i in 1:replications
        # Random.seed!(1234 * i)
        # push!(model_times, @elapsed run_model(n))
        Random.seed!(1234 * i)
        push!(model_new_times, @elapsed run_model_new(n))
    end
    println("model     = ", conf_interval(model_times))
    println("model_new = ", conf_interval(model_new_times))
    return
end

function benchmark_rook_model(; n, replications)
    @info "Benchmarking run_rook_model"
    run_rook_model(n)
    run_rook_model_new(n)
    model_times, model_new_times = Float64[], Float64[]
    for i in 1:replications
        Random.seed!(1234 * i)
        push!(model_times, @elapsed run_rook_model(n))
        Random.seed!(1234 * i)
        push!(model_new_times, @elapsed run_rook_model_new(n))
    end
    println("model     = ", conf_interval(model_times))
    println("model_new = ", conf_interval(model_new_times))
    return
end


benchmark_model(n = 10, replications = 10)
# benchmark_rook_model(n = 10, replications = 10)
