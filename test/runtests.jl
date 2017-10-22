using ShaleDrillingModel
using Base.Test
using StatsFuns
using JLD

jldpath = joinpath(Pkg.dir("ShaleDrillingModel"), "data/price-cost-transitions.jld")
@load jldpath pspace cspace Πpcr Πp1

# some primitives
β = (1.02 / 1.125) ^ (1./12.)  # real discount rate
royalty_rates = [0.125, 1./6., 0.1875, 0.20, 0.225, 0.25]
geology_types = 1:10
royalty_types = 1:length(royalty_rates)

# initial parameters
θt = [1.0, -2.6, -2.4, 1.2]
σv = 1.0

# problem sizes
nψ, dmx, nz, nv =  20, 2, size(Πp1,1), 10
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-6.0, 6.0, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
# nd, ns, nθ = length(dspace), length(wp), length(θt)

prim = dcdp_primitives(u_add, du_add, duσ_add, β, wp, zspace, Πp1, ψspace, vspace, 1)
tmpv = dcdp_tmpvars(length(θt), prim)
evs = dcdp_Emax(θt, prim)

# check sizes of models
ShaleDrillingModel.check_size(θt, prim, evs)

if false
    include("learning_transition.jl")
    include("utility.jl")
    include("logsumexp3.jl")
    include("vf_solve_terminal_and_infill.jl")
    include("vf_solve_exploratory.jl")

    for r in royalty_rates
        println("test royalty rate $r")
        check_EVjac(evs, tmpv, prim, θt, σv, r)
    end

    # check jacobian
    include("parallel_solution.jl")
end

shev = SharedEV([1,], vcat(θt, σv), prim, royalty_rates, 1:1)
isev = ItpSharedEV(shev, prim, σv)
set_g_dcdp_primitives(prim)
set_g_dcdp_tmpvars(tmpv)
set_g_SharedEV(shev)

# s = parallel_solve_vf_all!(shev, vcat(θt,σv), Val{true})
# fetch.(s)



# include("parallel_dEV_check.jl")

let itypidx = (4,1),
    tmp = Vector{Float64}(dmax(wp)+1),
    θfull = vcat(θt,σv),
    grad = similar(θfull),
    fdgrad = similar(θfull),
    θ1 = similar(θfull),
    θ2 = similar(θfull),
    T = eltype(θfull)

    serial_solve_vf_all!(shev, tmpv, prim, θfull, Val{true})

    for s_idx in 1:12 # length(wp)
        for d in action_iter(wp, s_idx)
            grad .= 0.0
            fdgrad .= 0.0

            z = rand.(zspace)
            uv = (0.1, 0.1)

            lp = logP!(grad, tmp, θfull, prim, isev, true, itypidx, uv, d+1, s_idx, z...)

            for k in 1:length(θfull)
                θ1 .= θfull
                θ2 .= θfull
                h = max( abs(θfull[k]), one(T) ) * cbrt(eps(T))
                θ1[k] -= h
                θ2[k] += h
                serial_solve_vf_all!(shev, tmpv, prim, θ1, Val{true})
                lp1 = logP!(Vector{T}(0), tmp, θ1, prim, isev, false, itypidx, uv, d+1, s_idx, z...)
                serial_solve_vf_all!(shev, tmpv, prim, θ2, Val{true})
                lp2 = logP!(Vector{T}(0), tmp, θ2, prim, isev, false, itypidx, uv, d+1, s_idx, z...)
                fdgrad[k] = (lp2-lp1)/(θ2[k] - θ1[k])
            end
            absd, abspos = findmax(abs.(grad .- fdgrad))
            isapprox(grad, fdgrad, atol=1e-5) || warn("bad gradient. d=$d, sidx=$s_idx, offender at θ[$abspos], absdiff = $absd") #. absdiff at grad[$absdi] = $absd ")
        end
    end
    @show "done"
end




























# include("action_probabilities.jl")


#
