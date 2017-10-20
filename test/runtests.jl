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
σv = 0.5

# problem sizes
nψ, dmx, nz, nv =  5, 2, size(Πp1,1), 5
wp = well_problem(dmx,4,10)
zspace, ψspace, dspace, d1space, vspace = (pspace,), linspace(-5.5, 5.5, nψ), 0:dmx, 0:1, linspace(-3.0, 3.0, nv)
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

    # check jacobian
    check_EVjac(evs, tmpv, prim, θt, σv, 0.2)
    include("parallel_solution.jl")
end

sev = SharedEV([1,], vcat(θt, σv), prim, royalty_rates, 1:1)

(z..., s_idx, royalty, geoid)
(u, v,)

function logP!(grad::AbstractVector, ubVfull::AbstractVector, dubVfull::AbstractMatrix, θtmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::InterpolatedSharedEV, uv::NTuple{2,<:Real}, d::Integer, st::Real...)

  # pull out information about the current state
  s_idx, roy, geoid = st[end-2:end]
  z = st[1:end-3]
  ψ = uv[1] + σ * uv[2]
  omroy = 1.0 - roy

  # states we can iterate over
  sprimes = sprime_idx(prim, s_idx)
  dmaxp1 = length(sprimes)
  s = state(prim, i)
  Dgt0 = s.D>0

  # coefs
  σ = _θt!(θtmp, θfull, geoid, prim.ngeo)

  # container
  ubv = @view(ubVfull[1:dmaxp1])

  for di, si in enumerate(sprimes)
    ubv[di] = prim.f(θtmp, σ, z..., ψ, d-1, s.d1, Dgt0, omroy) + prim.β * isev.EV[z..., ψ, si, roy, geoid]
  end

  if !dograd
    return ubv[d+1] - logsumexp(ubv)
  else
    logp = ubv[d+1] - logsumexp_and_softmax!(ubv)

    for di, si in enumerate(sprimes)
      for k in 1:length(θt)
        dubV[d,k] = prim.df(θtmp, σ, z..., ψ, k, d-1, Dgt0, omroy) + prim.β * isev.dEV[z..., ψ, k, si, roy, geoid]
      end
    end

    for j in 1:size(dubV,2)
      grad[geoid] +=

      for i in 1:size(dubV,1)

      end
    end



end













#
