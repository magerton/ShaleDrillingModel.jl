export logP!

function logP!(grad::AbstractVector{T}, tmp::Vector{T}, θt::AbstractVector{T}, σ::T, prim::dcdp_primitives{T}, isev::ItpSharedEV, uv::NTuple{2,T}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T<:Real}

  if dograd
    length(grad) == length(θt)+1 || throw(DimensionMismatch())
  end

  # unpack information about current state
  roy, geo = getitype.(isev.itypes, itypidx)

  # states we can iterate over
  s = state(prim.wp, s_idx)
  Dgt0 = s.D > 0

  # information
  ψ = Dgt0 ? uv[1] : uv[1] + σ*uv[2] # , extrema(_ψspace(prim, σ))...)
  v = uv[2]

  # containers
  drng = action_iter(prim.wp, s_idx)
  ubV = view(tmp, drng+1)

  @inbounds for d in drng
    ubV[d+1] = prim.f(θt, σ, z..., ψ, d, s.d1, Dgt0, roy, geo)::T + prim.β * isev.EV[z..., ψ, _sprime(prim.wp, s_idx, d), itypidx...]
  end

  dograd ||  return ubV[d_obs+1] - logsumexp(ubV)

  # do this in two steps so that we don't accientally overwrite ubV[d_obs] with Pr(d_obs)
  logp = ubV[d_obs+1]
  logp -= logsumexp_and_softmax!(ubV)

  nSexp1 = _nSexp(prim)+1

  @inbounds for d in drng
    wt = d==d_obs ? one(T) - ubV[d+1] : -ubV[d+1]
    sp_idx = _sprime(prim.wp, s_idx, d)

    for k in eachindex(θt) # NOTE: assumes 1-based linear indexing!!
      grad[k] += wt * (prim.dfθ( θt, σ, z..., ψ, k, d, s.d1, Dgt0, roy, geo)::T + prim.β * isev.dEV[z..., ψ, k, sp_idx, itypidx...] )
    end

    if !Dgt0
      dpsi = prim.dfψ(θt, σ, z..., ψ, d, roy, geo)::T + prim.β * gradient_d(length(z)+1, isev.EV, z..., ψ, sp_idx, itypidx...)
      dsig = prim.dfσ(θt, σ, z..., ψ, d, roy, geo)::T + prim.β * isev.dEVσ[z..., ψ, sp_idx, itypidx...]
      grad[end] += wt * (dpsi*v + dsig)
    end
  end

  return logp
end
