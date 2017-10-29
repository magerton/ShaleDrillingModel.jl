export logP!

# θt::AbstractVector{Float64},
function logP!(grad::AbstractVector{T}, tmp::Vector{T}, θt::Vector{T}, θfull::AbstractVector{T}, prim::dcdp_primitives{T}, isev::ItpSharedEV, uv::NTuple{2,T}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T<:Real}

  if dograd
    length(grad) == _nθt(prim) + _ngeo(prim) || throw(DimensionMismatch())
  end

  # unpack information about current state
  roy, geo = getitype.(isev.itypes, itypidx)


  # gradient & coef views
  # TODO: room for performance improvement if don't have to copy this over?
  _θt!(θt, θfull, prim, geo)
  lenθfull = length(θfull)
  σ = θfull[lenθfull]

  # states we can iterate over
  s = state(prim.wp, s_idx)
  Dgt0 = s.D > 0

  # information
  ψ = Dgt0 ? uv[1] : uv[1] + σ*uv[2] # , extrema(_ψspace(prim, σ))...)
  v = uv[2]

  # containers
  drng = action_iter(prim.wp, s_idx)
  ubV = view(tmp, drng+1)

  @inbounds for (di,d) in enumerate(drng)
    ubV[di] = prim.f(θt, σ, z..., ψ, d, s.d1, Dgt0, roy, geo)::T + prim.β * isev.EV[z..., ψ, _sprime(prim.wp, s_idx, d), itypidx...]
  end

  dograd ||  return ubV[d_obs] - logsumexp(ubV)

  # do this in two steps so that we don't accientally overwrite ubV[d_obs] with Pr(d_obs)
  logp = ubV[d_obs]
  logp -= logsumexp_and_softmax!(ubV)

  nSexp1 = _nSexp(prim)+1

  @inbounds for (di,d) in enumerate(drng)
    wt = di==d_obs ? one(T) - ubV[di] : -ubV[di]

    for k in Base.OneTo(_nθt(prim))
      gk = k == 1 ? geo : _ngeo(prim) + k - 1
      grad[gk] += wt * (prim.dfθ( θt, σ, z..., ψ, k, d, s.d1, Dgt0, roy, geo)::T + prim.β * isev.dEV[z..., ψ, k, _sprime(prim.wp, s_idx, d), itypidx...] )
    end

    if !Dgt0
      dpsi = prim.dfψ(θt, σ, z..., ψ, d, roy, geo)::T + prim.β * gradient_d(length(z)+1, isev.EV, z..., ψ, _sprime(prim.wp, s_idx, d), itypidx...)
      dsig = prim.dfσ(θt, σ, z..., ψ, d, roy, geo)::T + prim.β * isev.dEVσ[z..., ψ, _sprime(prim.wp, s_idx, d), itypidx...]
      grad[lenθfull] += wt * (dpsi*v + dsig)
    end
  end

  return logp
end
