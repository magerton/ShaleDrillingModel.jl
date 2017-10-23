export logP!


function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, dograd::Bool, itypidx::Tuple, uv::NTuple{2,<:Real}, d_obs::Integer, s_idx::Integer, z::Real...) where {T}

  # unpack information about current state
  roy, geo = getitype.(isev.itypes, itypidx)
  omroy = one(T) - roy

  # gradient & coef views
  θt =    @view(θfull[[geo, prim.ngeo+1:end-1...]])
  σ = θfull[end]

  # states we can iterate over
  s = state(prim, s_idx)
  Dgt0 = s.D > 0

  # information
  ψ = Dgt0 ? uv[1] : uv[1] + σ*uv[2] # , extrema(_ψspace(prim, σ))...)
  v = uv[2]

  # containers
  dmxp1rng = Base.OneTo(dmax(prim, s_idx)+1)
  ubV = @view(tmp[dmxp1rng])

  # @inbounds
  for di in dmxp1rng
    s_idxp = prim.wp.Sprimes[di,s_idx]
    ubV[di] = prim.f(θt, σ, z..., ψ, di-1, s.d1, Dgt0, omroy) + prim.β * isev.EV[z..., ψ, s_idxp, itypidx...]
  end

  dograd ||  return ubV[d_obs] - logsumexp(ubV)

  logp = ubV[d_obs] - logsumexp_and_softmax!(ubV)
  nSexp1 = _nSexp(prim)+1

  # @inbounds
  for di in dmxp1rng
    wt = di==d_obs ? one(T)-ubV[di] : -ubV[di]
    dim1 = di-1
    s_idxp = prim.wp.Sprimes[di,s_idx]  # TODO: create a proper function mapping sprime_idx to sprime_idx_for_σ given s_idx & wp/prim

    for k in Base.OneTo(length(θt))
      gk = k == 1 ? geo : prim.ngeo + k - 1
      grad[gk] += wt * (prim.dfθ( θt, σ, z..., ψ, k, dim1,   s.d1, Dgt0, omroy) + prim.β * isev.dEV[z..., ψ, k, s_idxp, itypidx...] )
    end

    if !Dgt0
      dpsi = prim.dfψ(θt, σ, z..., ψ, dim1, omroy) + prim.β * isev.dEVψ[z..., ψ, min(s_idxp,nSexp1), itypidx...]
      dsig = prim.dfσ(θt, σ, z..., ψ, dim1, omroy) + prim.β * isev.dEVσ[z..., ψ, min(s_idxp,nSexp1), itypidx...]
      grad[end] += wt * (dpsi*v + dsig)
    end
  end

  return logp
end
