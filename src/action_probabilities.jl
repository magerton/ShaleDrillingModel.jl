export logP!


function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, dograd::Bool, itypidx::Tuple, uv::NTuple{2,<:Real}, dp1::Integer, s_idx::Integer, z::Real...) where {T}

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
    # ubV[di] = prim.f(θt, σ, z..., ψ, di-1, s.d1, Dgt0, omroy) + prim.β * isev.EV[z..., ψ, s_idxp, itypidx...]
    ubV[di] = isev.EV[z..., ψ, s_idxp, itypidx...]
  end

  dograd ||  return ubV[dp1] # - logsumexp(ubV)

  logp = ubV[dp1] # - logsumexp_and_softmax!(ubV)
  nSexp1 = _nSexp(prim)+1
  gradvw = @view(grad[[geo, prim.ngeo+1:end...]])

  # TODO: create a proper function mapping sprime_idx to sprime_idx_for_σ given s_idx & wp/prim

  # @inbounds
  for di in dmxp1rng
    wt = di==dp1 ? one(T) : zero(T) # -ubV[di] : -ubV[di]
    dim1 = di-1
    s_idxp = prim.wp.Sprimes[di,s_idx]
    for k in Base.OneTo(length(θt))
      # gk = k == 1 ? geo : prim.ngeo + k - 1
      # gradvw[k]        += wt * (0.0*prim.df( θt, σ, z..., ψ, k, dim1,   s.d1, Dgt0, omroy) + prim.β * isev.dEV[ z..., ψ, k, s_idxp, itypidx...] )
      gradvw[k] += wt * isev.dEV[ z..., ψ, k, s_idxp, itypidx...]
    end
    # Dgt0 || (grad[end] += wt * (0.0*prim.dfσ(θt, σ, z..., ψ, v, dim1,               omroy) + prim.β * isev.dEVσ[z..., ψ, v, min(s_idxp,nSexp1), itypidx...] ))
    Dgt0 || (gradvw[end] += wt * isev.dEVσ[z..., ψ, v, min(s_idxp,nSexp1), itypidx...])
  end

  return logp
end
