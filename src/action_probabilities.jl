export logP!

nplus1_impl(N::Integer) = :(Val{$(N+1)})
@generated nplus1(::Type{Val{N}}) where {N} = nplus1_impl(N)

@inline function logP!(grad::AbstractVector{T}, tmp::Vector{T}, θt::AbstractVector{T}, σ::T, prim::dcdp_primitives{FF,T}, isev::ItpSharedEV, uv::NTuple{2,T}, z::NTuple{NZ,Real}, d_obs::Integer, s_idx::Integer, itypidx::NTuple{NI,Real}, dograd::Bool=true) where {FF,T<:Real,NZ,NI}

  if dograd
    length(grad) == length(θt)+1 || throw(DimensionMismatch())
  end

  # unpack information about current state
  roy, geo = getitype.(isev.itypes, itypidx)

  # states we can iterate over
  s = state(prim.wp, s_idx)
  Dgt0 = s.D > 0

  # information
  ρ = _ρ(σ)
  ψ = Dgt0 ? _ψ2(uv...,ρ) : _ψ1clamp(uv..., ρ, prim)

  # containers
  drng = action_iter(prim.wp, s_idx)
  ubV = view(tmp, drng+1)

  @inbounds for d in drng
    ubV[d+1] = flow(FF, θt, σ, z..., ψ, d, s.d1, Dgt0, roy, geo) + prim.β * isev.EV[z..., ψ, _sprime(prim.wp, s_idx, d), itypidx...]
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
      grad[k] += wt * (flowdθ(FF, θt, σ, z..., ψ, k, d, s.d1, Dgt0, roy, geo) + prim.β * isev.dEV[z..., ψ, k, sp_idx, itypidx...] )
    end

    if !Dgt0
      dpsi = flowdψ(FF, θt, σ, z..., ψ, d, roy, geo) + prim.β * gradient_d(nplus1(Val{NZ}), isev.EV, z..., ψ, sp_idx, itypidx...)::T
      dsig = flowdσ(FF, θt, σ, z..., ψ, d, roy, geo) + prim.β * isev.dEVσ[z..., ψ, sp_idx, itypidx...]
      grad[end] += wt * (dpsi*_dψ1dθρ(uv..., ρ, σ) + dsig)
    end
  end

  return logp
end
