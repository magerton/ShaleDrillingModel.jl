export logP!

nplus1_impl(N::Integer) = :(Val{$(N+1)})
@generated nplus1(::Type{Val{N}}) where {N} = nplus1_impl(N)

@inline function logP!(grad::AbstractVector{T}, tmp::Vector{T}, θt::AbstractVector{T}, σ::T, prim::dcdp_primitives{FF,T}, isev::ItpSharedEV, uv::NTuple{2,T}, z::NTuple{NZ,Real}, d_obs::Integer, s_idx::Integer, itypidx::NTuple{NI,Real}, dograd::Bool=true) where {FF,T<:Real,NZ,NI}

  if dograd
    length(grad) == length(θt)+1 || throw(DimensionMismatch())
  end

  # unpack information about current state
  itype = getitype.(isev.itypes, itypidx)
  wp = prim.wp
  drng = actionspace(wp, s_idx)

  # information
  # NOTE: truncation of ψ1 can lead to errors in gradient!!!!!!
  ρ = _ρ(σ)
  ψ = s_idx <= end_ex0(wp) ? _ψ1clamp(uv..., ρ, prim) : _ψ2(uv...,ρ)

  # we'll reuse this
  ubV = view(tmp, drng.+1)

  @inbounds for d in drng
    sp_idx = sprime(wp, s_idx, d)
    ubV[d+1] = flow(FF, wp, s_idx, θt, σ, z, ψ, d, itype...) + prim.β * isev.EV[z..., ψ, sp_idx, itypidx...]
  end

  dograd ||  return ubV[d_obs+1] - logsumexp(ubV)

  # do this in two steps so that we don't accientally overwrite ubV[d_obs] with Pr(d_obs)
  logp = ubV[d_obs+1]
  logp -= logsumexp_and_softmax!(ubV)

  @inbounds for d in drng
    wt = d==d_obs ? one(T) - ubV[d+1] : -ubV[d+1]
    sp_idx = sprime(wp, s_idx, d)

    @inbounds for k in eachindex(θt) # NOTE: assumes 1-based linear indexing!!
      grad[k] += wt * (flowdθ(FF, wp, s_idx, θt, σ, z, ψ, k, d, itype...) + prim.β * isev.dEV[z..., ψ, k, sp_idx, itypidx...] )
    end

    if s_idx <= end_ex0(wp)
      dpsi = flowdψ(FF, wp, s_idx, θt, σ, z, ψ, d, itype...) + prim.β * gradient_d(nplus1(Val{NZ}), isev.EV, z..., ψ, sp_idx, itypidx...)::T
      dsig = flowdσ(FF, wp, s_idx, θt, σ, z, ψ, d, itype...) + prim.β * isev.dEVσ[z..., ψ, sp_idx, itypidx...]
      grad[end] += wt * (dpsi*_dψ1dθρ(uv..., ρ, σ) + dsig)
    end
  end

  return logp
end
