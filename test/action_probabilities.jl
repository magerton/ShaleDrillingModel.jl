



ishev = ItpSharedEV(shev, prim, σv)
θfull = vcat(θt,σv)
T = eltype(θfull)
tmp = Vector{T}(dmax(wp)+1)
grad = zeros(T, length.((θfull, tmp, wp, shev.itypes...)))
fdgrad = zeros(T, size(grad))
CR = CartesianRange(length.(shev.itypes))
θ1 = similar(θfull)
θ2 = similar(θfull)
zs = [rand.(zspace) for CI in CR]
uvs = [(rand(vspace./2), rand(vspace./2)) for CI in CR]

zs = [(1.5,) for CI in CR]
uvs = [(0.0, 1.5,) for CI in CR]

println("Testing logP & gradient")

grad .= zero(T)
fdgrad .= zero(T)

dograd = true
for i in eachindex(shev.dEVσ)
    shev.dEVσ[i] = 0.0
end
parallel_solve_vf_all!(shev, θfull, Val{dograd})
for CI in CR
    uv,z = uvs[CI], zs[CI]
    for s_idx in 1:length(wp)
        for d in ShaleDrillingModel.action_iter(wp, s_idx)
            # FIXME: I CHANGED THE FUNCTION ARGUMENTS
            # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
            @views lp = logP!(grad[:,d+1, s_idx, CI], tmp, θfull, prim, ishev, dograd, CI.I, uv, d+1, s_idx, z...)
        end
    end
end

dograd = false
fdgrad .= zero(T)
for k in 1:length(θfull)
    θ1 .= θfull
    θ2 .= θfull
    h = max( abs(θfull[k]), one(T) ) * cbrt(eps(T))
    θ1[k] -= h
    θ2[k] += h
    hh = θ2[k] - θ1[k]

    parallel_solve_vf_all!(shev, θ1, Val{dograd})
    for CI in CR
        uv,z = uvs[CI], zs[CI]
        for s_idx in 1:length(wp)
            for d in ShaleDrillingModel.action_iter(wp, s_idx)
                # FIXME: I CHANGED THE FUNCTION ARGUMENTS
                # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
                @views ll = logP!(grad[:,d+1, s_idx, CI], tmp, θ1, prim, ishev, dograd, CI.I, uv, d+1, s_idx, z...)
                fdgrad[k,d+1, s_idx, CI] -= ll
            end
        end
    end

    parallel_solve_vf_all!(shev, θ2, Val{dograd})
    for CI in CR
        uv,z = uvs[CI], zs[CI]
        for s_idx in 1:length(wp)
            for d in ShaleDrillingModel.action_iter(wp, s_idx)
                # FIXME: I CHANGED THE FUNCTION ARGUMENTS
                # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
                @views ll = logP!(grad[:,d+1, s_idx, CI], tmp, θ2, prim, ishev, dograd, CI.I, uv, d+1, s_idx, z...)
                fdgrad[k,d+1,s_idx,CI] += ll
                fdgrad[k,d+1,s_idx,CI] /= hh
            end
        end
    end
end

@show badval, badidx =  findmax(abs.(grad.-fdgrad))

@show ind2sub(size(grad), badidx)


# grad dims: (θfull, dmaxp1, wp, shev.itypes...)))
(grad .- fdgrad)[5,:,7:11,:,1]

@test isapprox(grad, fdgrad, atol=1e-7)

#     println("logP gradient looks good! :)")
# end
