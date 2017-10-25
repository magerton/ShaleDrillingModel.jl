







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
            uv = (0.6, 0.6)

            # FIXME: I CHANGED THE FUNCTION ARGUMENTS
            # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
            lp = logP!(grad, tmp, θfull, prim, isev, true, itypidx, uv, d+1, s_idx, z...)

            for k in 1:length(θfull)
                θ1 .= θfull
                θ2 .= θfull
                h = max( abs(θfull[k]), one(T) ) * cbrt(eps(T))
                θ1[k] -= h
                θ2[k] += h
                serial_solve_vf_all!(shev, tmpv, prim, θ1, Val{true})
                # FIXME: I CHANGED THE FUNCTION ARGUMENTS
                # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
                lp1 = logP!(Vector{T}(0), tmp, θ1, prim, isev, false, itypidx, uv, d+1, s_idx, z...)
                serial_solve_vf_all!(shev, tmpv, prim, θ2, Val{true})
                # FIXME: I CHANGED THE FUNCTION ARGUMENTS
                # function logP!(grad::AbstractVector{T}, tmp::AbstractVector, θfull::AbstractVector, prim::dcdp_primitives, isev::ItpSharedEV, uv::NTuple{2,<:Real}, z::Tuple, d_obs::Integer, s_idx::Integer, itypidx::Tuple, dograd::Bool=true) where {T}
                lp2 = logP!(Vector{T}(0), tmp, θ2, prim, isev, false, itypidx, uv, d+1, s_idx, z...)
                fdgrad[k] = (lp2-lp1)/(θ2[k] - θ1[k])
            end
            absd, abspos = findmax(abs.(grad .- fdgrad))
            isapprox(grad, fdgrad, atol=1e-5) || warn("bad gradient. d=$d, sidx=$s_idx, offender at θ[$abspos], absdiff = $absd") #. absdiff at grad[$absdi] = $absd ")
        end
    end
    @show "done"
end


using Plots
gr()
plot(ψspace, shev.EV[16,:,1,4,1])

ψrng = 19:1:28
plot(vspace, shev.dEVσ[16,ψrng,:,1,4,1]', labels=string.(ψspace[ψrng]), ylabel="dEV/dsigma", xlabel="v")

plot(shev.dEVσ[16,:,:,1,4,1]' ./ vspace)




shev.dEVσ[16,51,:,1,4,1]
length(vspace)

ψspace[51]


size(shev.dEVσ)


ψspace
