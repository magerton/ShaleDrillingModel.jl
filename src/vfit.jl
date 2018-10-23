# --------------------------- basic VFIT ----------------------------

# simple Vfit
function vfit!(EV0::AbstractMatrix, ubV::AbstractArray, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix)
    logsumexp3!(lse,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
end

# preserves ubV & updates derivatives
function vfit!(EV0::AbstractMatrix, dEV0::AbstractArray3, ubV::AbstractArray3, dubV::AbstractArray4, q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix)
    logsumexp_and_softmax3!(lse,q,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
    sumdubV = @view(dubV[:,:,:,1])
    sumprod!(sumdubV,dubV,q)
    A_mul_B_md!(dEV0,Πz,sumdubV,1)
end

# destroys ubV and updates derivatives
vfit!(EV0::AbstractMatrix, dEV0::AbstractArray3, ubV::AbstractArray3, dubV::AbstractArray4, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix) = vfit!(EV0, dEV0, ubV, dubV, ubV, lse, tmp, Πz)

# --------------------------- VFIT until conv ----------------------------

function solve_inf_vfit!(EV0::AbstractMatrix, ubV::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Integer=6, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        vfit!(tmp, ubV, lse, tmp, Πz)
        bnds = extrema(tmp.-EV0) .* β ./ (1.0 .- β)
        ubV[:,:,1] .= β .* (EV0 .= tmp)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
    end
end

# --------------------------- basic pfit ----------------------------

function pfit!(EV0::AbstractMatrix, ubV::AbstractArray3, ΔEV::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; vftol::Real=1e-11)
    # initial vfit
    q0 = @view(ubV[:,:,1])
    logsumexp_and_softmax3!(ΔEV,q0,tmp,ubV)
    A_mul_B_md!(tmp, Πz, ΔEV, 1)

    # compute difference & check bnds
    bnds = extrema(ΔEV .= EV0 .- tmp) .* -β ./ (1.0 .- β)
    if all(abs.(bnds) .< vftol)
        EV0 .= tmp
        return bnds
    end

    # full PFit
    for j in 1:size(EV0, ndims(EV0))
        q0j = @view(q0[:,j])
        ΔEVj = @view(ΔEV[:,j])

        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lufact(IminusTEVp)
        A_ldiv_B!(fact, ΔEVj)                          # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= ΔEV                               # update V
    return extrema(ΔEV) .* -β ./ (1.0 .- β)   # get norm
end

# --------------------------- pfit until convergence ----------------------------

function solve_inf_pfit!(EV0::AbstractMatrix, ubV::AbstractArray3, ΔEV::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Integer=30, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        bnds = pfit!(EV0, ubV, ΔEV, tmp, IminusTEVp, Πz, β; vftol=vftol)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
        ubV[:,:,1] .= β .* EV0
    end
end

# --------------------------- inf horizon gradient ----------------------------


# note -- this destroys ubV
function gradinf!(dEV0::AbstractArray3{T}, ubV::AbstractArray3, dubV::AbstractArray4, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real) where {T}
    size(dubV,4) >= 2 || throw(DimensionMismatch("Need dubV with at least 2+ action possibilities"))

    sumdubV = @view(dubV[:,:,:,1])
    ΠsumdubV = @view(dubV[:,:,:,2])

    softmax3!(ubV, lse, tmp )
    sumprod!(sumdubV, dubV, ubV)
    A_mul_B_md!(ΠsumdubV, Πz, sumdubV, 1)

    # TODO: does copying over the data help??
    nθ = size(dubV,3)
    ΠsumdubVj = @view(lse[:,1:nθ]) # Array{T}(nz,nθ)
    dev0jtmp  = @view(tmp[:,1:nθ]) # Array{T}(nz,nθ)

    for j in 1:size(dubV,2)
        @views update_IminusTVp!(IminusTEVp, Πz, β, ubV[:,j,1])
        fact = lufact(IminusTEVp)
        @views ΠsumdubVj .= ΠsumdubV[:,j,:]

        # Note: cannot do this with @view(dEV0[:,j,:])
        @views A_ldiv_B!(dev0jtmp, fact, ΠsumdubVj) # ΠsumdubV[:,j,:])
        dEV0[:,j,:] .= dev0jtmp
    end
end

# --------------------------- helper function  ----------------------------


function sumprod!(red::AbstractArray3, big::AbstractArray4, small::AbstractArray3)
    nz,nψ,nv,nd = size(big)
    (nz,nψ,nd,) == size(small) || throw(DimensionMismatch())
    (nz,nψ,nv,) == size(red)   || throw(DimensionMismatch())

    # first loop w/ equals
    @inbounds for v in 1:nv, ψ in 1:nψ
        @simd for z in 1:nz
            red[z,ψ,v] = small[z,ψ,1] * big[z,ψ,v,1]
        end
    end

    # second set w/ plus equals
    @inbounds for d in 2:nd, v in 1:nv, ψ in 1:nψ
        @simd for z in 1:nz
            red[z,ψ,v] += small[z,ψ,d] * big[z,ψ,v,d]
        end
    end
end




function sumprod!(red::AbstractMatrix, big::AbstractArray3, small::AbstractArray3)
    nz,nψ,nd = size(big)
    (nz,nψ) == size(red) || throw(DimensionMismatch())
    (nz,nψ,nd) == size(small) || throw(DimensionMismatch())

    # first loop w/ equals
    @inbounds for d in 1:nd, ψ in 1:nψ
        @simd for z in 1:nz
            red[z,ψ] = small[z,ψ,1] * big[z,ψ,1]
        end
    end

    # second set w/ plus equals
    @inbounds for d in 2:nd, ψ in 1:nψ
        @simd for z in 1:nz
            red[z,ψ] += small[z,ψ,d] * big[z,ψ,d]
        end
    end
end
