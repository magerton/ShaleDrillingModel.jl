# --------------------------- basic VFIT ----------------------------

# simple Vfit
function vfit!(EV0::AbstractMatrix, lse::AbstractMatrix, tmp::AbstractMatrix, ubV::AbstractArray, Πz::AbstractMatrix)
    logsumexp3!(lse,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
end

# preserves ubV & updates derivatives
function vfit!(
    EV0::AbstractMatrix, dEV0::AbstractArray{T,3}, lse::AbstractMatrix, tmp::AbstractMatrix,
    q::AbstractArray{T,3}, sumdubV::AbstractArray{T,3},
    ubV::AbstractArray{T,3}, dubV::AbstractArray{T,4}, Πz::AbstractMatrix
    ) where {T}
    logsumexp_and_softmax3!(lse,q,tmp,ubV)
    A_mul_B_md!(EV0,Πz,lse,1)
    sumprod!(sumdubV,dubV,q)
    A_mul_B_md!(dEV0,Πz,sumdubV,1)
end

# destroys ubV and updates derivatives
vfit!(EV0::AbstractArray, dEV0::AbstractArray, lse::AbstractArray, tmp::AbstractArray, sumdubV::AbstractArray,
      ubV::AbstractArray, dubV::AbstractArray, Πz::AbstractMatrix) = vfit!(EV0, dEV0, lse, tmp, ubV, sumdubV, ubV, dubV, Πz)

# --------------------------- VFIT until conv ----------------------------

function solve_inf_vfit!(EV0::AbstractMatrix, tmp1::AbstractMatrix, tmp2::AbstractMatrix, ubV::AbstractArray, Πz::AbstractMatrix, β::Real; maxit::Int=6, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        vfit!(tmp2, tmp1, tmp2, ubV, Πz)
        bnds = extrema(tmp2.-EV0) .* β ./ (1.0 .- β)
        ubV[:,:,1] .= β .* (EV0 .= tmp2)
        iter += 1
        converged = all(abs.(bnds) .< vftol)
        if converged  ||  iter >= maxit
            return converged, iter, bnds
        end
    end
end



# --------------------------- basic pfit ----------------------------

function pfit!(EV0::AbstractMatrix, EVminusEVtmp::AbstractMatrix, EVtmp::AbstractMatrix, q0::AbstractMatrix, ubV::AbstractArray, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; vftol::Real=1e-11)
    # initial vfit
    logsumexp_and_softmax3!(EVminusEVtmp,q0,EVtmp,ubV)
    A_mul_B_md!(EVtmp, Πz, EVminusEVtmp, 1)

    # compute difference & check bnds
    bnds = extrema(EVminusEVtmp .= EV0 .- EVtmp) .* -β ./ (1.0 .- β)
    if all(abs.(bnds) .< vftol)
        EV0 .= EVtmp
        return bnds
    end

    # full PFit
    for j in 1:size(EV0, ndims(EV0))
        q0j = @view(q0[:,j])
        ΔEVj = @view(EVminusEVtmp[:,j])
        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lufact(IminusTEVp)
        A_ldiv_B!(fact, ΔEVj)                          # Vtmp = [I - T'(V)] \ [V - T(V)]
    end
    EV0 .-= EVminusEVtmp                               # update V
    return extrema(EVminusEVtmp) .* -β ./ (1.0 .- β)   # get norm
end

# --------------------------- pfit until convergence ----------------------------

function solve_inf_pfit!(EV0::AbstractMatrix, EVtmp::AbstractMatrix, logsumubV::AbstractMatrix, q0::AbstractMatrix, ubV::AbstractArray, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real; maxit::Int=20, vftol::Real=1e-11)
    iter = zero(maxit)
    while true
        bnds = pfit!(EV0, EVtmp, logsumubV, q0, ubV, IminusTEVp, Πz, β; vftol=vftol)
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
function gradinf!(dEV0::AbstractArray{T,3}, ubV::AbstractArray{T,3}, dubV::AbstractArray{T,4}, tmp1::AbstractMatrix, tmp2::AbstractMatrix, sumdubV::AbstractArray{T,3}, Πz_sumdubV::AbstractArray{T,3}, IminusTEVp::AbstractMatrix, Πz::AbstractMatrix, β::Real) where {T}
    softmax3!(ubV,tmp1,tmp2)
    sumprod!(sumdubV, dubV, ubV)
    A_mul_B_md!(Πz_sumdubV, Πz, sumdubV, 1)
    for j in 1:size(sumdubV, 2)
        q0j = @view(ubV[:,j,1])
        update_IminusTVp!(IminusTEVp, Πz, β, q0j)
        fact = lufact(IminusTEVp)
        A_ldiv_B!( @view(dEV0[:,j,:]), fact, @view(Πz_sumdubV[:,j,:]) )
    end
end


# --------------------------- helper function  ----------------------------


function sumprod!(red::AbstractArray{T,3}, big::AbstractArray{T,4}, small::AbstractArray{T,3}) where {T}
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
