function solve_vf_terminal!(EV::AbstractArray3{T}) where {T}
    EV[:,:,end] .= zero(T)
end

function solve_vf_terminal!(EV::AbstractArray3{T}, dEV::AbstractArray4{T}) where {T}
    solve_vf_terminal!(EV)
    length(dEV) > 0  &&  (dEV[:,:,:,end] .= zero(T))
end

function solve_vf_terminal!(EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4{T}) where {T}
    solve_vf_terminal!(EV,dEV)
    length(dEV_σ) > 0 && (dEV_σ[:,:,:,end] .= zero(T))
end




function solve_vf_infill!(EV::AbstractArray3{T}, uin::AbstractArray4, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, β::Real; kwargs...) where {T}
    zeros4 = Array{T}(0,0,0,0)
    zeros5 = Array{T}(0,0,0,0,0)
    solve_vf_infill!(EV, zeros4, uin, zeros5, ubVfull, zeros4, lse, tmp, IminusTEVp, wp, Πz, β; kwargs...)
end


function solve_vf_infill!(
    EV::AbstractArray3, dEV::AbstractArray4,                                  # value function
    uin::AbstractArray4, duin::AbstractArray5,                                # flow utilities
    ubVfull::AbstractArray3, dubVfull::AbstractArray4,                        # choice-specific VF
    lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,     # temp vars
    wp::well_problem, Πz::AbstractMatrix, β::Real;                            # problem structure
    maxit0::Integer=12, maxit1::Integer=20, vftol::Real=1e-11                 # convergence options
    )

    nz,nψ,nS = size(EV)
    dmaxp1 = dmax(wp)+1

    dograd = (length(dEV) > 0)

    # ------------------------ size checks ----------------------------------

    (nz,nψ,dmaxp1,2) == size(uin)           || throw(DimensionMismatch())  # uin[z,ψ,d,d1]
    (nz,nz) == size(IminusTEVp) == size(Πz) || throw(DimensionMismatch())
    (nz,nψ) == size(tmp) == size(lse)       || throw(DimensionMismatch())
    (nz,nψ,dmaxp1) == size(ubVfull)         || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)       == size(dEV)    || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1,2) == size(duin)   || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(dubVfull) || throw(DimensionMismatch())
    end

    # ------------------------ compute things ----------------------------------

    for i in infill_state_inds(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        ubV  = @view(ubVfull[:,:,idxd])
        EV0  = @view(EV[:,:,i])
        ubV .= @view(uin[:,:,idxd,1+s.d1]) .+ β .* @view(EV[:,:,idxs])

        if dograd
            dubV = @view(dubVfull[:,:,:,idxd])
            dEV0 = @view(dEV[:,:,:,i])
            dubV .= @view(duin[:,:,:,idxd,1+s.d1]) .+ β .* @view(dEV[:,:,:,idxs])
        end

        if horzn == :Finite
            dograd || vfit!(EV0,       ubV,       lse, tmp, Πz)
            dograd && vfit!(EV0, dEV0, ubV, dubV, lse, tmp, Πz)

        elseif horzn == :Infinite
            solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=maxit0, vftol=vftol)
            converged, iter, bnds = solve_inf_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=maxit1, vftol=vftol)
            converged || throw(error("Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds"))
            if dograd
                ubV[:,:,1] .= β .* EV0
                gradinf!(dEV0, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)   # note: destroys ubV & dubV
            end
        else
            throw(error("horizon must be :Finite or :Infinite"))
        end
    end
end


function learningUpdate!(ubV::AbstractArray3, uex::AbstractArray3, EV::AbstractArray3, idx::AbstractVector, βΠψ::AbstractMatrix)
    ubV1 = @view(ubV[:,:,2:end])
    EV1 = @view(EV[:,:,idx])
    uex1 = @view(uex[:,:,2:end])

    A_mul_B_md!(ubV1, βΠψ, EV1, 2)
    ubV[:,:,2:end] .+= uex1
end

function learningUpdate!(dubV::AbstractArray4, dubV_σ::AbstractArray4, duex::AbstractArray4, duexσ::AbstractArray4, EV::AbstractArray3, dEV::AbstractArray4, idx::AbstractVector, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix)
    dubV1 = @view(dubV[:,:,:,2:end])
    dubV_σ1 = @view(dubV_σ[:,:,:,2:end])
    EV1 = @view(EV[:,:,idx])
    dEV1 = @view(dEV[:,:,:,idx])
    duex1 = @view(duex[:,:,:,2:end])
    duex_σ1 = @view(duexσ[:,:,:,2:end])

    A_mul_B_md!(dubV1, βΠψ, dEV1, 2)
    dubV[  :,:,:,2:end] .+= duex1

    for d in 1:size(dubV1,4)
        dubV_σ1a = @view(dubV_σ1[:,:,1,d])
        dubV_σ1b = @view(dubV_σ1[:,:,2:end,d])
        EV1a = @view(EV1[:,:,d])
        A_mul_B_md!(dubV_σ1a, βdΠψ, EV1a, 2)
        dubV_σ1b .= dubV_σ1a
    end
    dubV_σ[:,:,:,2:end] .+= duex_σ1
end




function solve_vf_explore!(
    EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4,                          # complete VF
    uex::AbstractArray3, duex::AbstractArray4, duexσ::AbstractArray4,                        # flow payoffs
    ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray4,               # choice-specific VF
    q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix,                             # temp vars
    wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix, β::Real # transitions, etc
    )

    nz,nψ,nS = size(EV)
    nSex, dmaxp1, nd = length(explore_state_inds(wp)), exploratory_dmax(wp)+1, dmax(wp)+1
    s_idx2_fm_1 = infill_state_idx_from_exploratory(wp)

    dograd = (length(dEV) > 0)

    (nz,nψ,dmaxp1) == size(uex) || throw(DimensionMismatch())
    (nz,nψ,nd) == size(ubVfull) || throw(DimensionMismatch())
    (nz,nz) == size(Πz)         || throw(DimensionMismatch())
    (nψ,nψ) == size(βΠψ)        || throw(DimensionMismatch())
    (nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())

    if dograd
        nθ, nv = size(dEV,3), size(dEV_σ,3)
        (nz,nψ,nθ,nS)     == size(dEV)      || throw(DimensionMismatch())
        (nz,nψ,nv,nSex+1) == size(dEV_σ)    || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(duex)     || throw(DimensionMismatch())
        (nψ,nψ)           == size(βdΠψ)     || throw(DimensionMismatch())
        (nz,nψ,dmaxp1)    == size(q)        || throw(DimensionMismatch())
        (nz,nψ,nθ,nd)     == size(dubVfull) || throw(DimensionMismatch())
    end

    # --------- integrate over learning --------------

    ubV = @view(ubVfull[:,:,1:dmaxp1])
    learningUpdate!(ubV, uex, EV, s_idx2_fm_1, βΠψ)

    if dograd
        dubV = @view(dubVfull[:,:,:,1:dmaxp1])
        learningUpdate!(dubV, dubV_σ, duex, duexσ, EV, dEV, s_idx2_fm_1, βΠψ, βdΠψ)
    end

    # --------- VFIt --------------

    for i in explore_state_inds(wp)
        ip = action0(wp,i)

        EV0 = @view(EV[:,:,i])

        ubV[:,:,1] .= β .* @view(EV[:,:,ip])

        if dograd
            dEV0 = @view(dEV[:,:,:,i])
            dEV1 = @view(dEV[:,:,:,ip])
            dEV0_σ = @view(dEV_σ[:,:,:,i])
            dEV1_σ = @view(dEV_σ[:,:,:,min(ip,nSex+1)])  # because we don't use regime2 but need a TVC
            sumdubV_σ = @view(dubV_σ[:,:,:,1])

            dubV[:,:,:,1] .= β .* dEV1
            dubV_σ[:,:,:,1] .= β .* dEV1_σ

            vfit!(EV0, dEV0, ubV, dubV, q, lse, tmp, Πz)
            sumprod!(sumdubV_σ, dubV_σ, q)
            A_mul_B_md!(dEV0_σ, Πz, sumdubV_σ, 1)
        else
            vfit!(EV0, ubV, lse, tmp, Πz)
        end
    end
end


function solve_vf_explore!(EV::AbstractArray3{T}, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, β::Real) where {T}
    zeros4 = Array{T}(0,0,0,0)
    zeros3 = Array{T}(0,0,0)
    zeros2 = Array{T}(0,0)
    solve_vf_explore!(EV, zeros4, zeros4, uex, zeros4, zeros4, ubVfull, zeros4, zeros4, zeros3, lse, tmp, wp, Πz, βΠψ, zeros2, β)
end






function solve_vf_all!(
    EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4,                           # complete VF
    uin::AbstractArray4, uex::AbstractArray3,                                                 # flow payofs
    duin::AbstractArray5, duex::AbstractArray4, duexσ::AbstractArray4,                        # flow gradient
    ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray4,                # choice-specific VF
    q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,  # temp vars
    wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix, β::Real; # transitions, etc
    kwargs...
    )
    solve_vf_terminal!(EV, dEV, dEV_σ)
    solve_vf_infill!(  EV, dEV,        uin, duin,        ubVfull, dubVfull,            lse, tmp, IminusTEVp, wp, Πz,            β)
    solve_vf_explore!( EV, dEV, dEV_σ, uex, duex, duexσ, ubVfull, dubVfull, dubV_σ, q, lse, tmp,             wp, Πz, βΠψ, βdΠψ, β)
end


function solve_vf_all!(EV::AbstractArray3, uin::AbstractArray4, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix,  β::Real; kwargs...)
    solve_vf_terminal!(EV)
    solve_vf_infill!(  EV, uin, ubVfull, lse, tmp, IminusTEVp, wp, Πz,      β)
    solve_vf_explore!( EV, uex, ubVfull, lse, tmp,             wp, Πz, βΠψ, β)
end





#
