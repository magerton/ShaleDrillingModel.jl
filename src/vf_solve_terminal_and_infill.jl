export solve_vf_terminal!, solve_vf_infill!, zero!


zero!(x::AbstractArray{T}) where {T<:Real} = fill!(x, zero(T))

function solve_vf_terminal!(EV::AbstractArray3)
    @views zero!(EV[:,:,end])
    @views zero!(EV[:,:,exploratory_terminal(wp)])
end


function solve_vf_terminal!(EV::AbstractArray3, dEV::AbstractArray4, dEVσ::AbstractArray3, wp::well_problem)
    @views zero!(EV[:,:,end])
    @views zero!(dEV[:,:,:,end])
    @views zero!(dEVσ[:,:,end])
    # @views zero!(dEV_ψ[:,:,end])
    exp_trm = exploratory_terminal(wp)
    @views zero!(EV[   :,:,  exp_trm])
    @views zero!(dEV[  :,:,:,exp_trm])
    @views zero!(dEVσ[:,:,  exp_trm])
    # @views zero!(dEV_ψ[:,:,  exp_trm])
end

solve_vf_terminal!(evs::dcdp_Emax, wp::well_problem) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, wp)
solve_vf_terminal!(evs::dcdp_Emax, prim::dcdp_primitives) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, prim.wp)

# ---------------------------------------------

function solve_vf_infill!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, dograd::Bool, itype::Tuple; maxit0::Integer=40, maxit1::Integer=20, vftol::Real=1e-9)

    # EV::AbstractArray3     , dEV::AbstractArray4     , dEVσ::AbstractArray3 , # dEV_ψ::AbstractArray3 ,  # complete VF
    # uex::AbstractArray3    , duex::AbstractArray4    , duexσ::AbstractArray3 , # duexψ::AbstractArray3 ,  # flow payoffs
    # ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray3, # dubV_ψ::AbstractArray3,  # choice-specific VF
    # q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix,                                        # temp vars
    # wp::well_problem, Πz::AbstractMatrix, β::Real,                                                      # transitions, etc
    # )

    EV         = evs.EV
    dEV        = evs.dEV
    uin        = t.u
    duin       = t.du
    ubVfull    = t.ubVfull
    dubVfull   = t.dubVfull
    lse        = t.lse
    tmp        = t.tmp
    IminusTEVp = t.IminusTEVp
    wp         = p.wp
    Πz         = p.Πz
    β          = p.β

    nz,nψ,nS = size(EV)
    dmaxp1 = dmax(wp)+1

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

    @views fillflows!(FF, flow,   t.u,   θ, σ, makepdct(p, Val{:u})...,  itype...)
    @views fillflows!(FF, flowdθ, t.du,  θ, σ, makepdct(p, Val{:du})..., itype...)
    fillflows!(       FF, flowdσ, t.duσ, θ, σ, makepdct(p, Val{:u})...,  itype...)

    updcts  = makepdct(p.zspace, p.ψspace, p.wp, length(θ), Val{:u})
    dupdcts = makepdct(p.zspace, p.ψspace, p.wp, length(θ), Val{:du})

    for i in ind_inf(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        @views ubV  = ubVfull[:,:,idxd]
        @views EV0  = EV[:,:,i]

        fillflows!(flow(p), flow, ubV, θ, σ, updcts..., s.d1, true, false, 0, itype...)
        @views ubV .+= β .* EV[:,:,idxs]

        if dograd
            @views dubV = dubVfull[:,:,:,idxd]
            @views dEV0 = dEV[:,:,:,i]

            fillflows!(flow(p), flowdθ, dubV, θ, σ, dupdcts..., s.d1, true, false, 0, itype...)
            fill!(dEV0, 0.0)
            @views dubV .+= β .* dEV[:,:,:,idxs]
        end

        if horzn == :Finite
            if dograd
                vfit!(EV0, dEV0, ubV, dubV, lse, tmp, Πz)
            else
                vfit!(EV0,       ubV,       lse, tmp, Πz)
            end

        elseif horzn == :Infinite
            solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=maxit0, vftol=vftol)

           # try-catch loop in case we have insane parameters that have Pr(no action) = 0, producing a singular IminusTEVp matrix.
            converged, iter, bnds = try
                solve_inf_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=maxit1, vftol=vftol)
            catch
                @warn "pfit threw error. trying vfit."
                solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=5000, vftol=vftol)
            end

            # converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds"
            if dograd
                # TODO: only allows 0-payoff if no action
                ubV[:,:,1] .= β .* EV0
                gradinf!(dEV0, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)   # note: destroys ubV & dubV
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
end
