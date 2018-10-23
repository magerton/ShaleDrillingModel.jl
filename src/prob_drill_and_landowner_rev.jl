export dcdp_tmpcntrfact, vwEUP, prDrill_infill!, serial_counterfact_all!

struct dcdp_tmpcntrfact{T<:Real}
    rin::Array{T,3}
    rex::Array{T,3}
    ubUfull::Array{T,3}
    q::Array{T,3}
end

function dcdp_tmpcntrfact(p::dcdp_primitives{FF,T}) where {FF,T}
    nz = _nz(p)
    nψ = _nψ(p)
    nd = _nd(p)
    rin = Array{T,3}(nz,nψ,nd)
    rex = similar(rin)
    ubUfull = similar(rin)
    q = similar(rin)
    return dcdp_tmpcntrfact{T}(rin,rex,ubUfull,q)
end

function vwEUP(sEU::AbstractArray{T,N}, sP::AbstractArray{T,N2}, typidx::Integer...) where {T,N,N2}
    ntyps = length(typidx)

    dimz = N - ntyps - 2

    szP = size(sP)
    zdims = szP[1:dimz]
    nψ, dmaxp1, nS = szP[dimz+(1:3)]
    typdims = szP[dimz+4:end]
    (zdims..., nψ, nS, typdims...) == size(sEU) || throw(DimensionMismatch())

    colons(n) = ntuple((x)-> Colon(), n)
    nz = prod(zdims)

    @views EU = reshape(sEU[colons(dimz+2)..., typidx...], nz, nψ, nS)
    @views P  = reshape(sP[ colons(dimz+3)..., typidx...], nz, nψ, dmaxp1, nS)

    return EU, P
end

function prDrill_infill!(
    EV::AbstractArray3, uin::AbstractArray4, uex::AbstractArray3, ubVfull::AbstractArray3,
    EU::AbstractArray3, rin::AbstractArray3, rex::AbstractArray3, ubUfull::AbstractArray3,
    P::AbstractArray4, q::AbstractArray3,

    lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,

    wp::well_problem, Πz::AbstractMatrix,

    Πψtmp::AbstractMatrix, ψspace::AbstractVector, σ::Real, β::Real
    )

    nz,nψ,nS = size(EV)
    dmaxp1 = dmax(wp)+1

    # ------------------------ size checks ----------------------------------

    (nz,nψ,nS) == size(EU)                           || throw(DimensionMismatch())
    (nz,nψ,dmaxp1,nS) == size(P)                     || throw(DimensionMismatch())
    (nz,nψ,dmaxp1,2) == size(uin)                    || throw(DimensionMismatch())
    (nz,nz) == size(IminusTEVp) == size(Πz)          || throw(DimensionMismatch())
    (nz,nψ) == size(tmp) == size(lse)                || throw(DimensionMismatch())
    (nz,nψ,dmaxp1) == size(rin) == size(rex) == size(q) || throw(DimensionMismatch())
    (nz,nψ,dmaxp1) == size(ubVfull) == size(ubUfull) || throw(DimensionMismatch())
    nψ == size(Πψtmp,1) == size(Πψtmp,2) == length(ψspace) || throw(DimensionMismatch())

    # ------------------------ compute things ----------------------------------

    # initialize P
    P .= 0.0

    # terminal conditions
    EV[:,:,end] .= 0.0
    EU[:,:,end] .= 0.0

    # ----------- Infill drilling ----------------
    for i in ind_inf(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        @views EV0  = EV[:,:,i]
        @views EU0  = EU[:,:,i]
        @views ubV  = ubVfull[:,:,idxd]
        @views ubU  = ubUfull[:,:,idxd]
        @views ubV .= uin[:,:,idxd,1+s.d1] .+ β .* EV[:,:,idxs]
        @views ubU .= rin[:,:,idxd] .+ β .* EU[:,:,idxs]

        # Make probabilities
        softmax3!(ubV, lse, tmp)
        P[:,:,idxd,i] .= ubV

        # Form expected revenue = (I⊗Πz)∑_d q_d .* ( u .+ β .* EV)
        sumprod!(lse, ubV, ubU)
        A_mul_B_md!(EU0, Πz, lse, 1)

        # If infinite horizon, handle this
        if horzn == :Infinite
            for j in 1:nψ
                @views update_IminusTVp!(IminusTEVp, Πz, β, ubV[:,j,1])
                fact = lufact(IminusTEVp)
                A_ldiv_B!(fact, EU0[:,j])
            end
        end
    end

    # ----------- Learning integration ----------------
    d2plus  = 2:dmax(wp)+1  # TODO: would be good to make this more general.
    lrn2inf = inf_fm_lrn(wp)
    exp2lrn = exploratory_learning(wp)

    # integrate over U
    @views ubU1 = ubUfull[:,:,d2plus]
    @views EU1  = EU[ :,:,lrn2inf]
    _βΠψ!(Πψtmp, ψspace, σ, β)
    A_mul_B_md!(ubU1, Πψtmp, EU1, 2)
    @views ubU1 .+= rex[:,:,d2plus]

    # ubVtilde = u[:,:,2:dmaxp1] + β * Πψ ⊗ I * EV[:,:,2:dmaxp1]
    ubVfull[:,:,d2plus] .= uex[:,:,d2plus] .+ β .* EV[:,:,exp2lrn]

    # ----------- Exploratory drilling ----------------
    for i in explore_state_inds(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)
        ip = action0(wp,i)

        @views EU0  = EU[:,:,i]

        @views ubV  = ubVfull[:,:,idxd]
        @views ubU  = ubUfull[:,:,idxd]
        @views qvw  = q[:,:,idxd]

        @views ubV[:,:,1] .= 0.0 + β .* EV[:,:,ip]
        @views ubU[:,:,1] .= 0.0 + β .* EU[:,:,ip]

        softmax3!(qvw, lse, tmp, ubV)
        P[:,:,idxd,i] .= qvw
        sumprod!(lse, qvw, ubU)
        A_mul_B_md!(EU0, Πz, lse, 1)
    end

end

function prDrill_infill!(evs::dcdp_Emax, tmpv::dcdp_tmpvars, prim::dcdp_primitives, EU::AbstractArray3, P::AbstractArray4, tmpc::dcdp_tmpcntrfact, σ::Real)
    return prDrill_infill!(
        evs.EV, tmpv.uin, tmpv.uex, tmpv.ubVfull,
        EU,     tmpc.rin, tmpc.rex, tmpc.ubUfull,
        P, tmpc.q,
        tmpv.lse, tmpv.tmp, tmpv.IminusTEVp,
        prim.wp, prim.Πz,
        tmpv.Πψtmp, prim.ψspace, σ, prim.β
    )
end

function prDrill_infill!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives{FF}, EU::AbstractArray3, P::AbstractArray4, tmpc::dcdp_tmpcntrfact, θ::AbstractVector, itype::Tuple) where {FF}
    θt, σ = _θt(θ, p, itype[2]), _σv(θ)
    pdct = makepdct(p, Val{:u})
    fillflows!(t, p, θt, σ, itype...)
    fillflowrevs!(FF, flowrev, tmpc.rin, tmpc.rex, θt, σ, pdct, itype...)
    prDrill_infill!(evs, t, p, EU, P, tmpc, σ)
end


function serial_counterfact_all!(sev::SharedEV, tmpv::dcdp_tmpvars, prim::dcdp_primitives, sEU::AbstractArray, sP::AbstractArray, tmpc::dcdp_tmpcntrfact, θ::AbstractVector)
    for Idx in CartesianRange( length.(sev.itypes) )
        evs, typs = dcdp_Emax(sev, Idx.I...)
        EU,P = vwEUP(sEU, sP, Idx.I...)
        itype = getindex.(sev.itypes, Idx.I)
        prDrill_infill!(evs, tmpv, prim, EU, P, tmpc, θ, itype)
    end
end
