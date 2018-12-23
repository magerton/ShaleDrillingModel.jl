@testset "Maxlease" begin
    for wpp in (well_problem(3,4, 5,3,2), well_problem(3,4, 2,3,2), well_problem(3,4, 1,3,5), well_problem(3,16, 5,10,3))
        @test maximum(ShaleDrillingModel._τrem(s) for s in wpp.SS) == ShaleDrillingModel.maxlease(wpp)
    end
end

@testset "Drilling transitions are ok" begin
    wp = well_problem(4, 4, 5, 3, 2)
    nS = length(wp)

    # infill: d1 == 0 cascades down
    for i = wp.endpts[4]+2 : 2 : nS-1
        @test ShaleDrillingModel._sprime(i,0,wp) == i
    end

    # infill: d1 == 1
    for i = wp.endpts[4]+1 : 2 : nS-2
        @test ShaleDrillingModel._sprime(i,0,wp) == i+1
        for d in 1:ShaleDrillingModel._max_action(i,wp)
            @test ShaleDrillingModel._sprime(i,d,wp) == i+2*d
        end
    end

    # initial lease term
    for i = 1 :  wp.endpts[2]-1
        @test ShaleDrillingModel._sprime(i,0,wp) == i+1
    end

    # last lease term
    for i = wp.endpts[2]+1 : wp.endpts[3]
        @test ShaleDrillingModel._sprime(i,0,wp) == i+1
    end

    # any drilling during a primary term
    for i = wp.endpts[1]+1 : wp.endpts[3]
        for d = 1:4
            @test ShaleDrillingModel._sprime(i,d,wp) == wp.endpts[3]+1+d
        end
    end
end

@testset "sprime is in 1:length(wp)?" begin
    for dmx in 3:4
        wp = well_problem(dmx, 4, 5, 3, 2)
        for (i,s) in enumerate(wp.SS)
            for d in  ShaleDrillingModel._actionspace(i,wp)
                sp = ShaleDrillingModel._sprime(i,d,wp)
                s = wp.SS[i]  # this state
                t = wp.SS[sp] # next state
                sp ∈ 1:length(wp) || println("dmax = $dmx d = $d, i = $i with si = $(wp.SS[i]), sprime = $(ShaleDrillingModel._sprime(i,d,wp))")
                @test sp ∈ 1:length(wp)
                @test t.D == s.D + d
                if  i < length(wp) && i ∉ ShaleDrillingModel.ind_lrn(wp.endpts)
                    @test t.d1 == Int(d > 0)
                end

                if i ∈ ShaleDrillingModel.ind_ex1(wp.endpts)
                    @test t.τ1 == ( d == 0 ? s.τ1 -1 : -1)
                    @test t.τ0 == ( d == 0 ? s.τ0 - (i == wp.endpts[2]) : -1)
                end

                if i ∈ ShaleDrillingModel.ind_ex0(wp.endpts)
                    @test t.τ1 ==  s.τ1 == -1
                    @test t.τ0 == ( d == 0 ? s.τ0 - 1 : -1)
                end

            end
        end
    end
end


@testset "make State Space" begin

    let dmx = 4,
        Dmx = 4,
        τ0mx = 5,
        τ1mx = 3,
        emx = 2,
        wp = well_problem(dmx, Dmx, τ0mx, τ1mx, emx)

        ep = ShaleDrillingModel.end_pts(    dmx, Dmx, τ0mx, τ1mx, emx)
        SS = ShaleDrillingModel.state_space(dmx, Dmx, τ0mx, τ1mx, emx)

        ShaleDrillingModel.end_pts(    dmx, Dmx, τ0mx)
        ShaleDrillingModel.state_space(dmx, Dmx, τ0mx)

        SS[ 1:ep[2] ]                                # Exploratory 1
        SS[ 1:ep[3] ]                                # Exploratory 0
        SS[  ep[3] .+ (1:dmx+1)]                     # Exploratory Terminal + learning update
        SS[  (τ0mx + τ1mx + dmx+3) .+ (1:2*Dmx-2)]   # Infill drilling
        SS[  (τ0mx + τ1mx + dmx+3)  +   2*Dmx-2+1]   # Terminal

        nS = length(SS)
        [ShaleDrillingModel._actionspace(i, dmx, Dmx, ep) for i in 1:nS]

        dDte = dmx,Dmx,ep
        a = SS, [
            (i,
            ShaleDrillingModel._regime(i,ep),
            ShaleDrillingModel._τ1(i,ep),
            ShaleDrillingModel._τ0(i,ep),
            ShaleDrillingModel._D(i,ep),
            ShaleDrillingModel._d1(i,ep),
            collect(ShaleDrillingModel._sprime(i,d,ep) for d in ShaleDrillingModel._actionspace(i,dDte...))
            ) for i in 1:nS]


        idxs = [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)..., ShaleDrillingModel.learn_state_inds(wp)...]

        @test idxs ⊆ 1:length(wp)
        @test 1:length(wp) ⊆ idxs

        # test that we get back the state we want to get.
        for i in [ShaleDrillingModel.explore_state_inds(wp)..., ShaleDrillingModel.infill_state_inds(wp)..., ShaleDrillingModel.terminal_state_ind(wp)...,]
            st = SS[i]
            i_of_st = state_idx(st.τ1, st.τ0, st.D, st.d1,     dmx, Dmx, τ0mx, τ1mx, emx)
            @test i_of_st == i
        end

        for s in ShaleDrillingModel.ind_inf(wp)
            @test ShaleDrillingModel._horizon(s,wp.endpts) ∈ (:Infinite, :Finite)
        end

        for s in ShaleDrillingModel.ind_exp(wp.endpts)
            @test ShaleDrillingModel._horizon(s,wp.endpts) == :Finite
        end

        for s in ShaleDrillingModel.ind_lrn(wp.endpts)
            @test ShaleDrillingModel._horizon(s,wp.endpts) ∈ (:Terminal, :Learning)
        end
    end
end
