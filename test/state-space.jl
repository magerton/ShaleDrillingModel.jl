@testset "Maxlease and state_if_never_drilled" begin
    for unitprob ∈ (LeasedProblem, LeasedProblemContsDrill,)
        for wpp in (unitprob(3,4, 5,3,2), unitprob(3,4, 2,3,2), unitprob(3,4, 1,3,5), unitprob(3,16, 5,10,3), unitprob(3,4,1,-1,0))
            SS = ShaleDrillingModel.state_space_vector(wpp)
            @test maximum(ShaleDrillingModel._τrem(s) for s in SS) == ShaleDrillingModel.maxlease(wpp)
            for sidx ∈ 1:length(wpp)
                for d in ShaleDrillingModel.actionspace(wpp,sidx)
                    @test sprime(wpp,sidx,d) ∈ 1:length(wpp)
                end
            end

            # test state if never drilled...
            for state0 = 1:ShaleDrillingModel.end_ex0(wpp)
                statet = state0
                for t = 1:ShaleDrillingModel.end_ex0(wpp)
                    statet = sprime(wpp,statet,0)
                    ifno_drill = ShaleDrillingModel.state_if_never_drilled(wpp,state0,t)
                    @test statet == ifno_drill
                    # println("t,state0 = ($t,$state0): recursive sprime = $statet, state_if_never_drilled = $ifno_drill")
                end
            end

        end
    end
end

@testset "Drilling transitions are ok" begin
    for unitprob ∈ (LeasedProblem, LeasedProblemContsDrill,PerpetualProblem,)
        wp = unitprob(4, 4, 5, 3, 2)
        nS = length(wp)

        # infill: d1 == 0 cascades down
        for i = end_lrn(wp)+ShaleDrillingModel._nstates_per_D(wp) : ShaleDrillingModel._nstates_per_D(wp) : nS-1
            @test sprime(wp,i,0) == i
        end

        # infill: d1 == 1
        if isa(wp,LeasedProblemContsDrill)
            for i = end_lrn(wp)+1 : 2 : nS-2
                @test sprime(wp,i,0) == i+1
                for d in 1:ShaleDrillingModel._dmax(wp,i)
                    @test sprime(wp,i,d) == i+2*d
                end
            end
        end

        # initial lease term
        if isa(wp,PerpetualProblem)
            @test sprime(wp,1,0) == 1
        else
            for i = 1 :  end_ex1(wp)-1
                @test sprime(wp,i,0) == i+1
            end

            # last lease term
            for i = end_ex1(wp)+1 : end_ex0(wp)
                @test sprime(wp,i,0) == i+1
            end

            # any drilling during a primary term
            for i = 0+1 : end_ex0(wp)
                for d = 1:4
                    @test sprime(wp,i,d) == end_ex0(wp)+1+d
                end
            end
        end
    end
end

@testset "sprime is in 1:length(wp)?" begin
    for unitprob ∈ (ShaleDrillingModel.LeasedProblem, ShaleDrillingModel.LeasedProblemContsDrill, ShaleDrillingModel.PerpetualProblem,)
        for dmx in 3:4
            wp = unitprob(dmx, 4, 5, 3, 2)
            SS = ShaleDrillingModel.state_space_vector(wp)

            @test length(ShaleDrillingModel.inf_fm_lrn(wp)) == length(ShaleDrillingModel.exploratory_learning(wp))

            for (i,s) in enumerate(SS)
                for d in  ShaleDrillingModel.actionspace(wp,i)
                    sp = sprime(wp,i,d)
                    s = SS[i]  # this state
                    t = SS[sp] # next state
                    sp ∈ 1:length(wp) || println("dmax = $dmx d = $d, i = $i with si = $(SS[i]), sprime = $(sprime(wp,i,d))")
                    @test sp ∈ 1:length(wp)
                    @test t.D == s.D + d

                    @test SS[i] == state(wp,i)
                    @test SS[sp] == state(wp,sp)

                    if  i < length(wp) && i ∉ ShaleDrillingModel.ind_lrn(wp) && isa(wp,LeasedProblemContsDrill)
                        @test t.d1 == Int(d > 0)
                    end

                    if i ∈ ShaleDrillingModel.ind_ex1(wp)
                        @test t.τ1 == ( d == 0 ? s.τ1 -1 : -1)
                        @test t.τ0 == ( d == 0 ? s.τ0 - (i == end_ex1(wp)) : -1)
                    end

                    if i ∈ ShaleDrillingModel.ind_ex0(wp)
                        @test t.τ1 ==  s.τ1 == -1
                        if isa(wp, PerpetualProblem)
                            @test t.τ0 == -1
                        else
                            @test t.τ0 == ( d == 0 ? s.τ0 - 1 : -1)
                        end
                    end

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
        emx = 2

        for unitprob ∈ (ShaleDrillingModel.LeasedProblem, ShaleDrillingModel.LeasedProblemContsDrill, ShaleDrillingModel.PerpetualProblem,)
            wp = unitprob(dmx, Dmx, τ0mx, τ1mx, emx)

            SS = ShaleDrillingModel.state_space_vector(wp)

            SS[ 1:end_ex1(wp) ]                                # Exploratory 1
            SS[ 1:end_ex0(wp) ]                                # Exploratory 0
            SS[  end_ex0(wp) .+ (1:dmx+1)]                     # Exploratory Terminal + learning update
            # SS[  (τ0mx + τ1mx + dmx+3) .+ (1:2*Dmx-2)]   # Infill drilling
            # SS[  (τ0mx + τ1mx + dmx+3)  +   2*Dmx-2+1]   # Terminal

            nS = length(SS)
            [ShaleDrillingModel.actionspace(wp,i) for i in 1:nS]

            a = SS, [
                (i,
                ShaleDrillingModel._regime(wp,i),
                ShaleDrillingModel._τ1(wp,i),
                ShaleDrillingModel._τ0(wp,i),
                ShaleDrillingModel._D(wp,i),
                ShaleDrillingModel._d1(wp,i),
                collect(sprime(wp,i,d) for d in ShaleDrillingModel.actionspace(wp,i))
                ) for i in 1:nS]

            idxs = [ShaleDrillingModel.ind_exp(wp)..., ShaleDrillingModel.ind_inf(wp)..., end_inf(wp)..., ShaleDrillingModel.ind_lrn(wp)...]

            @test idxs ⊆ 1:length(wp)
            @test 1:length(wp) ⊆ idxs

            # test that we get back the state we want to get.
            for i in [ShaleDrillingModel.ind_exp(wp)..., ShaleDrillingModel.ind_inf(wp)..., end_inf(wp)...,]
                st = SS[i]
                i_of_st = state_idx(wp, st.τ1, st.τ0, st.D, st.d1)
                @test i_of_st == i
                @test i ∈ ShaleDrillingModel.s_of_D(wp, st.D)
            end

            for s in ShaleDrillingModel.ind_inf(wp)
                @test ShaleDrillingModel._horizon(wp,s) ∈ (:Infinite, :Finite)
            end

            for s in ShaleDrillingModel.ind_exp(wp)
                if isa(wp,PerpetualProblem)
                    @test ShaleDrillingModel._horizon(wp,s) == :Infinite
                else
                    @test ShaleDrillingModel._horizon(wp,s) == :Finite
                end
            end

            for s in ShaleDrillingModel.ind_lrn(wp)
                @test ShaleDrillingModel._horizon(wp,s) ∈ (:Terminal, :Learning)
            end
        end
    end
end


@testset "Checking on dims of dEVσ" begin
    for unitprob ∈ (LeasedProblem, LeasedProblemContsDrill,)
        for wpp in (unitprob(3,4, 5,3,2), unitprob(3,4, 2,3,2), unitprob(3,4, 1,3,5), unitprob(3,16, 5,10,3), unitprob(3,4,1,-1,0))
            nsexp = ShaleDrillingModel._nSexp(wpp)
            for sidx ∈ 1:length(wpp)
                if sidx <= end_ex0(wpp)
                    for d in ShaleDrillingModel.actionspace(wpp,sidx)
                        @test sprime(wpp,sidx,d) ∈ 1:nsexp
                    end
                end
            end
        end
    end
end
