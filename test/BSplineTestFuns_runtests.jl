using SharedArrays
using Interpolations: BSplineInterpolation, tweight, scale
using Random

@testset "bspline funs" begin
    # using BSplineExtensions
    # using Interpolations
    # using Test

    x = SharedArray{Float64}(100, 100, 10, 10)
    endsz = (map(i -> size(x,i), 3:4)...,)
    rand!(x)

    x2210 = copy(x)
    x2211 = copy(x)
    x1111 = copy(x)

    b1 = BSpline(Linear())
    b2 = BSpline(Quadratic(InPlace()))
    b0 = NoInterp()

    it2210 = (b2, b2, b1, b0)
    it2211 = (b2, b2, b1, b1)
    it1111 = (b1, b1, b1, b1)

    # interpolate a bunch of stuff in place
    itp_2210 = interpolate!(x2210, it2210, OnCell())
    itp_2211 = interpolate!(x2211, it2211, OnCell())
    itp_1111 = interpolate!(x1111, it1111, OnCell())

    @test itp_2210.coefs === x2210
    @test itp_2211.coefs === x2211
    @test itp_1111.coefs === x1111

    @test !all(x2210 .== x)
    @test !all(x2211 .== x)
    @test  all(x1111 .== x)

    # store these coefs
    coefs2210 = copy(x2210)
    coefs2211 = copy(x2211)
    coefs1111 = copy(x1111)

    # restore original x
    x2210 .= x
    x2211 .= x
    x1111 .= x

    # interpolate w/ homemade algorithm and see what we get!
    myitp_2210 = BSplineInterpolation(tweight(x2210), x2210, it2210, OnCell(), Val{0}())
    myitp_2211 = BSplineInterpolation(tweight(x2211), x2211, it2211, OnCell(), Val{0}())
    myitp_1111 = BSplineInterpolation(tweight(x1111), x1111, it1111, OnCell(), Val{0}())

    @test typeof(myitp_2210) == typeof(itp_2210)
    @test typeof(myitp_2211) == typeof(itp_2211)
    @test typeof(myitp_1111) == typeof(itp_1111)

    @test myitp_2210.coefs === x2210
    @test myitp_2211.coefs === x2211
    @test myitp_1111.coefs === x1111

    serial_prefilterByView!(x2210, myitp_2210, endsz...)
    serial_prefilterByView!(x2211, myitp_2211, endsz...)
    serial_prefilterByView!(x1111, myitp_1111, endsz...)

    @test all(myitp_2210.coefs .== coefs2210)
    @test all(myitp_2211.coefs .== coefs2211)
    @test all(myitp_1111.coefs .== coefs1111)

    gradient_d_impl(3, typeof(itp_2210))
    # gradient_impl(typeof(itp_2210))

    function test_grad_d(itp::BSplineInterpolation, x::Number...)
        g = Interpolations.gradient(itp, x...)
        for d = 1:length(g)
            @test g[d] == gradient_d(Val{d}, itp, x...)
        end
    end

    xs = 1.2, 2.2, 3.2, Int(4)
    # @code_warntype gradient_d(Val{1}, itp_2210, xs...)
    gradient!(Vector{Float64}(undef, 3), itp_2210, xs...)

    Interpolations.gradient(itp_2210, xs...)
    test_grad_d(itp_2210, xs...)
    test_grad_d(itp_1111, xs...)
    test_grad_d(itp_2211, xs...)

    @test_throws DomainError    gradient_d(Val{5}, itp_1111, 1:4...)
    @test_throws ErrorException gradient_d(Val{4}, itp_2210, 1:4...)

    # ------------------------ scale & gradient ---------------------

    rng1 = range(-30, stop=30, length=100)
    rng2 = range(20, stop=50, length=100)
    rng3 = range(30, stop=39, length=10)
    rng4 = 1:10

    sitp_2210 = scale(itp_2210, rng1, rng2, rng3, rng4)

    Random.seed!(1234)
    z0 = rand(10,10)
    za = copy(z0)
    zb = copy(z0')
    @test all(za .== zb')

    itpa = interpolate(za, (BSpline(Linear()), NoInterp()), OnGrid())
    itpb = interpolate(zb, (NoInterp(), BSpline(Linear())), OnGrid())
    @test all(itpa.coefs .== itpb.coefs')

    rng = range(1.0, stop=19.0, length=10)
    sitpa = scale(itpa, rng, 1:10)
    sitpb = scale(itpb, 1:10, rng)
    @test sitpa[2.0, 3] == sitpb[3,2.0]
    @test sitpa[2.0, 3] != sitpb[3,2.1]
    @test itpa[2.0, 3] == sitpa[3.0, 3]


    @test Interpolations.gradient(itpa, 2.0, 3)/step(rng) == Interpolations.gradient(sitpa, 3.0, 3)
    @test Interpolations.gradient(itpa, 2.0, 3)/step(rng) == Interpolations.gradient(itpb, 3, 2.0)/step(rng)
    # @test Interpolations.gradient(itpb, 3, 2.0)/step(rng) == Interpolations.gradient(sitpb, 3, 3.0)

    # @test Interpolations.gradient(sitpb, 3, 3.0) == Interpolations.gradient(sitpa, 3.0, 3)
    @test ShaleDrillingModel.fixedgradient(sitpa, 3.0, 3) == ShaleDrillingModel.fixedgradient(sitpb, 3, 3.0)
    @test gradient_d(Val{1}, sitpa, 3.0, 3) ==  ShaleDrillingModel.fixedgradient(sitpa, 3.0, 3)[1]

    println("done!")

    # Interpolations.gradient(sitpb, 3, 2.0)

    # @show mygradient(sitpa, 2.0, 3) , mygradient(sitpb, 3, 2.0)
    # @test Interpolations.gradient(sitpa, 2.0, 3) ==  Interpolations.gradient(sitpb, 3, 2.0)

    # @test Interpolations.gradient(sitpa, 2.0, 3) ==  Interpolations.gradient(sitpb, 3, 2.0)

    # it2201 = (b2, b2, b0, b1)
    # x2201 = copy(x)
    # itp_2201 = interpolate!(x2201, it2201, OnCell())
    # sitp_2201 = scale(itp_2201, rng1, rng2, rng4, rng3)
    #
    # function f(sitp::ScaledInterpolation{T,N,ITPT,IT,GT}) where {T,N,ITPT,IT<:DimSpec{InterpolationType},GT<:DimSpec{GridType}}
    #     @show IT.parameters
    #     @show length(IT.parameters)
    #     @show interp_types = length(IT.parameters) == N ? IT.parameters : tuple([IT.parameters[1] for _ in 1:N]...)
    #     @show interp_dimens = map(it -> interp_types[it] != NoInterp, 1:N)
    #     @show interp_indices = map(i -> interp_dimens[i] ? :(coordlookup(sitp.ranges[$i], xs[$i])) : :(xs[$i]), 1:N)
    # end
    #
    # mygradient(sitp_2201, 1.2, 2.2, Int(4), 3.2)
    # mygradient(sitp_2210, 1.2, 2.2, 3.2, Int(4))


    # f(sitp_2201)
end
