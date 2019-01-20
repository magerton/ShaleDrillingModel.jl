using Interpolations

to11interval(x,a,b) = 2*(x - a)/(b-a)-1

years = to11interval.(2003:2016,-1,1)
year_space = LinRange(-1,1,14)


A = [log(x) for x in years]



# linear interpolation
interp_linear = LinearInterpolation(xs, A)
interp_linear(3) # exactly log(3)
interp_linear(3.1) # approximately log(3.1)

# cubic spline interpolation
interp_cubic = CubicSplineInterpolation(xs, A)
interp_cubic(3) # exactly log(3)
interp_cubic(3.1) # approximately log(3.1)
