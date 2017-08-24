def abs (x)
  if x > 0
    return x
  else
    return -x
  end
end

def sqrt (x)
  eps = 1e-10
  x = x * 1.0
  r = x / 2
  residual = r ** 2 - x
  while abs(residual) > eps
    r_d = -residual / (2 * r)
    r += r_d
    residual = r ** 2 - x
  end
  return r
end

puts(sqrt(1))
puts(sqrt(2))
puts(sqrt(3))
puts(sqrt(4))
puts(sqrt(5))
puts(sqrt(6))
puts(sqrt(7000))
