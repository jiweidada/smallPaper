import torch
# Reproduce exactly: m / u where u tiny denormal, then minimum with 1
m=torch.tensor([0.01], requires_grad=False)
u=torch.tensor([4.7181376310816106e-26], requires_grad=True)
r = m/u
print("m/u =", float(r))
s = torch.minimum(torch.ones_like(r), r)
s.sum().backward()
print("grad u via minimum:", u.grad, "nan", bool(torch.isnan(u.grad)))
# now the key: is u^2 underflow the cause? check grad formula m/u^2
u2=torch.tensor([4.7181376310816106e-26], requires_grad=True)
r2=m/u2
r2.sum().backward()
print("grad u direct divide:", u2.grad, "isinf", bool(torch.isinf(u2.grad)))
# and 0 * inf
z=torch.tensor([0.0], requires_grad=False)
g=torch.tensor([float('inf')])
print("0*inf =", float(z*g))
