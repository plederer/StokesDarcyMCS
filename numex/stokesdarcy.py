from netgen.occ import *
from ngsolve import *
wp = WorkPlane()

recD = wp.Rectangle(1, 1).Face()
recD.edges[0].name = "bottom_D"
recD.edges[1].name = "right_D"
recD.edges[2].name = "interface"
recD.edges[3].name = "left_D"
recD.faces[0].name = "Omega_D"

recS = wp.MoveTo(0,1).Rectangle(1, 1).Face()
recS.edges[0].name = "interface"
recS.edges[1].name = "right_S"
recS.edges[2].name = "top_S"
recS.edges[3].name = "left_S"
recS.faces[0].name = "Omega_S"

# geo = OCCGeometry(Glue([recS,recD]), dim=2)
geo = OCCGeometry(Glue([recS,recD]), dim=2)
mesh = Mesh( geo.GenerateMesh(maxh=0.1))

# dir_bnd = "bottom_S|top_D|right_S|right_D|left_S|left_D"
dir_bnd = "top_S|right_S|left_S"


# print(mesh.GetBoundaries())
# print(mesh.GetMaterials())
# Draw(mesh)

# V = H1(mesh, order = 2, order_policy=ORDER_POLICY.VARIABLE)

# print(V.ndof)
# for el in mesh.Materials("Omega_D").Elements():
#     no = NodeId(FACE, el.nr)
#     V.SetOrder(no, 10)
# V.UpdateDofTables()
# print(V.ndof)
# # NODE
nu = 1
K = 1
G = 1

p_D = - pi/4 * cos(pi * x / 2) * y
p_S = -pi/4 * cos(pi * x /2) * (y - 2 * cos(pi * y/2)**2)
u_S = CF((- cos(pi*y/2)**2*sin(pi*x/2), 1/4* cos(pi*x/2)* (sin(pi*y) + pi * y)))
u_D = K * CF((p_D.Diff(x),p_D.Diff(y)))

f_D = -(u_D[0].Diff(x) + u_D[0].Diff(y))

grad_u_S = CF((u_S[0].Diff(x), u_S[0].Diff(y), u_S[1].Diff(x), u_S[1].Diff(y)), dims = (2,2))

f_S = -nu * CF((grad_u_S[0,0].Diff(x) + grad_u_S[0,1].Diff(y), grad_u_S[1,0].Diff(x) + grad_u_S[1,1].Diff(y))) + CF((p_S.Diff(x), p_S.Diff(y)))


Draw(p_D, mesh, "p_D")
Draw(p_S, mesh, "p_S")
Draw(u_S, mesh, "u_S")
Draw(u_S[0].Diff(x) + u_S[1].Diff(y), mesh, "div_uS")
Draw(u_D, mesh, "u_D")

print("mean value = ", Integrate(p_S, mesh, definedon = mesh.Materials("Omega_S")))



order = 3
el_int = False


Vs = HDiv(mesh, order=order, dirichlet=dir_bnd)
S = HCurlDiv(mesh, order = order-1, orderinner = order) #, definedon = mesh.Materials("Omega_S"))        
Q = L2(mesh, order = order-1, lowest_order_wb = el_int  )        
V = S * Vs * Q

sigma, u, p = V.TrialFunction()
tau, v, q = V.TestFunction()

n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size

def tang(vec):
    return vec - (vec*n)*n

def trace(mat):
    return 0.5 * (mat[0,0] + mat[1,1])

a = BilinearForm (V, eliminate_internal = el_int, symmetric = True)

a += -1/nu * InnerProduct ( sigma,tau) * dx("Omega_S")
a += -(div(sigma) * v + div(tau) * u) * dx("Omega_S")
a += (((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n))*dx(element_boundary = True, definedon = mesh.Materials("Omega_S"))
a += (-div(u) * q - div(v)*p) * dx()
a += -1e-8 * p * q * dx
a += -1/K *u * v * dx("Omega_D")
# a += -G * tang(sigma*n) * tang(tau*n) * ds("gamma", skeleton = True)


# force_S = CF((0,0))
# force_D = 1
f = LinearForm(V)
f += f_S * v * dx("Omega_S", bonus_intorder=10)
# f += f_D * q * dx("Omega_D")

# f += -p_D * v.Trace()*n * ds("bottom_D|right_D|left_D")
f += -tang(u_S) * tang(tau.Trace()*n) * ds("top_S|right_S|left_S|gamma")
# f += -p_D * v.Trace() * n * ds("gamma") 

gfu = GridFunction(V)
gfu.components[1].Set(u_S, definedon = mesh.Boundaries("top_S|right_S|left_S|gamma"))
res = gfu.vec.CreateVector()

Draw(gfu.components[0], mesh, "sigma")
Draw(gfu.components[1], mesh, "u")
Draw(gfu.components[2], mesh, "p")
# input()

stokes_dofs = V.FreeDofs() & V.GetDofs(mesh.Materials("Omega_S"))
# print(stokes_dofs)

with TaskManager():
    a.Assemble()        
    f.Assemble()
        
    # inv = a.mat.Inverse(V.FreeDofs(el_int), inverse = "umfpack")
    inv = a.mat.Inverse(stokes_dofs, inverse = "umfpack")

    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += inv * res

Draw(gfu.components[1]-u_S, mesh, "error")
Redraw()