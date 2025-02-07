from dolfin import *
from mshr import *
from ufl import replace
import numpy as np
import csv
import matplotlib.pyplot as plt
from math import *
from ufl import grad as ufl_grad
import sys
sys.setrecursionlimit(15000)

def my_range(start, end, step):
    while start <= end+step/2:
        yield start
        start += step

def Grad(v):
    return ufl_grad(v)

# Second order identity tensor
def SecondOrderIdentity(u):
    d = u.geometric_dimension()
    return variable(Identity(d))

# Deformation gradient
def DeformationGradient(u):
    I = SecondOrderIdentity(u)
    return variable(I + Grad(u))

# Determinant of the deformation gradient
def Jacobian(u):
    F = DeformationGradient(u)
    return variable(det(F))

# Right Cauchy-Green tensor
def RightCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F.T*F)

# Left Cauchy-Green tensor
def LeftCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F*F.T)

# Invariants of an arbitrary tensor, A
def Invariants(A):
    I1 = tr(A)
    I2 = 0.5*(tr(A)**2 - tr(A*A))
    I3 = det(A)
    return [variable(I1), variable(I2), variable(I3)]

# Elastic Cauchy stress tensor 
def elastic_Cauchy(u):
    F = DeformationGradient(u)
    B = LeftCauchyGreen(u)
    J = Jacobian(u)
    I = SecondOrderIdentity(u)
    I1, I2, I3 = Invariants(B)
   
    psi = mu_0/2*(1/2+gamma)*(I1 - 3) + mu_0/2*(1/2-gamma)*(I2 - 3)  
    W1 = diff(psi,I1) 
    W2 = diff(psi,I2)
    W3 = diff(psi,I3)
    
    beta0 = 2/J*(I2*W2+I3*W3)
    beta1 = 2/J*W1
    beta_1 = -2*J*W2
    
    Te = beta0*I+beta1*B+beta_1*inv(B)
    return variable(Te)
    
# Piola trasformation of the deviatoric part of the Cauchy stress
def Pi_D(u):
    F = DeformationGradient(u)
    J = Jacobian(u)
    Te = elastic_Cauchy(u)
    return variable(J*inv(F)*dev(Te)*inv(F).T)
     
# Cauchy stress tensor 
def Cauchy(u,p):
    F = DeformationGradient(u)
    I = SecondOrderIdentity(u)
    Pie_D = Pi_D(u)
    integral = Function(TS)
    integral.interpolate(Expression((("0.0", "0.0", "0.0"),("0.0", "0.0", "0.0"),("0.0", "0.0", "0.0")), degree=1))
    
    if (t > t_star):
        time_steps = list(my_range(0.000, round(t_star,3), 0.01))    
        
         # Cache Pie_D_tensor for all necessary time steps
        tensor_cache = {}
        f_in = XDMFFile("tensorPi.xdmf")
        for idx, s in enumerate(time_steps[:]):  # Skip the last time step as it's handled separately
            Pie_D_tensor = Function(TS)
            f_in.read_checkpoint(Pie_D_tensor, "Pie_D_tensor", idx)
            tensor_cache[round(s, 3)] = Pie_D_tensor
        f_in.close()  # Close the file after reading all the data

        for idx, s in enumerate(my_range(0.000, round(t_star,1), dt)):        
            D_diff = -mu_1/(mu_0*tau_1)*exp(-(t-s)/tau_1)-mu_2/(mu_0*tau_2)*exp(-(t-s)/tau_2)-mu_3/(mu_0*tau_3)*exp(-(t-s)/tau_3)-mu_4/(mu_0*tau_4)*exp(-(t-s)/tau_4)
            Pie_D_tensor = tensor_cache[round(s, 3)] 
            if (round(s,3) == 0.000) or (round(s,3) == round(t_star,1)):              
                    integral = integral + 0.5 * dt * D_diff * Pie_D_tensor 
            else:
                    integral = integral + dt * D_diff * Pie_D_tensor
                         
        for idx, s in enumerate(my_range(round(t_star,3), round(t, 3), dt)):       
            D_diff = -mu_1/(mu_0*tau_1)*exp(-(t-s)/tau_1)-mu_2/(mu_0*tau_2)*exp(-(t-s)/tau_2)-mu_3/(mu_0*tau_3)*exp(-(t-s)/tau_3)-mu_4/(mu_0*tau_4)*exp(-(t-s)/tau_4)
            if (round(s,3) != round(t,3)):
                 Pie_D_tensor = tensor_cache[round(t_star,2)]   
                 if round(s,3) == round(t_star,3):     
                         integral = integral + 0.5 * dt * D_diff * Pie_D_tensor 
                 else:
                         integral = integral + dt * D_diff * Pie_D_tensor
                         
            elif round(s,3) == round(t,3): 
                 integral = integral + 0.5 * dt * D_diff * Pie_D  
    elif (t > 0):
        time_steps = list(my_range(0.000, round(t-dt,3), 0.01))    
        
         # Cache Pie_D_tensor for all necessary time steps
        tensor_cache = {}
        f_in = XDMFFile("tensorPi.xdmf")
        for idx, s in enumerate(time_steps[:]):  # Skip the last time step as it's handled separately
            Pie_D_tensor = Function(TS)
            f_in.read_checkpoint(Pie_D_tensor, "Pie_D_tensor", idx)
            tensor_cache[round(s, 3)] = Pie_D_tensor
        f_in.close()  # Close the file after reading all the data

        for idx, s in enumerate(my_range(0.000, t, dt)):        
            D_diff = -mu_1/(mu_0*tau_1)*exp(-(t-s)/tau_1)-mu_2/(mu_0*tau_2)*exp(-(t-s)/tau_2)-mu_3/(mu_0*tau_3)*exp(-(t-s)/tau_3)-mu_4/(mu_0*tau_4)*exp(-(t-s)/tau_4)
            if (round(s,3) != round(t,3)):
                 Pie_D_tensor = tensor_cache[round(s, 3)]    
                 if round(s,3) == 0.000:     
                         integral = integral + 0.5 * dt * D_diff * Pie_D_tensor 
                 else:
                         integral = integral + dt * D_diff * Pie_D_tensor
                         
            elif round(s,3) == round(t,3): 
                 integral = integral + 0.5 * dt * D_diff * Pie_D 
  
    T = F*(Pie_D+integral)*F.T-p*I       
    
    return variable(T)

# First Piola-Kirchhoff stress tensor
def FirstPiola(u,p):
    F = DeformationGradient(u)
    I = SecondOrderIdentity(u)
    J = Jacobian(u)
    T = Cauchy(u,p)
    
    return variable(J*T*inv(F).T)

def geometry_3d(r1,r2,h1,h2):
     #Making a cylindrical geometry
     #cylinder = Cylinder('coordinate of center of the top circle', 'coordinate of center of the bottom circle', 'radius of the circle at the top', 'radius of the circle at the bottom')

     cylinder = Cylinder(Point(0, 0, h2), Point(0, 0, h1), r1, r2)
     geometry = cylinder
     # Making Mesh (30 corresponds to the mesh density)
     mesh = generate_mesh(geometry, 25)
 
     boundary_parts = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
     boundary_parts.set_all(0)
     
     bottom = AutoSubDomain(lambda x: near(x[2], h1))
     top = AutoSubDomain(lambda x: near(x[2], h2))
     
     bottom.mark(boundary_parts, 1)
     top.mark(boundary_parts, 2)

     return boundary_parts

# Create mesh and define function space ============================================
r0 = 0.0125 #m
h0 = 0.0088319 #m

facet_function = geometry_3d(r0,r0,0.0,h0)
mesh = facet_function.mesh()
gdim = mesh.geometry().dim()
dx = Measure("dx")
ds = Measure("ds", domain=mesh, subdomain_data=facet_function)
x = SpatialCoordinate(mesh)
print('Number of nodes: ',mesh.num_vertices())
print('Number of cells: ',mesh.num_cells())

#Save the mesh to an XDMF file
xdmf_file = XDMFFile("mesh.xdmf")
xdmf_file.write(mesh)

# mesh = Mesh()
# mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
# with XDMFFile("dominio2.xdmf") as infile:
   # infile.read(mesh)
   
dx = Measure("dx")
x = SpatialCoordinate(mesh)

print('Number of nodes: ',mesh.num_vertices())
print('Number of cells: ',mesh.num_cells())
   
# Limit quadrature degree
dx = dx(metadata={'quadrature_degree': 4})

tol = 1e-06

def boundary_bottom(x, on_boundary):
    if on_boundary:
        if near(x[2], 0, tol):
            return True
        else:
            return False
    else:
        return False
        
def boundary_top(x, on_boundary):
    if on_boundary:
        if near(x[2], h0, tol):
            return True
        else:
            return False
    else:
        return False

### Create function space

P2 = VectorElement("CG", mesh.ufl_cell(), 2) # displacement u
P1 = FiniteElement("CG", mesh.ufl_cell(), 1) # pressure p
TH = MixedElement([P2, P1])
V = FunctionSpace(mesh, TH)
U = FunctionSpace(mesh, P2)
TT = TensorElement("CG", mesh.ufl_cell(), 1)
TS = FunctionSpace(mesh, TT)

### Define functions for variational problems

# Incremental displacement and pressure
dup = TrialFunction(V)
(du, dp) = split(dup)

# Test functions for displacement and pressure
u_, p_ = TestFunctions(V)

# Displacement and pressure (current value)
up = Function(V)
(u, p) = split(up)

# Displacement and pressure (previous value)
up_prev = Function(V)
up_prev = interpolate(Expression(("0.0", "0.0", "0.0", "0.0"), degree=1), V)
#up_prev = Function(V, "out_solution.xml") 
(u_prev, p_prev) = split(up_prev)
u_init = project(u_prev,U)
rotation1 = Function(TS)
rotation2 = Function(TS)

# Time stepping parameters
dt = 0.01
t = 0
Tfin = 201

### Input parameters
phi_0 = 86.466
lambda0 = 0.99
t_star =  2.2
mu_inf = Constant(125.55) #Pa
mu_1 = Constant(606.76) #Pa  
mu_2 = Constant(111.28) #Pa  
mu_3 = Constant(73.07) #Pa  
mu_4 = Constant(1.0213) #Pa     
tau_1 = Constant(0.6811)
tau_2 = Constant(9.5764) 
tau_3 = Constant(82.406)
tau_4 = Constant(48.45)
mu_0 = Constant(mu_inf + mu_1 + mu_2 + mu_3 + mu_4) #Pa
 
c2 = Constant(458.84) #Pa
gamma = Constant(1/2-2*c2/mu_0)

### Boundary conditions

twist = Expression(("x[0] == 0 ? sqrt(pow(x[0],2)+pow(x[1],2))/sqrt(pre_stretch) : sqrt(pow(x[0],2)+pow(x[1],2))/sqrt(pre_stretch)*cos(atan2(x[1],x[0])+theta*x[2])-x[0]", "x[0] == 0 ? sqrt(pow(x[0],2)+pow(x[1],2))/sqrt(pre_stretch) : sqrt(pow(x[0],2)+pow(x[1],2))/sqrt(pre_stretch)*sin(atan2(x[1],x[0])+theta*x[2])-x[1]", "(pre_stretch-1)*x[2]"), theta=0.0, pre_stretch=lambda0, degree=2)  

# bc_bottom = DirichletBC(V.sub(0), u_init, boundary_bottom)
# bc_top = DirichletBC(V.sub(0), twist, boundary_top)
bc_bottom = DirichletBC(V.sub(0).sub(2), Constant(0), facet_function, 1)
bc_top = DirichletBC(V.sub(0), twist, facet_function, 2)
bcs = [bc_bottom, bc_top]

### Save in a file

displacement_file = File("u.pvd")
pressure_file = File("p.pvd")
T_file = File("T.pvd")

f_out = XDMFFile("tensorPi.xdmf") 
f_out.parameters['rewrite_function_mesh'] = False

out_solution_file = File("out_solution.xml")

parameters['krylov_solver']['nonzero_initial_guess'] = True

while t <= Tfin+tol:
     
     t = round(t,3)
     print('time: ', t)
     
     # Increase torsion
     if (t < t_star-tol):
        alpha = phi_0*(t/t_star)
     else:
        alpha = phi_0
        
     twist.theta = alpha
  
     up = Function(V)     
     up.assign(up_prev)
     (u, p) = split(up)
     J = Jacobian(u)
     P = FirstPiola(u,p)
     L1 = inner(P, Grad(u_))*dx 
     L2 = (J-1)*p_*dx
     L = L1 + L2 
     j = derivative(L, up, dup)
     problem = NonlinearVariationalProblem(L, up, bcs, J=j)
     solver = NonlinearVariationalSolver(problem)

     solver.solve()
     
     u, p = up.split()
     
     if t == 0:
        u_0 = interpolate(u, U)
        #bc_bottom = DirichletBC(V.sub(0), u_0, boundary_bottom)
        bc_bottom = DirichletBC(V.sub(0), u_0, facet_function, 1)
        bcs = [bc_bottom, bc_top]
        
     if t <= t_star:        
        f_out.write_checkpoint(project(Pi_D(u),TS), "Pie_D_tensor", t, XDMFFile.Encoding.HDF5, True) 
         
     u.rename("u", "")
     p.rename("p","")
     
     if ((t < 3) or (t >= 3 and t % 1 == 0)):     
         displacement_file << (u, t)
         pressure_file << (p, t)
         out_solution_file << up

         # Compute the normal and the torque 
         sigma = project(Cauchy(u,p),TS)    
         rotation1.interpolate(Expression((("cos(atan2(x[1],x[0])+alpha*x[2])", "sin(atan2(x[1],x[0])+alpha*x[2])", "0.0"),("-sin(atan2(x[1],x[0])+alpha*x[2])", "cos(atan2(x[1],x[0])+alpha*x[2])", "0.0"),("0.0", "0.0", "1.0")), alpha=alpha, degree=2))  
         rotation2.interpolate(Expression((("cos(atan2(x[1],x[0])+alpha*x[2])", "-sin(atan2(x[1],x[0])+alpha*x[2])", "0.0"),("sin(atan2(x[1],x[0])+alpha*x[2])", "cos(atan2(x[1],x[0])+alpha*x[2])", "0.0"),("0.0", "0.0", "1.0")), alpha=alpha, degree=2))

         sigma_cyl = rotation1*sigma*rotation2
         sigma_v = project(sigma_cyl,TS) #Put here Cauchy stress tensor in cylindrical coordinates
         sigma_v.rename("T","")
         T_file << (sigma_v, t)
     
     up_prev.assign(up)
     
     # time increment
     if t < 3:
        t += dt
     else: 
        t += 1 

f_out.close()
