---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# CompMech04-Linear Algebra Project
## Practical Linear Algebra for Finite Element Analysis

+++

In this project we will perform a linear-elastic finite element analysis (FEA) on a support structure made of 11 beams that are riveted in 7 locations to create a truss as shown in the image below. 

![Mesh image of truss](../images/mesh.png)

+++

The triangular truss shown above can be modeled using a [direct stiffness method [1]](https://en.wikipedia.org/wiki/Direct_stiffness_method), that is detailed in the [extra-FEA_material](./extra-FEA_material.ipynb) notebook. The end result of converting this structure to a FE model. Is that each joint, labeled $n~1-7$, short for _node 1-7_ can move in the x- and y-directions, but causes a force modeled with Hooke's law. Each beam labeled $el~1-11$, short for _element 1-11_, contributes to the stiffness of the structure. We have 14 equations where the sum of the components of forces = 0, represented by the equation

$\mathbf{F-Ku}=\mathbf{0}$

Where, $\mathbf{F}$ are externally applied forces, $\mathbf{u}$ are x- and y- displacements of nodes, and $\mathbf{K}$ is the stiffness matrix given in `fea_arrays.npz` as `K`, shown below

_note: the array shown is 1000x(`K`). You can use units of MPa (N/mm^2), N, and mm. The array `K` is in 1/mm_

$\mathbf{K}=EA*$

$  \left[ \begin{array}{cccccccccccccc}
 4.2 & 1.4 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 1.4 & 2.5 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 8.3 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 4.2 & -1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & -1.4 & 2.5 \\
\end{array}\right]~\frac{1}{m}$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
K;
```

In this project we are solving the problem, $\mathbf{F}=\mathbf{Ku}$, where $\mathbf{F}$ is measured in Newtons, $\mathbf{K}$ `=E*A*K` is the stiffness in N/mm, `E` is Young's modulus measured in MPa (N/mm^2), and `A` is the cross-sectional area of the beam measured in mm^2. 

There are three constraints on the motion of the joints:

i. node 1 displacement in the x-direction is 0 = `u[0]`

ii. node 1 displacement in the y-direction is 0 = `u[1]`

iii. node 7 displacement in the y-direction is 0 = `u[13]`

We can satisfy these constraints by leaving out the first, second, and last rows and columns from our linear algebra description.

+++

### 1. Calculate the condition of `K` and the condition of `K[2:13,2:13]`. 

a. What error would you expect when you solve for `u` in `K*u = F`? 

b. Why is the condition of `K` so large? __The problem is underconstrained. It describes stiffness of structure, but not the BC's. So, we end up with sumF=0 and -sumF=0__

c. What error would you expect when you solve for `u[2:13]` in `K[2:13,2:13]*u=F[2:13]`

```{code-cell} ipython3
print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))

print('expected error in x=solve(K,b) is {}'.format(10**(16-16)))
print('expected error in x=solve(K[2:13,2:13],b) is {}'.format(10**(2-16)))
```

### 2. Apply a 300-N downward force to the central top node (n 4)

a. Create the LU matrix for K[2:13,2:13]

b. Use cross-sectional area of $0.1~mm^2$ and steel and almuminum moduli, $E=200~GPa~and~E=70~GPa,$ respectively. Solve the forward and backward substitution methods for 

* $\mathbf{Ly}=\mathbf{F}\frac{1}{EA}$

* $\mathbf{Uu}=\mathbf{y}$

_your array `F` is zeros, except for `F[5]=-300`, to create a -300 N load at node 4._

c. Plug in the values for $\mathbf{u}$ into the full equation, $\mathbf{Ku}=\mathbf{F}$, to solve for the reaction forces

d. Create a plot of the undeformed and deformed structure with the displacements and forces plotted as vectors (via `quiver`). Your result for steel should match the following result from [extra-FEA_material](./extra-FEA_material.ipynb). _note: The scale factor is applied to displacements $\mathbf{u}$, not forces._

> __Note__: Look at the [extra FEA material](./extra-FEA_material). It
> has example code that you can plug in here to make these plots.
> Including background information and the source code for this plot
> below.


![Deformed structure with loads applied](../images/deformed_truss.png)

```{code-cell} ipython3
def LUNaive(A):
    '''LUNaive: naive LU decomposition
    L,U = LUNaive(A): LU decomposition without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    returns:
    ---------
    L = Lower triangular matrix
    U = Upper triangular matrix
    '''
    [m,n] = np.shape(A)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination
    U = A.astype(float)
    L = np.eye(n)

    for k in range(0,n-1):
        for i in range(k+1,n):
            if U[k,k] != 0.0:
                factor = U[i,k]/U[k,k]
                L[i,k]=factor
                U[i,:] = U[i,:] - factor*U[k,:]
    return L,U
```

```{code-cell} ipython3
def solveLU(L,U,b):
    '''solveLU: solve for x when LUx = b
    x = solveLU(L,U,b): solves for x given the lower and upper 
    triangular matrix storage
    uses forward substitution for 
    1. Ly = b
    then backward substitution for
    2. Ux = y
    
    Arguments:
    ----------
    L = Lower triangular matrix
    U = Upper triangular matrix
    b = output vector
    
    returns:
    ---------
    x = solution of LUx=b '''
    n=len(b)
    x=np.zeros(n)
    y=np.zeros(n)
        
    # forward substitution
    for k in range(0,n):
        y[k] = b[k] - L[k,0:k]@y[0:k]
    # backward substitution
    for k in range(n-1,-1,-1):
        x[k] = (y[k] - U[k,k+1:n]@x[k+1:n])/U[k,k]
    return x
```

```{code-cell} ipython3
L, U = LUNaive(K[2:13,2:13])
# print('L\n', L, '\n\n', 'U\n', U, '\n')
A = 0.1 # mm^2
E_st = 200e3 # MPa - steel
E_al = 70e3 # MPa - aluminum
F = np.zeros(11)
F[5] = -300 # 300N load at node 4
F_EA_st = F/(E_st*A)
F_EA_al = F/(E_al*A)

ufree_st = solveLU(L,U,F_EA_st)
ufree_al = solveLU(L,U,F_EA_al)

u_st = np.zeros(14)
u_st[2:13] = ufree_st
u_al = np.zeros(14)
u_al[2:13] = ufree_al

print('Steel displacements (mm):\n', u_st, '\n')
print('Aluminum displacements (mm):\n', u_al, '\n')


F_r_st = E_st*A*K@u_st # steel reaction forces
F_r_al = E_al*A*K@u_al # aluminum reaction forces

print('Reaction forces (N) steel:\n', F_r_st, '\n\nReaction forces (N) aluminum:\n', F_r_al)
```

```{code-cell} ipython3
scale = 5
# r is initial geometry
# each beam is 300 mm
r0=np.array([0,0,150,150*2**0.5,300,0,450,150*2**0.5,600,0,750,150*2**0.5,900,0]) # all in mm
r_st=r0+u_st*scale
r_al=r0+u_al*scale

ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

plt.figure()
plt.plot(r0[ix],r0[iy],'s-',color='k')
plt.quiver(r0[ix],r0[iy],u_st[ix],u_st[iy],color=(0,1,1,1),label='displacements')
plt.plot(r_st[ix],r_st[iy],'o-',color='g')

plt.quiver(r0[ix],r0[iy],F_r_st[ix],F_r_st[iy],color=(1,0,0,1),label='applied forces')
plt.axis([-100,1100,-200,400])
plt.legend(loc='center left', bbox_to_anchor=(1,0.5));
plt.title('original (blk) and deformed (gr) steel structure\nscale = {}x'.format(scale));
```

```{code-cell} ipython3
plt.figure()
plt.plot(r0[ix],r0[iy],'s-',color='k')
plt.quiver(r0[ix],r0[iy],u_al[ix],u_al[iy],color=(0,1,1,1),label='displacements')
plt.plot(r_al[ix],r_al[iy],'o-',color='g')
plt.quiver(r0[ix],r0[iy],F_r_al[ix],F_r_al[iy],color=(1,0,0,1),label='applied forces')
plt.axis([-200,1200,-400,400])
plt.legend(loc='center left', bbox_to_anchor=(1,0.5));
plt.title('original (blk) and deformed (gr) aluminum structure\nscale = {}x'.format(scale));
```

### 3. Determine cross-sectional area

a. Using aluminum, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

b. Using steel, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

c. What are the weights of the aluminum and steel trusses with the
chosen cross-sectional areas?

```{code-cell} ipython3
L, U = LUNaive(K[2:13,2:13])
# A_al = 9.28 # mm^2
# A_st = 3.25 # mm^2
# # if not abs value

A_al = 23.04 # mm^2
A_st = 8.07 # mm^2

E_st = 200e3 # MPa - steel
E_al = 70e3 # MPa - aluminum
F = np.zeros(11)
F[5] = -300 # 300N load at node 4
F_EA_st = F/(E_st*A_st)
F_EA_al = F/(E_al*A_al)

ufree_st = solveLU(L,U,F_EA_st)
ufree_al = solveLU(L,U,F_EA_al)
print('Al\n', ufree_al, '\n\nSt\n', ufree_st)

print(all(np.abs(i)<0.2 for i in ufree_al))
print(all(np.abs(i)<0.2 for i in ufree_st))
```

From a guess-and-check method, to keep the magnitude of the y-deflections less than 0.2 mm, the min cross-sectional area is 23.04 mm^2 for aluminum and 8.07 mm^2 for steel.

```{code-cell} ipython3
p_al = 0.0027 # g/mm^3
p_st = 0.00785 # g/mm^3
vol_al = 11*300*A_al
vol_st = 11*300*A_st
weight_al = p_al*vol_al
weight_st = p_st*vol_st

print('Aluminum truss weight: {:.2f}g \nSteel truss weight: {:0.2f}g'.format(weight_al, weight_st))
```

Using a density of 2.7 g/cm^3 for aluminum and a density of 7.85 g/cm^3 for steel.

+++

## References

1. <https://en.wikipedia.org/wiki/Direct_stiffness_method>
