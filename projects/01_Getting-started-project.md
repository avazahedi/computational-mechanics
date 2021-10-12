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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
K = -(74-85)/(2*(85-65))
print(K)
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def emp_const(T1, T2, Ta, t):
    K = -(T2-T1)/(t*(T1-Ta))
    return K

print(emp_const(85, 74, 65, 2))
```

```{code-cell} ipython3

```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
import matplotlib.pyplot as plt

def temp(N):
    T_num = np.zeros(N)
    T_num[0] = 85
    K = emp_const(85,74,65,2)
    t = np.linspace(0,30,N)
    Ta = 65
    T_analytical = Ta + (T_num[0]-Ta)*np.exp(-0.275*t)
    for i in range(1,N):
        T_num[i] = T_num[i-1] - K*(T_num[i-1]-Ta)*(np.diff(t)[i-1])
    return T_analytical, T_num, t
    
T_analytical, T_num, t = temp(10)
plt.plot(t,T_analytical,'-',label='analytical')
plt.plot(t,T_num,'o-',label='numerical')
plt.legend()
plt.xlabel('time (hours)')
plt.ylabel('Temperature (deg F)')
```

```{code-cell} ipython3
T_analytical, T_num, t = temp(20)
plt.plot(t,T_analytical,'-',label='analytical')
plt.plot(t,T_num,'o-',label='numerical')
plt.legend()
plt.xlabel('time (hours)')
plt.ylabel('Temperature (deg F)')
```

```{code-cell} ipython3
T_analytical, T_num, t = temp(30)
plt.plot(t,T_analytical,'-',label='analytical')
plt.plot(t,T_num,'o-',label='numerical')
plt.legend()
plt.xlabel('time (hours)')
plt.ylabel('Temperature (deg F)')
```

a. We can see that the Euler integration converges as the time step is decreased.

+++

b. As t-->infinity, the final temperature is 65 degF.

```{code-cell} ipython3
Ta = 65
T0 = 85
time_of_death = np.log((98.6-Ta)/(T0-Ta)) / -0.275
TOD = 11+np.round(time_of_death,2)
mins_decimal = np.round(TOD-np.round(TOD), 2)
mins = int(np.round(mins_decimal*60, 0))
print('c. Time of death: ' + str(int(np.round(TOD,0))) + ':0' + str(mins) + 'am')
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
# 8am onwards, Ta = 5(t+3)+55
times = np.array([-5,-4,-3,-2,-1,0,1,2])
temps = [50,51,55,60,65,70,75,80]

def current_temp(t):
    if t>=-5 and t<-4:
        Ta = 50
    elif t>=-4 and t<-3:
        Ta = 51
    elif t>=-3 and t<=2:
        Ta = 5*(t+3)+55
    else:
        Ta = 65
    return Ta

day_times = np.linspace(-5,2)
Ta_values = np.array([current_temp(t) for t in day_times])

plt.plot(day_times,Ta_values,'-', label='Ambient Temp')
plt.legend()
plt.xlabel('time (hours)')
plt.ylabel('Temperature (deg F)')
```

I would say this looks correct with the data given.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Ta = 5*(t+3)+55 #for t between -3 and 2

def temp_new(N):
    T_num = np.zeros(N)
    T_num[0] = 85
    K = emp_const(85,74,65,2)
    t = np.linspace(0,4,N)
    Ta = 65
    Ta_values = np.array([current_temp(time1) for time1 in np.linspace(0,4,N)])
    T_analytical = Ta + (T_num[0]-Ta)*np.exp(-0.275*t)
    for i in range(1,N):
        T_num[i] = T_num[i-1] - K*(T_num[i-1]-Ta_values[i-1])*(np.diff(t)[i-1])
    return T_analytical, T_num, t
    
T_analytical, T_num, t = temp_new(15)
plt.plot(t,T_analytical,'-',label='analytical')
plt.plot(t,T_num,'o-',label='numerical')
plt.legend()
plt.xlabel('time (hours)')
plt.ylabel('Temperature (deg F)')
```

The new Euler approximation has a different shape than the analytical solution. This makes sense because the analytical solution considers assumptions such as the ambient temperature being constant, unlike this new Euler approximation.

Using the changing ambient temperatures and my knowledge that the time of death is sometime between 9 and 10am, I used the analytical solution and ambient temperature = 60 degF to recalculate the time of death as 9:25am.

```{code-cell} ipython3

```
