#!/usr/bin/python

"""
Galileo is a code that performs Monte Carlo simulations on a 3-dimensional
dipole lattice. 
A detailed reference on the method can be found here:
    C. Motta, F. El Mellouhi, S. Sanvito "Exploring the cation dynamics in lead-bromide hybrid perovskites" , Phys. Rev. B 93, 235412 (2016)
    http://journals.aps.org/prb/abstract/10.1103/PhysRevB.93.235412
"""

import numpy as np
import math
import os.path
import timeit


#----------------------------------------------------------------------#
# Input parameters
#----------------------------------------------------------------------#
NX = 4                           # lattice size along (1,0,0)
NY = NX                           # lattice size along (0,1,0)
NZ = NX                           # lattice size along (0,0,1)
nearest_neighbours = 3            # dipole-dipole interaction cutoff
dip_int_str = 1.                  # dipole-dipole interaction strength, in units of k_B
T = 300                           # simulation temperature (K)
K = 3.5                           # dipole-strain interaction
E_field = [0,0,0]                 # external electic field
out_mc_steps = 30                 # steps for the outer MC cycle
init_config = "polarized_100"     # initial lattice configuration
                                  # (options:polarized_111/polarized_100/random/from_file)
attempt_time = 0.95               # dipole rotational attempt time (ps)
selection_rule =  -1              # =-1 for completely randomly oriented new dipoles (default)
                                  # =0 for new random dipoles in same hemisphere
                                  # = x in (-1,1) for general domain...look the code
calc_pol = True                   # to calculate the polarization
calc_corr = None                  # to calculate the correlation function
#
print_lattice = None              # outputs the dipole array every <print_clock> MC iterations 
print_clock = 1000                # MC iteration interval
#----------------------------------------------------------------------#
# End input parameters
#----------------------------------------------------------------------#

#----------------------------------------------------------------------#
# Development input parameters
#----------------------------------------------------------------------#
pi = math.pi
do_fourier = True
q_list = [[pi,0.,0.]]
#----------------------------------------------------------------------#

# Notes:
# Energies are in units of k_B*T, T=300K
# the dipole-dipole interaction for methylammonia is 28 meV
# and for formamidinium is 0.3 meV, so in units of (k_B*300):
#                    MA: dip_int_str = 1
#                    FA: dip_int_str = 0.012
# k_B = 0.086173324 meV/K
# at T=300K ==> k_B*T = 25.8519972 meV
# Attempt frequency ~= 35cm-1=1.04927 THz => attempt_time = 0.95 (ps) 
# or, considering pure mol. rotational freq:
#                     MA : attempt_time = 0.238 (ps)
#                     FA : attempt_time = 0.179 (ps)
# Rotational barrier: MA : 0.09 eV ==> K = 3.5
#                     FA : 0.14 eV ==> K = 5.4

#----------------------------------------------------------------------#
# Utility: check if file exists in pwd
#----------------------------------------------------------------------#
def file_exists(filename):
    if( os.path.exists("./"+filename)):
        print "*** Warning !"
        print "*** File %s already present in pwd, exiting..." % (filename)
        quit()
    return()


#----------------------------------------------------------------------#
# Generates a random 3D unit vector with a uniform spherical 
# distribution by Marsaglia's method (1972) 
# http://mathworld.wolfram.com/SpherePointPicking.html
#----------------------------------------------------------------------#
def random_three_vector():
    norm = 1.1
    while norm >= 1.:
        x1 = np.random.uniform(-1,1)
        x2 = np.random.uniform(-1,1)
        norm = x1*x1+x2*x2
    x = 2*x1*math.sqrt(1 - x1**2 - x2**2)
    y = 2*x2*math.sqrt(1 - x1**2 - x2**2)
    z = 1-2*(x1**2 + x2**2)
    uv = []
    uv.extend([x,y,z])
    return uv


#----------------------------------------------------------------------#
# Generates a random 3D unit vector with a uniform spherical 
# distribution - OLD 
#----------------------------------------------------------------------#
def random_three_vector_old():
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)
    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    uv = []
    uv.extend([x,y,z])
    return uv


#----------------------------------------------------------------------#
# Dot product between two vectors v1 and v2
#----------------------------------------------------------------------#
def dotp(v1,v2):
    length = len(v1)
    output = 0.
    for i in range(length):
        output += v1[i]*v2[i]
    return output

#----------------------------------------------------------------------#
# Initialize lattice with random positions
#----------------------------------------------------------------------#
def initialise_lattice(lattice,mode):
    if mode == "random" or mode == "Random":
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    lattice[i][j][k][:] = random_three_vector()
    elif mode == "polarized_111" or mode == "Polarized_111":
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    lattice[i][j][k][:] = \
                    np.array([1./math.sqrt(3),1./math.sqrt(3),1./math.sqrt(3)])
    elif mode == "polarized_100" or mode == "Polarized_100":
        for i in range(NX):
            for j in range(NY):
                for k in range(NZ):
                    lattice[i][j][k][:] = \
                    np.array([1.,0.,0.])
    elif mode == "from_file":
        lf = open("lattice_input.dat", 'r') 
        for line in lf.readlines():
            field = line.split()
            i = int(field[0])
            j = int(field[1])
            k = int(field[2])
            vx = float(field[3])
            vy = float(field[4])
            vz = float(field[5])
            lattice[i][j][k]=np.array([vx,vy,vz])
        lf.close()
    else: 
        print "Initial lattice configuration undefined:", \
              " select 'random' or 'polarized'"
        quit()
    return()



#----------------------------------------------------------------------#
# Check periodic boundary conditions
#----------------------------------------------------------------------#
def bcx(i):
    if   i > NX-1:
        return i - NX
    elif i < 0:
        return NX + i
    else:
        return i
def bcy(i):
    if   i > NY-1:
        return i - NY
    elif i < 0:
        return NY + i
    else:
        return i
def bcz(i):
    if   i > NZ-1:
        return i - NZ
    elif i < 0:
        return NZ + i
    else:
        return i


#----------------------------------------------------------------------#
# Generates the matrice of the index mapping from 3D to 1D 
#----------------------------------------------------------------------#
def index_mapping_3d_1d(NX,NY,NZ):
    indices = numpy.zeros( (NX, NY, NZ) )
    for i1 in range(NX):
        for i2 in range(NY):
            for i3 in range(NZ):
                indices[i][j][k]= i1*NX*NY + i2*NY + i3
    return indices


#----------------------------------------------------------------------#
# Calculate the site energy 
#----------------------------------------------------------------------#
def site_energy(i_site,j_site,k_site,new_dipole):

    old_dipole = lattice[i_site][j_site][k_site]
    nn = nearest_neighbours
    dE = 0.

    for i in range(i_site -nn , i_site + nn + 1):
        for j in range(j_site -nn , j_site + nn + 1):
            for k in range(k_site -nn , k_site + nn + 1):

                if i==i_site and j==j_site and k==k_site: continue  
                
                distance = math.sqrt( (i_site-i)**2 + (j_site-j)**2 + \
                                      (k_site-k)**2 ) 
                
                if distance > nn : continue  # cutoff!
                
                #uv =  [(i_site-i)/distance, (j_site-j)/distance,(k_site-k)/distance]
                uv = np.array( [i_site-i, j_site-j, k_site-k] ) / distance

                # print bcx(i), bcy(j), bcz(k)    # check correct
                                                  # nearest-neighb.
        
                test_dipole = lattice[bcx(i)][bcy(j)][bcz(k)]
                
                # dipole-dipole interaction
                dE += +dip_int_str * ( np.dot(new_dipole,test_dipole) \
                      -3 * np.dot(uv,new_dipole) * np.dot(uv,test_dipole) ) /distance**3 \
                      -dip_int_str * ( np.dot(old_dipole,test_dipole) \
                      -3 * np.dot(uv,old_dipole) * np.dot(uv,test_dipole) ) / distance**3 

# dipole-strain interaction
    dE += +K * np.fabs( np.dot(new_dipole,[1.,0.,0.])) \
          -K * np.fabs( np.dot(old_dipole,[1.,0.,0.]))
    dE += +K * np.fabs( np.dot(new_dipole,[0.,1.,0.])) \
          -K * np.fabs( np.dot(old_dipole,[0.,1.,0.]))
    dE += +K * np.fabs( np.dot(new_dipole,[0.,0.,1.])) \
          -K * np.fabs( np.dot(old_dipole,[0.,0.,1.]))

    # dipole-electric_field interaction
    dE += -np.dot(E_field,new_dipole) +np.dot(E_field,old_dipole)

    return dE


#----------------------------------------------------------------------#
# Single Monte Carlo move
#----------------------------------------------------------------------#
def MC_move(pick_x,pick_y,pick_z):

    global ACCEPT
    global REJECT

    # take a random dipole
    #ran_dip = random_three_vector()

    # Take a random dipole that is in the same
    # hemisphere described by the pick vector
    dot_prod = -2
    while dot_prod < selection_rule:
        ran_dip = random_three_vector()
        dot_prod = np.dot(lattice[pick_x][pick_y][pick_z],ran_dip) 

    dE = site_energy(pick_x,pick_y,pick_z,ran_dip) 
    
    if dE <= 0. or math.exp(-dE * beta) > np.random.random():
        lattice[pick_x][pick_y][pick_z][:] = ran_dip[:]
        ACCEPT += 1
    else:
        REJECT += 1


#----------------------------------------------------------------------#
# Calculate the total polarization of the lattice
#----------------------------------------------------------------------#
def polarization():
    tot_pol = np.array([0.,0.,0.])
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                tot_pol += lattice[i][j][k][:] 
    return np.linalg.norm(tot_pol) / NX / NY / NZ


#----------------------------------------------------------------------#
# Calculate the dipole-dipole correlation function on spheres
#----------------------------------------------------------------------#
def dipole_correlation(i_site,j_site,k_site,R_list):
    # R_list has to be something like [1,2,3,4,5] or better
    # [1, math.sqrt(2), sqrt(3), 2, 2*math.sqrt(2), 2*math.sqrt(3), 3]
    tolerance = 0.01
    corr = []
    for R in R_list:
        dip_dip = 0.
        count_dipole_in_sphere = 0
        for i in range(i_site - int(R)-1 , i_site + int(R)+1 + 1):
            for j in range(j_site - int(R)-1 , j_site + int(R)+1 + 1):
                for k in range(k_site - int(R)-1 , k_site + int(R)+1 + 1):
    
                    if i==i_site and j==j_site and k==k_site: continue  # no self-interaction 
                    
                    distance = math.sqrt( (i_site-i)**2 + (j_site-j)**2 + \
                                          (k_site-k)**2 ) 
                    
                    if distance > R+tolerance : continue  # cutoff!
                    if distance < R-tolerance : continue  # cutoff!

                    test_dipole = lattice[bcx(i)][bcy(j)][bcz(k)]
                    ref_dipole = lattice[i_site][j_site][k_site]
                    dip_dip += np.dot(ref_dipole,test_dipole)
                    count_dipole_in_sphere += 1
                    # ( np.dot(new_dipole,test_dipole) \
        dip_dip = dip_dip / count_dipole_in_sphere
        corr.append(dip_dip)
    return corr
                    

#----------------------------------------------------------------------#
# Calculate the dipole-dipole correlation function between two sites
#----------------------------------------------------------------------#
def sites_correlation(i_site,j_site,k_site,site_list):
    # i_site,j_site,k_site: coordinates of the reference site
    # site_list: list of the probe sites, e.g. [[s1x,s1y,s1z],[s2x,s2y,s2z],...] 
    corr = []
    for probe in site_list:
        test_dipole = lattice[bcx(probe[0])][bcy(probe[1])][bcz(probe[2])]
        ref_dipole = lattice[i_site][j_site][k_site]
        corr.append(np.dot(ref_dipole,test_dipole))
    return corr


#----------------------------------------------------------------------#
# Output the full 3D dipole field
#----------------------------------------------------------------------#
def output_field(tag):
    ofile = open("Lattice_"+tag+".dat", 'w')
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
            	print >> ofile, i, j, k, lattice[i][j][k][0], lattice[i][j][k][1], \
                       lattice[i][j][k][2]
    ofile.close()



print "                              (``')"
print "                             / `''/"
print "          G                 /    /"
print "          A              O\/    /"
print "          L              \,    /"
print "          I              (    /"
print "          L             /x`''7/"
print "          E            (x   // \ "
print "          O           / `''7'`\_\ "
print "                     /    /   /__\ "
print "                    (    /   `|~~|`"
print "                     `'''     |==|"
print "      by:                     |  |"
print "      Carlo Motta             |==|"
print "      Feb. 2015               |  |"
print "                              |__|"
print "                            /`____`\ "
print "                  ,-------'`~~~~~~~~`'-------,"
print "                 `~~~~~~~~~~~~~~~~~~~~~~~~~~~~`  "


# Open a log file
logname = 'montecarlo_L_%s_T_%s_D_%s_K_%s.log' % (NX,T,dip_int_str,K)
file_exists(logname)
log = open(logname,"w")
print >>log, "Montecarlo calculation starting, parameters:"
log.write("%4s %5.3f %25s %10.5f \n" % ("T = ",T,"Dipole-Dipole strength=",dip_int_str))
print >>log, "N                  = ", NX
print >>log, "nearest_neighbours = ", nearest_neighbours
print >>log, "K                  = ", K
print >>log, "E_field            = ", E_field
print >>log, "out_mc_steps       = ", out_mc_steps
print >>log, "init_config        = ", init_config
print >>log, "attempt_time       = ", attempt_time
print >>log, "selection_rule     = ", selection_rule

# Initialise a 3D lattice
lattice = np.zeros( (NX, NY, NZ, 3) )
print >>log, "# lattice shape = ", lattice.shape
initialise_lattice(lattice,init_config)

# Initialise some counters
ACCEPT = 0
REJECT = 0
beta = 1. / ( T / 300. )
correlation_spheres = [1,math.sqrt(2),math.sqrt(3),2,3]
average_correlation = np.zeros(len(correlation_spheres))
average_polarization = 0.

# Open output polarization file
if calc_pol:
    pol_file_name = 'polarization_L_%s_T_%s_D_%s_K_%s.dat' % (NX,T,dip_int_str,K)
    #pol_file_name = 'polarization_MA.dat' 
    file_exists(pol_file_name)
    pol_file = open(pol_file_name,"w")
    #print >> pol_file, "# MC_Step   Polarization"
    print >> pol_file, "# Time(ps)   Polarization"

# Open output correlation file
if calc_corr:
    corr_file_name = 'correlation_L_%s_T_%s_D_%s_K_%s.dat' % (NX,T,dip_int_str,K)
    file_exists(corr_file_name)
    corr_file = open(corr_file_name,"w")

start_time = timeit.default_timer()

# Outer MC cycle
for mc_counter in range(out_mc_steps):
    #print "t = ", mc_counter
    
    #--- output the polarization (comment here if output at every step (below)
    #if calc_pol:
    #    temp_pol = polarization()
    #    print >>pol_file,  mc_counter * attempt_time, temp_pol
    #    average_polarization += temp_pol
    #---

    # correlation by spheres
    if calc_corr:
        temp = dipole_correlation(NX/2,NY/2,NZ/2,correlation_spheres)
        for el in temp:
            corr_file.write(str(el)+ ' ' )
        #corr_file.write('\n')

        for ii in range(len(correlation_spheres)):
            average_correlation[ii] += temp[ii]/out_mc_steps

    # Inner MC cycle: sweeps!
    ic = 0.
    for pick_i in range(NX):
        for pick_j in range(NY):
            for pick_k in range(NZ):

                #--- output the polarization
                if calc_pol and (ic % 100 == 0):
                    temp_pol = polarization()
                    print >>pol_file,  mc_counter * attempt_time + ic/(NX*NY*NZ), temp_pol
                    average_polarization += temp_pol 
                #---

                pick_x = np.random.randint(0,NX)
                pick_y = np.random.randint(0,NY)
                pick_z = np.random.randint(0,NZ)
                MC_move(pick_x,pick_y,pick_z)

                #MC_move(pick_i,pick_j,pick_k)
                ic += 1

    #print ".",
    if print_lattice  and (mc_counter % print_clock == 0) :
        output_field(str(mc_counter))

stop_time = timeit.default_timer()

#print >>log, "Accepted moves = ", ACCEPT, "Rejected moves = ", REJECT," Total moves = ",ACCEPT+REJECT
print >>log, "Acceptance/Rejection ratio = ", float(ACCEPT)/float(REJECT)," Total moves = ",ACCEPT+REJECT
if calc_corr:
    log.write("Correlation spheres = ")
    for el in correlation_spheres:
        log.write(str(el)+ ' ' )
    log.write('\n')
    log.write("Average correlation per sphere = ")
    for el in average_correlation:
        log.write(str(el)+ ' ' )
    log.write('\n')

if calc_pol:
    print >>log, "Average polarization =", average_polarization / out_mc_steps

print >>log, "Run time = ", stop_time - start_time, " s"

log.close()
if calc_pol:    pol_file.close()
if calc_corr:    corr_file.close()



#----------------------------------------------------------------------#

