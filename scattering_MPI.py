import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
def factorsum(MDsnap,period,ScatteringF,Qpoints,cell):
    natom=len(MDsnap);
    sumvalue=0.0j;
    comm = MPI.COMM_WORLD;
    rank = comm.Get_rank();
    size = comm.Get_size();
    for i in range(rank,natom,size):
        reciprocal=np.array([2*np.pi,2*np.pi,2*np.pi])/(period/cell);
        value=np.dot(MDsnap[i],Qpoints*reciprocal);
        sumvalue=sumvalue+np.exp(-1*1j*value)*ScatteringF[i];
    divisor=0.0;
    for i in range(natom):
        divisor=divisor+np.absolute(ScatteringF[i])**2;
    sumvalue=comm.allreduce(sumvalue,op=MPI.SUM);
    return np.absolute(sumvalue)**2/divisor;
Ncell=20;
MDsector=5*Ncell**3+9;
totalatoms=5*Ncell**3;
comm = MPI.COMM_WORLD;
rank = comm.Get_rank();
if rank==0:
    print('Loading Data.....on rank {0:d}'.format(rank));
    f=open("./EE_600_Z/dump.xyz",'r');
    lines=f.readlines();
    f.close();
    totaltime=int(len(lines)/(MDsector));
    MDtraject=np.zeros((totaltime,totalatoms,3));
    Period=np.zeros((totaltime,3));
    Atomic_Scattering=np.loadtxt("ATOMIC_Q.txt",dtype=complex);
    NumberQ=len(Atomic_Scattering);
    ScatteringF=np.zeros((NumberQ,totalatoms),dtype=complex);
    for i in range(totaltime):
        for k in range(3):
            line=lines[i*MDsector+5+k];
            line=line.split();
            Period[i][k]=float(line[1])-float(line[0]);
        for k in range(5*Ncell**3):
            line=lines[i*MDsector+9+k].split();
            for t in range(3):
                MDtraject[i][k][t]=float(line[t]);
    for j in range(NumberQ):
        for i in range(Ncell**3):
            ScatteringF[j][i]=Atomic_Scattering[j][3];
        for i in range(Ncell**3,2*Ncell**3):
            ScatteringF[j][i]=Atomic_Scattering[j][4];
        for i in range(3*Ncell**3,5*Ncell**3):
            ScatteringF[j][i]=Atomic_Scattering[j][5];
    print('Finished Loading Data');
    print('Reset the cross boundary atoms');
    f=open("traject_check",'w')
    for i in range(1,totaltime):
        for j in range(totalatoms):
            MDtraject[i][j]=MDtraject[i][j]-MDtraject[i-1][j]-np.round((MDtraject[i][j]-MDtraject[i-1][j])/Period[i])*Period[i]+MDtraject[i-1][j];
            f.write("{0:20.10f} {1:20.10f} {2:20.10f}\n".format(MDtraject[i][j][0],MDtraject[i][j][1],MDtraject[i][j][2]));
    f.close();
if rank!=0:
    totaltime=0;
    NumberQ=0;
totaltime=comm.bcast(totaltime,root=0);
NumberQ=comm.bcast(NumberQ,root=0);
if rank!=0: # define variables for other rank
    MDtraject=np.zeros((totaltime,totalatoms,3));
    ScatteringF=np.zeros((NumberQ,totalatoms),dtype=complex);
    Period=np.zeros((totaltime,3));
    Atomic_Scattering=np.zeros((NumberQ,6),dtype=complex);
comm.Bcast([MDtraject,MPI.DOUBLE],root=0);
comm.Bcast([ScatteringF,MPI.COMPLEX],root=0);
comm.Bcast([Period,MPI.DOUBLE],root=0);
comm.Bcast([Atomic_Scattering,MPI.COMPLEX],root=0);
for qind in range(NumberQ):
    SQintensity=[];
    Qpoints=np.real(Atomic_Scattering[qind][0:3]);
    for i in range(totaltime):
        inte=factorsum(MDtraject[i],Period[i],ScatteringF[qind],Qpoints,Ncell);
        SQintensity.append(np.absolute(inte));
    if rank==0:
        f=open("SQ_time{0:d}".format(qind),'w');
        for i in range(totaltime):
            f.write("{0:10.7f} {1:20.10f}\n".format(i,SQintensity[i]))
        f.close();
