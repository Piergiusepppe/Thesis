import nengo
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from nengo.utils.filter_design import cont2discrete


model = nengo.Network(label="Cerebellum")
with model:


    # parameters of motor input
    freq = 2  # frequency limit
    rms = 0.30  # amplitude of input (set to keep within [-1, 1])

    # simulation parameters
    dt = 0.001  # simulation timestep
    sim_t =100  # length of simulation
    seed = 0  # fixed for deterministic results
    
    # create the input signal (it would be trajectories later)
    MotorCommand = nengo.Node(
            output=nengo.processes.WhiteSignal(
                high=freq, period=sim_t, y0=0, seed=seed
            )
        )      
    SensoryFeedback = nengo.Node(
            output=nengo.processes.WhiteSignal(
                high=freq, period=sim_t, rms=3*rms, y0=0, seed=seed
            )
        )
        
    # Create inferior olive        
    IO = nengo.Network(label="Inferior Olive")
    with IO:
        # Create 3 ensembles each containing 100 leaky integrate-and-fire neurons
        A = nengo.Ensemble(100, dimensions=1)
        B = nengo.Ensemble(100, dimensions=1)
        DiffGroup = nengo.Ensemble(100, dimensions=1)
    

        # Connect the input nodes to the appropriate ensembles
        nengo.Connection(MotorCommand, A)
        nengo.Connection(SensoryFeedback, B)
        
        # Connect difference MC-SF
        nengo.Connection(A, DiffGroup)
        nengo.Connection(B, DiffGroup, transform = -1)   
        
    # Create Granular Layer (RNN)
    # The GrC represents the state of the integrator and the connections
    # define the dynamics of the integrator
    
    GcL = nengo.Network(label="Granular Layer")
    with GcL:
        GrC = nengo.Ensemble(n_neurons=200, dimensions=2)
        # Connect control value (Motor Command) and real value (Feedback)
        nengo.Connection(SensoryFeedback,GrC[0])
        nengo.Connection(MotorCommand,GrC[1])
        # Create recurrent connection with using a higher-dimensional mapping
        nengo.Connection(GrC, GrC[0],
                     function=lambda x: x[0] * x[1]) 
                     

    # Purkinje Layer 
    PL = nengo.Network(label="Purkinje Cells Layer")
    with PL:
        PcArray = nengo.Ensemble(n_neurons=2, dimensions=3)
        CFconnection = nengo.Connection(DiffGroup,PcArray[0])
        MFconnection = nengo.Connection(GrC,PcArray[1:])