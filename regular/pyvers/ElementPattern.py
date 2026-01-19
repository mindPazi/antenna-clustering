import numpy as np

def ElementPattern(P, ELE, AZI, load_file, file_name):
    """
    Version: 1.0;
    Developed by Milan RC. delivered on 06/30/2017.
    
    This function generates or loads the single element radiation pattern.
    
    INPUT:
    P: scalar. Set P=0 to achieve an isotropic element pattern, set P=1 for
       cosine element pattern, set P=3 for loading antenna element from HFSS;
    ELE: matrix elevation angle in deg [N AZI x N ELE];
    AZI: matrix azimuth angle in deg [N AZI x N ELE].
    load_file: set load_file =1 to load antenna element RPE from HFSS,
       set load_file =0 to generate isotropic pattern
    file_name: name of the file with antenna element RPE from HFSS
    
    OUTPUT:
    Fel: matrix [N AZI x N ELE] with radiation pattern values.
    """
    
    if load_file:
        with open(file_name, 'r') as fid:
            tempor = []
            cont_limit = 1
            angsp = 361
            while cont_limit <= angsp*angsp + 2:
                tempor.append(fid.readline().strip())
                cont_limit = cont_limit + 1
        
        RPE_elem = []
        for ij in range(1, len(tempor)):
            tem = [float(x) for x in tempor[ij].split()]
            if len(tem) >= 3:
                RPE_elem.append(tem[2])
        
        RPE = np.array(RPE_elem).reshape(angsp, angsp)
        RPE = RPE.T
        Fel = 10.0**(RPE/20.0)
    else:
        Fel = ((np.cos(np.radians(ELE*0.9)) * np.cos(np.radians(AZI*0.9)))**P)
    
    return Fel
