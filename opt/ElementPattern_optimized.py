import numpy as np

def ElementPattern(P, ELE, AZI, load_file, file_name):
    """
    Version: 1.0;
    Developed by Milan RC. delivered on 06/30/2017.
    Optimized version with more efficient file I/O
    
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
        # OPTIMIZATION: Read entire file at once instead of line-by-line
        with open(file_name, 'r') as fid:
            lines = fid.readlines()
        
        angsp = 361
        
        # OPTIMIZATION: Use list comprehension with direct indexing
        # Skip first two lines (headers) and process data lines
        RPE_elem = []
        for line in lines[1:]:  # Skip first line (header)
            parts = line.strip().split()
            if len(parts) >= 3:
                RPE_elem.append(float(parts[2]))
            if len(RPE_elem) >= angsp * angsp:
                break
        
        # Convert to array and reshape in one operation
        RPE = np.array(RPE_elem[:angsp * angsp]).reshape(angsp, angsp)
        RPE = RPE.T
        Fel = 10.0**(RPE/20.0)
    else:
        # OPTIMIZATION: Element-wise power operation is already vectorized
        # Minor improvement: combine multiplications
        Fel = ((np.cos(np.radians(ELE * 0.9)) * np.cos(np.radians(AZI * 0.9)))**P)
    
    return Fel
