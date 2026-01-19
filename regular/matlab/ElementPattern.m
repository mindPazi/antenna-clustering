function Fel = ElementPattern(P,ELE,AZI,load_file,file_name)
% Version: 1.0;
% Developed by Milan RC. delivered on 06/30/2017.

% This function generates or loads the single element radiation pattern.

%%% INPUT:
% P: scalar. Set P=0 to achieve an isotropic element pattern, set P=1 for
%   cosine element pattern, set P=3 for loading antenna element from HFSS;
% ELE: matrix elevation angle in deg [N AZI x N ELE];
% AZI: matrix azimuth angle in deg [N AZI x N ELE].
% load_file: set load_file =1 to load antenna element RPE from HFSS,
%   set load_file =0 to generate isotropic pattern
% file_name: name of the file with antenna element RPE from HFSS
%
%%% OUTPUT:
% Fel: matrix [N AZI x N ELE] with radiation pattern values.
%

if load_file
    fid = fopen(file_name,'r+');
    cont_limit = 1;
    angsp=361;
    while cont_limit<=angsp*angsp+2
        tempor{cont_limit} = fgetl(fid);
        cont_limit=cont_limit+1;
    end
    fclose(fid);

    for ij=2:size(tempor,2)
        tem=str2num(tempor{ij});
        RPE_elem(ij-1)=tem(3);
    end
    RPE=reshape(RPE_elem,angsp,angsp);
    RPE=RPE';
    Fel= 10.^(RPE./20);
else
    Fel=((cosd(ELE*0.9).*cosd(AZI*0.9)).^P);
end

end
