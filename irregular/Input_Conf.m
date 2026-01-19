% INPUT PARAMETERS File

%%% SYSTEM PARAMETERS
SYS.f                      =29.5e9;                    % [GHz]
SYS.azi0                   =0;                         % [deg] azimuth steering angle
SYS.ele0                   =30;                        % [deg] elevation steering angle
SYS.dele                   =0.5;                       % angle resolution [deg]
SYS.dazi                   =0.5;                       % angle resolution [deg]

%%% LATTICE ANTENNA ARRAY PARAMETERS
LAT.Nz                     =24;                        % Number of rows
LAT.Ny                     =16;                        % Number of columns
LAT.dist_z                 =0.6;                       % antenna distance on z axis/elevation axis [times lambda]
LAT.dist_y                 =0.53;                      % antenna distance on y axis/azimuthal axis [times lambda]
LAT.lattice_type           =1;                         % select the type of lattice. If =1 is Rectangular grid, if =2 is Squared grid,
% if ==3 Triangular Equilateral grid, if ==4 Triangular NON-equilateral Grid, if==5 Exagonal grid (triangular with alpha=30 deg).

%%% MASK SPECIFICATION:
MASK.elem                  =30;                        % [deg] half-FoV width elevation plane
MASK.azim                  =60;                        % [deg] half-FoV width azimuthal plane
MASK.SLL                   =10;                        % [dB] SLL level outside the FoV

%%% SINGLE ELEMENT RPE:
EEF.P                      =1;                         % Set P=0 to achieve an isotropic element pattern, set P=1 for cosine element pattern
EEF.Ge1                    =5;                         % Maximum antenna Gain [dB]
EEF.load_file              =3;                         % Set load_file =1 to load antenna element RPE from HFSS, set load_file =2 to generate Cosine-like pattern
% if ==3 make analytic expression of pattern
EEF.rpe_folder             = [actual '\single_element_RPE\'];     % name of the folder with antenna element RPE from HFSS
EEF.rpe_file_name          = 'RPE_element.csv';        % name of the file with antenna element RPE from HFSS

%%% SIMULATION PARAMETERS
Niter                      =1000;                      %10e8; % Number of iteration
Cost_thr                   =1000;

%%% CLUSTER TYPE: deselect the one you need
CL.rotation_cluster        =0;                         % if flag==1 make cluster rotation, otherwise don't
% CL.Cluster_type{1}         =[0,0];                    % single element cluster / NO clustering support
%%%%% cluster size: 2 antenna elements
CL.Cluster_type{1}         =[0,0;0,1];                 % vertical linear cluster
% CL.Cluster_type{2}         =[0,0;1,0];                 % horizontal linear cluster
%%%%% cluster size: 3 antenna elements
% CL.Cluster_type{3}         =[0,0;0,1;0,2];             % vertical linear cluster
% CL.Cluster_type{1}         =[0,0;1,0;2,0];             % horizontal linear cluster
%%%%% cluster size: 4 antenna elements
% CL.Cluster_type{1}         =[0,0;0,1;0,2;0,3];         % vertical linear cluster
% CL.Cluster_type{1}         =[0,0;1,0;2,0;3,0];         % horizontal linear cluster

%%
FLAG.make_plot             = true;                     % if flag==1 make plot, otherwise don't
FLAG.save_data             = true;                     % if flag==1 save data, otherwise don't
FLAG.folder_name           = [actual '\Recursive_tool_results\AOB'];  % name of folder where data
% FLAG.file_name             ='Antenna_12x16_cluster_vert_3_0_d2d0_7.mat';         % name of saved file
% FLAG.file_name             ='Antenna_12x16_no_cluster_d2d0_6.mat';                % name of saved file
% FLAG.file_name             ='Antenna_12x16_cluster_vert_3_2_1_dz=0_7_dam.mat';    % name of saved file
% FLAG.file_name             ='Antenna_15x16_cluster_vert_3_2_1_dz=0_6.mat';
% FLAG.file_name             ='Antenna_16x16_cluster_vert_3_2_1_dz=0_6.mat';
% FLAG.file_name             ='Antenna_24x16_cluster_vert_3_2_1_dz=0_6.mat';
FLAG.file_name             ='Antenna_24x16_cluster_vert_2_dz=0_6.mat';