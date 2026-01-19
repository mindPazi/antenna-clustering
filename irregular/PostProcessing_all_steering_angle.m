% PostProcessing_all_steering_angle.m
%
% Analysis to evaluate the performance of simulated regular/irregular
% clustering over the entire FoV. Performance are evaluated in terms of SLL
% inside FoV, SLL outside FoV (and those exceeding the SLL threshols), Peak
% Gain and EIRP.
%
% Version: 1.0;
% Develloped by Laura Resteghini (Milan RC.) delivered on 15/06/2018.
%___________________________________________________________________________
%%CLEANING OF WORKSPACE, GLOBAL VARIABLES AND GENERAL PARAMETERS

clc
clear all
close all
actual=cd;

%
%%%_______________________________________________________________
%%% INPUT:

% ANTENNA ARRAY PARAMETERS
folder_results = [actual '\Recursive_tool_results\AOB'];
file_name = 'example_file.mat';                            % selected element in the results file
selezionato=16;                                            % color of the results
color_line='r';                                            % label of the results
label_line='Irregular clustering';                         % where save results
save_folder='';

% MASK SPECIFICATION:
azi0=-52:4:52;  % [deg] angle for the maximization (minimization) of directivity
ele0=-12:4:12;  % [deg] angle for the maximization (minimization) of directivity

elem=15;        % [deg] half-FoV width elevation plane
azim=60;        % [deg] half-FoV width azimuthal plane

SLL_level=15;   % [dB] SLL level outside the FoV
SLLin=10;       % [dB] SLL level inside the FoV

% SINGLE ELEMENT RPE:
P=1;            % Set P=0 to achieve an isotropic element pattern, set P=1 for cosine element pattern
Gel=5;          % Maximum antenna Gain [dB]
load_file=0;    % Set load_file =1 to load antenna element RPE from HFSS, set load_file =o to generate isotropic pattern
rpe_folder= [actual '\single_element_RPE']; % name of the folder with antenna element RPE from HFSS
rpe_file_name = 'RPE_element.csv'; % name of the file with antenna element RPE from HFSS

P_tx_chain=9; %[dBm] % select Tx chain power
Losses= 0; %[dB] select Losses

% FLAGS:
save_data=0; % if flag==1 save data, otherwise don't
eirp_eval=1; % if flag==1 make EIRP analysis, otherwise don't

%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%/%%%%%%%%%%%             LOAD SIMULATED DATA                %%%%%/%%%%%%%%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(folder_results)
load(file_name)
load(['selected_cluster_' file_name]) % 'B'
load(['solution_clusters_' file_name]) % 'simulation'
cd(actual)

f=simulationBF.f;
Nz=simulationBF.Nz;                          % number of rows
Ny=simulationBF.Ny;                          % number of columns
dist_z=simulationBF.dist_z;  % antenna distance on z axis [times lambda]
dist_y=simulationBF.dist_y;  % antenna distance on y axis [times lambda]
x1=simulationBF.x1;
x2=simulationBF.x2;
Smod=simulationBF.Smod;
C_ori=simulationBF.C_ori;

% CUSTOMIZED PARAMETERS
% f=40*10^9;
% dist_z=0.6981;% per broadway 4:1
% dist_y=0.531;% per broadway 4:1
% x1=[3*10^8/f*dist_y 0];%simulationBF.x1;
% x2=[0 3*10^8/f*dist_z];%simulationBF.x2;

%_________________________________________________________________________

disp(['**** SYSTEM PARAMETERS:'])
disp(['-> Working frequency f=' num2str(f/10^9) ' GHz'])
disp(['-> Number of elements Nz=' num2str(Nz) ',Ny=' num2str(Ny) ' --> Tot=' num2str(Nz*Ny)])
disp(['-> Required SLL suppression outside sector SLL{out}=' num2str(SLL_level) ' dB and inside sector SLL{in}=' num2str(SLLin) ' dB'])
disp(['-> Steering angle ele_0=' num2str(ele0) '°/ azi_0=' num2str(azi0) '°'])
disp(['-> Inter-elemnt distance dist_z=' num2str(dist_z) ' / dist_y=' num2str(dist_y) ])

if save_data
    mkdir(save_folder);
end
%_________________________________________________________________________

scale=3e8/f*1000; % [mm]
lambda=3e8/f; % [m]
beta=2*pi/lambda;

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  LATTICE AND CLUSTER EVALUATION  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LATTICE SELECTION - select the type of lattice
% Generate the basic lattice of grid-points
[Y,Z,NN,MM,Dy,Dz,ArrayMask]=GenerateLattice(Ny,Nz,x1,x2,Inf,Inf);

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  POLAR COORDINATE AND STEERING VECTOR  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% AZIMUT AND ELEVATION SAMPLING (for plots)
dele=.5; % angle resolution [deg]
dazi=.5; % angle resolution [deg]
ele=-90:dele:90;
azi=-90:dazi:90;
[AZI,ELE]=meshgrid(azi,ele);
WWae=beta*cosd(90-ELE);
Wvae=beta*sind(90-ELE).*sind(AZI);

%%% Nyquist SPECTRAL SAMPLING FOR OPTIMIZATION [ Antenna Synthesis]
chi=2;                              % sampling factor
Nw=floor(chi*4*Dz/lambda);          % Power Pattern Sampling (double)
Nv=floor(chi*4*Dy/lambda);          % Power Pattern Sampling (double)
ww=linspace(0,beta,Nw+1);
ww=[-fliplr(ww(2:end)), ww];
vv=linspace(0,beta,Nv+1);
vv=[-fliplr(vv(2:end)), vv];
[WW,VV]=meshgrid(ww,vv); % uniform sampling step in u/v coordinate
% Equivalent azimuth and elevation positions
% non-uniform sampling step in ele/azimuth angle
ELEi=90-acosd(WW./beta);
AZIi=real(asind(VV./(beta*sind(90-ELEi)))); % remove the NaN values
AZIi(Nv+1,2:2*Nw)=0;
AZIi(Nv+1,[1,2*Nw+1])=90;

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  RPE and ARRAY FACTOR  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%% ELEMENT FACTOR - single element radiation pattern
[Fel, Fel_VW, RPE, RPE_ele_max] = ElementPattern_v2d0(P,Gel,ELE,AZI,ELEi,AZIi,load_file,rpe_folder,rpe_file_name);

Nel=Nz*Ny;          % number of array elements
G_boresight=max(max(RPE))+10*log10(Nel);

%%% ALGORITHM X SECTION - Sub-array definition  [B0 defines the basic structure of the sub-array]
vectorrow=simulation(selezionato,1:end-3);

ElementExc=ones(Nz,Ny); % Fixed array tapering [BFN]
delta=0;
elemi=0;
for ib=1:size(B,2)
    vectorrow_ib=vectorrow(delta+1:delta+size(C_ori{ib},1));
    selected_rows = find(vectorrow_ib==1);
    for ic=1:size(selected_rows,2)
        elemi=elemi+1;
        Cluster{elemi}=Smod{ib}(:,2*(selected_rows(ic)-1)+1:2*(selected_rows(ic)-1)+2);
    end
    delta=delta+size(C_ori{ib},1);
end

[Yc,Zc,Ac]= Index2Position_cluster_v2d0(Cluster,Y,Z,ElementExc,NN,MM);     % Sub-array partitioning presentation

Ntrans=size(Yc,2);
Lsub=NaN*zeros(1,Ntrans);
Zc_m=NaN*zeros(1,Ntrans);
Yc_m=NaN*zeros(1,Ntrans);
for kk=1:Ntrans
    Lsub(kk) = size(Cluster{kk},1);
    Zc_m(kk) = mean(Zc(1:Lsub(kk),kk));  % Phase center of sub-array
    Yc_m(kk) = mean(Yc(1:Lsub(kk),kk));  % Phase center of sub-array
end

SLL_out_all=NaN*zeros(size(ele0,2),size(azi0,2));
for iaz=1:size(azi0,2)
    azi0(iaz)
    for iel=1:size(ele0,2)
        %
        %%% EXCITATIONS
        v0=beta*sind(90-ele0(iel))*sind(azi0(iaz));
        w0=beta*cosd(90-ele0(iel));
        Iele=find((ele-ele0(iel))>=0,1);
        Iszi=find((azi-azi0(iaz))>=0,1);
        c0=coefficient_evaluation(w0,v0,Zc_m,Yc_m,Lsub);

        %%% FAR FIELD TRANSFORMATION KERNELS
        [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(Nw, Nv, Lsub, Ac, WV, WW, Wvae, WWae, Yc, Zc, c0, Fel_VW, Nel);

        %%% POST PROCESSING %%%
        %%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Where is the MAXIMUM RPE value
        [maxNumCol, maxIndexCol] = max(FF_I_dB);
        [maxNum, Iazi_max] = max(maxNumCol);
        Iele_max = maxIndexCol(Iazi_max);

        SLL_threshold=G_boresight-SLL_level;
        [SLL_in, SLL_out]=SLL_in_out(FF_I_dB,ele,azi,elem,azim,AZI,ELE,Iazi,Iele_max,Iazi_max);
        SLL_out_sll(iel,iaz)=SLL_out;
        SLL_in_sll(iel,iaz)=SLL_in;
        RPE_peak_in(iel,iaz)=max(max(FF_I_dB));

    end
end

figure
contourf(azi0,ele0,SLL_out_sll)
xlabel('Beam steering in azimuth \phi_0 [deg]')
ylabel('Beam steering in elevation \theta_0 [deg]')
c=colorbar;
c.Label.String = 'SLL out [dB]';
title([label_line ' - max(SLL_{out})'])
mean(SLL_out_all(:))
if save_data
    cd(save_folder)
    saveas(gcf,'SLL_out.fig')
    saveas(gcf,'SLL_out.png')
end

figure
contourf(azi0,ele0,SLL_in_all)
xlabel('Beam steering in azimuth \phi_0 [deg]')
ylabel('Beam steering in elevation \theta_0 [deg]')
c=colorbar;
c.Label.String = 'SLL in [dB]';
title([label_line ' - max(SLL_{in})'])
mean(SLL_in_all(:))
if save_data
    cd(save_folder)
    saveas(gcf,'SLL_in.fig')
    saveas(gcf,'SLL_in.png')
end

figure
contourf(azi0,ele0,RPE_peak_in)
xlabel('Beam steering in azimuth \phi_0 [deg]')
ylabel('Beam steering in elevation \theta_0 [deg]')
c=colorbar;
c.Label.String = 'Realized Gain [dBi]';
title([label_line ' - Realized Gain'])
mean(RPE_peak_in(:))
if save_data
    cd(save_folder)
    saveas(gcf,'RG.fig')
    saveas(gcf,'RG.png')
end

if eirp_eval==1
    P_tx = 10*log10(Ntrans)+P_tx_chain-Losses;
    EIRP=RPE_peak_in+P_tx;
    figure
    contourf(azi0,ele0,EIRP)
    xlabel('Beam steering in azimuth \phi_0 [deg]')
    ylabel('Beam steering in elevation \theta_0 [deg]')
    c=colorbar;
    c.Label.String = 'EIRP [dBm]';
    title('Irregular clustering')
    mean(RPE_peak_in(:))
end

if save_data
    cd(save_folder)
    save('file_EIRP.mat','azi0','ele0','EIRP','RPE_peak_in','SLL_in_all','SLL_out_all','SLL_threshold')
end