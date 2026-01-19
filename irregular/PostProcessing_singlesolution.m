% PostProcessing_singlesolution.m
%
% This code is used to evaluate REGULAR/IRREGULAR clustering performance.
% The user can set different kind of parameters according to the single
% antenna element radiation pattern, the mask characteristics and the
% cluster type.
%
% Version: 1.0;
% Develloped by Laura Resteghini (Milan RC.) delivered on 15/06/2018.
%
%%CLEANING OF WORKSPACE, GLOBAL VARIABLES AND GENERAL PARAMETERS

clc; clear all; close all;
actual=cd;

%
%%%% INPUT:

% ANTENNA ARRAY PARAMETERS
folder_results = [actual '\Recursive_tool_results\AOB'];     % name of folder where is the saved file
file_name = 'example_file.mat';                               % name of the results file
selezionato=16;                                               % selected element in the results file
color_line='r';                                               % color of the results
label_line='Irregular clustering';                            % label of the results
save_folder='';                                               % where save results

% MASK SPECIFICATION:
azi0=0;                                                       % [deg] azimuth steering angle
ele0=10;                                                      % [deg] elevation steering angle
elem=15;                                                      % [deg] half-FoV width elevation plane
azim=60;                                                      % [deg] half-FoV width azimuthal plane

SLL_level=15;                                                 % [dB] SLL level outside the FoV
SLlin=10;                                                     % [dB] SLL level inside the FoV

% SINGLE ELEMENT RPE:
P=1;                                                          % Set P=0 to achieve an isotropic element pattern, set P=1 for cosine element pattern
Gel=5;                                                        % Maximum antenna Gain [dB]
load_file=0;                                                  % Set load_file =1 to load antenna element RPE from HFSS,set load_file =0 to generate isotropic pattern
rpe_folder= [actual '\single_element_RPE'];                   % name of the folder with antenna element RPE from HFSS
rpe_file_name = 'RPE_element.csv';                            % name of the file with antenna element RPE from HFSS

% FLAGS:
save_data=0;                                                  % if flag=1 save data, otherwise don't
half_antenna=0; % if flag=1 turn on right-half panel; % if flag=2 turn on left-half panel;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
LOAD SIMULATED DATA                      %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cd(folder_results)
load(file_name)
load(['selected_cluster_' file_name]) % 'B'         .
load(['solution_clusters_' file_name]) % 'simulation'
cd(actual)

f=simulationBF.f;
Nz=simulationBF.Nz;
Ny=simulationBF.Ny;                                           % number of columns
dist_z=simulationBF.dist_z;                                   % antenna distance on z axis [times lambda]
dist_y=simulationBF.dist_y;                                   % antenna distance on y axis [times lambda]
x1=simulationBF.x1;
x2=simulationBF.x2;
Smod=simulationBF.Smod;
C_ori=simulationBF.C_ori;

% CUSTOMIZED PARAMETERS
% f=40*10^9;
% dist_z=0.6981;% per broadway 4:1
% dist_y=0.531;% per broadway 4:1
% x1=[3*10^8/f*dist_y_0];%simulationBF.x1;
% x2=[0 3*10^8/f*dist_z_1];%simulationBF.x2;
%
if save_data
    cd(save_folder)
    diary(['label_line '_v' num2str(ele0) 'h' num2str(azi0) '.txt'])
    cd(actual)
    diary on
end
disp(['**** SYSTEM PARAMETERS:'])
disp(['-> Working frequency f=' num2str(f/10^9) ' GHz'])
disp(['-> Number of elements Nz=' num2str(Nz) ',Ny=' num2str(Ny) ' --> Tot=' num2str(Nz*Ny)])
disp(['-> Required SLL suppression outside sector SLL_{in}=-' num2str(SLL_level) ' dB and inside sector SLL_{in}=-' num2str(SLL_level)])
disp(['-> Steering angle ele_0=' num2str(ele0) ' [°]/ azi_0=' num2str(azi0) ' [°]'])
disp(['-> Inter-elemt distance dist_z=' num2str(dist_z) ' *lambda / dist_y=' num2str(dist_y) ' *lambda'])

scale=3e8/f*1000; % [mm]
lambda=3e8/f; % [m]
beta=2*pi/lambda;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  LATTICE  AND CLUSTER EVALUATION  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LATTICE SELECTION - select the type of lattice
% Generate the basic lattice of grid-points
[Y,Z,NN,MM,Dy,Dz,ArrayMask]=GenerateLattice(Ny,Nz,x1,x2);
Dy=Dy+x1(1);
Dz=Dz+x2(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  POLAR COORDINATE AND STEERING VECTOR  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AZIMUT AND ELEVATION SAMPLING (for plots)
dele=.5; % angle resolution [deg]
dazi=.5; % angle resolution [deg]
eles=-90:dele:90;
azis=-90:dazi:90;
[AZI,ELE]=meshgrid(azi,ele);
WWae=beta*cosd(90-ELE);
Vvae=beta*sind(90-ELE).*sind(AZI);

% Nyquist SPECTRAL SAMPLING FOR OPTIMIZATION [ Antenna Synthesis]
chi=2; % sampling factor
Nw=floor(chi*4*Dz/lambda);  % Power Pattern Sampling (double)
Nv=floor(chi*4*Dy/lambda);  % Power Pattern Sampling (double)

ww=linspace(0,beta,Nw+1);
ww=[-flipir(ww(2:end)), ww];
vv=linspace(0,beta,Nv+1);
vv=[-flipir(vv(2:end)), vv];
[WW,VV]=meshgrid(ww,vv); % uniform sampling step in u/v coordinate
% Equivalent azimuth and elevation positions
% non-uniform sampling step in ele/azimuth angle
ELEi=90-acosd(WW./beta);
AZTi=real(asind(VV./(beta*sind(90-ELEi)))); % remove the NaN values
AZTi(Nv+1,1:2*Nw+1)=90;
AZTi(Nv+1,[1,2*Nw+1])=90;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%        RPE and ARRAY FACTOR        %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ELEMENT FACTOR - single element radiation pattern
[Fel, Fel_VW, RPE, RPE_ele_max] = ElementPattern_v2d0(P,Gel,ELE,AZI,AZTi,ELEi,AZTi,load_file,rpe_folder,rpe_file_name);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MASK EVALUATION
Nel=Nz*Ny;                                                    % number of array elements
[Isll_in,Isll_out,Mask_1D,Mask_2D,Mask_EA]=mask_design_v2d0(Nel,Nv,Nw,vv,ww,WW,VV,WWae,Vvae,beta,ELE,AZI,elem,azim,SLL_level,RPE_ele_max);
[Isll_in_in,Mask_2D_in,Mask_2D_in,Mask_EA_in]=mask_design_v2d0(Nel,Nv,Nw,vv,ww,WW,VV,WWae,Vvae,beta,ELE,AZI,0,0,SLlin,RPE_ele_max);

ElementExc=ones(Nz,Ny); % Fixed array tapering [BFN]

delta=0;
elem1=0;
for ib=1:size(B,2)
    vectrow_ib=vectrow(delta+1:delta+size(C_ori{ib},1));
    selected_rows = find(vectrow_ib==1);
    for ic=1:size(selected_rows,2)
        elem1=elem1+1;
        Cluster{elem1}=Smod{ib}(:,2*(selected_rows(ic)-1)+1:2*(selected_rows(ic)-1)+2);
    end
    delta=delta+size(C_ori{ib},1);
end

[Yc,Zc,Ac]= Index2Position_cluster_v2d0(Cluster,Y,Z,ElementExc,NN,MM);    % Sub-array partitioning presentation

Ntrans=size(Yc,2);
Lsub=NaN*zeros(1,Ntrans);
Zc_m=NaN*zeros(1,Ntrans);
Yc_m=NaN*zeros(1,Ntrans);
for kk=1:Ntrans
    Lsub(kk) = size(Cluster{kk},1);                          % Phase center of sub-array
    Zc_m(kk) = mean(Zc(1:Lsub(kk),kk));                      % Phase center of sub-array
    Yc_m(kk) = mean(Yc(1:Lsub(kk),kk));                      % Phase center of sub-array,
end

%%% EXCITATIONS
v0=beta*sind(90-ele0)*sind(azi0);
w0=beta*cosd(90-ele0);

Iele=find((ele-ele0)>=0,1);
Iazi=find((azi-azi0)>=0,1);

c0=coefficient_evaluation(w0,v0,Zc_m,Yc_m,Lsub);

if half_antenna==1
    quali=find(Yc_m<=0);
    %          quali=1:Ntrans/2;
    c0(quali)=0; % spenti
    sum(Lsub(quali))

    figure
    for ih=1:size(Yc,2)
        plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor',[65,105,225]./255,'MarkerFaceColor',[65,105,225]./255,'MarkerSize',8);
        hold on
    end
    grid
    xlabel('y [m]')
    ylabel('z [m]')
    title('In rosso gli spenti')
    axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])

    plot(Yc_m(quali),Zc_m(quali),'sq','MarkerEdgeColor','r','MarkerSize',10);
elseif half_antenna==2
    quali=find(Yc_m>0);
    %          quali=1:Ntrans/2;
    c0(quali)=0; % spenti
    sum(Lsub(quali))

    figure
    for ih=1:size(Yc,2)
        plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor',[65,105,225]./255,'MarkerFaceColor',[65,105,225]./255,'MarkerSize',8);
        hold on
    end
    grid
    xlabel('y [m]')
    ylabel('z [m]')
    title('In rosso gli spenti')
    axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])

    plot(Yc_m(quali),'sq','MarkerEdgeColor','r','MarkerSize',10);
end

%%% FAR FIELD TRANSFORMATION KERNELS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(Nw, Nv, Lsub, Ac, W, WW, Wae, WWae, Yc, Zc, c0, Fel_VW, Nel);

%%% POST PROCESSING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Cost function evaluation
Constr=FF_I_dB-Mask_EA;          % SSL constraint
Cm=sum(sum(Constr>0));

%%% Where is the MAXIMUM RPE value
[maxNumCol, maxIndexCol] = max(FF_I_dB);
[maxNum, Iazi_max] = max(maxNumCol);
Iele_max = maxIndexCol(Iazi_max);
theta_max =ele(Iele_max);
phi_maxs= azi(Iazi_max);
theta_st =ele(Iele);
phi_st= azi(Iazi);
disp(' ')
disp('max(theta) max(phi) theta_0 phi_0 ')
disp([theta_max phi_max theta_st phi_st])

%%%% RPE parameters
c0=c0.';
G_boresight=max(max(RPE))+10*log10(sum(Lsub(find(c0~=0))));
G0=10*log10( (max abs(KerFF_sub*c0)).^2/ sum(Lsub.*abs(c0.').^2));    %[dB] % clustering with optimization
SL_maxpointing=G_boresight-FF_I_dB(Iele_max,Iazi_max) % scan loss in the maximum RPE beam
SL_theta_phi=G_boresight-FF_I_dB(Iele,Iazi)               % scan loss in the desired direction
SLL_threshold=G_boresight-SLL_level
[SLL_in, SLL_out]=SLL_in_out(FF_I_dB,ele,azi,elem,azim,AZI,ELE,Iazi,Iele_max,Iazi_max)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%  PLOTS  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
for ih=1:size(Yc,2)
    RGBcolor=rand(1,3);
    subplot(1,2,1)
    plot(Yc(:,ih),Zc(:,ih),'sq','MarkerEdgeColor',RGBcolor,'MarkerFaceColor',RGBcolor,'MarkerSize',8);
    hold on
    subplot(1,2,2)
    plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor',RGBcolor,'MarkerFaceColor',RGBcolor,'MarkerSize',8);
    hold on
end
subplot(1,2,1)
grid
xlabel('y [m]')
ylabel('z [m]')
title('Antenna Sub-arrays Cluster Element')
axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])
subplot(1,2,2)
grid
xlabel('y [m]')
ylabel('z [m]')
title('Antenna Sub-arrays Phase Center')
axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])
if save_data
    cd(save_folder)
    saveas(gcf,['map.fig'])
end

figure
for ih=1:size(Yc,2)
    plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',8);
    hold on
end
grid
xlabel('y [m]')
ylabel('z [m]')
title('Antenna Sub-arrays Phase Center')
axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])

figure
subplot(2,2,1)
for ih=1:size(Yc,2)
    RGBcolor=rand(1,3);
    plot(Yc(:,ih),Zc(:,ih),'sq','MarkerEdgeColor',RGBcolor,'MarkerFaceColor',RGBcolor,'MarkerSize',8);
    hold on
end
grid
xlabel('y [m]')
ylabel('z [m]')
title('Antenna Sub-arrays')
axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])
subplot(2,2,2)
y=1:max(Lsub);
for ik1=1:size(x,2)
    y(ik1)=size(find(Lsub==ik1),2);
end
bar(x,y)
xlabel('Cluster size')
ylabel('Number of cluster')
title(num2str(sum(Lsub)))
% Figure plotting cardinal planes FF cuts
subplot(2,2,3)
plot(ele,FF_I_dB(:,Iazi),'b','Linewidth',2)
hold on
plot(ele,(Mask_EA(:,Iazi)),'g','Linewidth',2)
axis([-90,90,-30,max(max(Mask_EA(:,Iazi)))]+0.5]);grid
xlabel('\theta');
ylabel('RPE R(\theta,\phi)');
legend('RPE','Mask')
title('Vertical plane')
subplot(2,2,4)
plot(azi,FF_I_dB(Iele,:),'b','Linewidth',2)
hold on
plot(azi,(Mask_EA(Iele,:)),'g','Linewidth',2)
axis([-90,90,-30,max(max(Mask_EA(Iele,:)))]+0.5]);grid
xlabel('\phi');
ylabel('RPE R(\theta,\phi)');
legend('RPE','Mask','RPE_max')
subplot(2,2,3)
plot(ele,FF_I_dB(:,Iazi_max),'r','Linewidth',2)
legend('RPE','Mask','RPE_max')
subplot(2,2,4)
plot(azi,FF_I_dB(Iele_max,:),'r','Linewidth',2)
legend('RPE','Mask','RPE_max')

figure
subplot(1,2,1)
plot(ele,FF_I_dB(:,Izi),color_line,'Linewidth',2)
hold on
plot(ele,(Mask_EA(:,Izi)),'g','Linewidth',2)
axis([-90,90,-30,max(max(Mask_EA(:,Izi)))]+0.5]);grid
xlabel('\theta');
ylabel('RPE R(\theta,\phi)');
legend([label_line ' \phi=' num2str(azi0) ' [◆]'],'Mask')
title('Vertical plane')
subplot(1,2,2)
plot(azi,FF_I_dB(Iele,:),color_line,'Linewidth',2)
hold on
plot(azi,(Mask_EA(Iele,:)),'g','Linewidth',2)
axis([-90,90,-30,max(max(Mask_EA(Iele,:)))]+0.5]);grid
xlabel('\phi');
ylabel('RPE R(\theta,\phi)');
legend('RPE','Mask')
title('Horizontal plane')
legend([label_line ' \theta=' num2str(ele0) ' [◆]'],'Mask')
if save_data
    cd(save_folder)
    saveas(gcf,['cut_RPE_v' num2str(ele0) 'h' num2str(azi0) '.fig'])
end

F_plot=FF_I_dB;
F_plot(FF_I_dB<-30)=-30;
figure
contourf(azi, ele,F_plot)
c=colorbar;
c.Label.String = 'Realized Gain [dBi]';
title([label_line ' - Radiation Pattern [ \theta=' num2str(ele0) ' [◆, \phi=' num2str(azi0) ' [◆]'])
xlabel('\phi [deg]');
ylabel('\theta [deg]');

if save_data
    cd(save_folder)
    saveas(gcf,['3D_RPE_v' num2str(ele0) 'h' num2str(azi0) '.fig'])
    saveas(gcf,['3D_RPE_v' num2str(ele0) 'h' num2str(azi0) '.png'])
end

figure
surf(azi(1:4:361), ele(1:4:361),F_plot(1:4:361,1:4:361))
c=colorbar;
c.Label.String = 'Realized Gain [dBi]';
title([label_line ' - Radiation Pattern [ \theta=' num2str(ele0) ' [◆, \phi=' num2str(azi0) ' [◆]'])
xlabel('\phi [deg]');
ylabel('\theta [deg]');

SL_real=max(max(RPE))+10*log10(Nel)-max(max(FF_I_dB(Iele,Iazi)))

%%%%%%%%%%%% CDF
FF_cdf_dB = FF_norm_dB;
y_vector=FF_cdf_dB(Isll_in); % select IN of sector values
[x_cdf y_cdf]=cdf_plot(y_vector);
legend(['\theta=' num2str(ele0) ' [◆ - \phi=' num2str(azi0) ' [◆ - maxCDF=' num2str(y_cdf(1)) ])

FF_cdf_dB = FF_norm_dB;
y_vector=FF_cdf_dB(Isll_out); % select Out of sector values
[x_cdf y_cdf]=cdf_plot(y_vector);
legend(['\theta=' num2str(ele0) ' [◆ - \phi=' num2str(azi0) ' [◆ - maxCDF=' num2str(y_cdf(1)) ])

if save_data
    diary off
end