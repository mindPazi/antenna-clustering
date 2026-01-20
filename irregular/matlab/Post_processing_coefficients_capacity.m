% Post_processing_coefficients_capacity.m
%
% In this code we evaluate the coefficients of a given antenna
% configuration and the information necessary to make the capacity
% performance.
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
%
% ANTENNA ARRAY PARAMETERS
folder_results = [actual '\Recursive_tool_results\AOB'];     % name of folder where is the saved file
file_name = 'example_file.mat';                            % name of the results file
selezionato=16;                                            % selected element in the results file
color_line='r';                                            % color of the results
label_line='Irregular clustering';                         % label of the results

file_folder_coeff = '\Generate_HFSS_coefficients_file\AOB';
coeff_file_name = 'example_file.dat';

folder_capacity= '\Dati_capacity_canale';
capacity_file_name='example_file.dat';

% MASK SPECIFICATION:
azi0=0;                    % [deg] azimuth steering angle
ele0=10;                   % [deg] elevation steering angle

elem=15;                   % [deg] half-FoV width elevation plane
azim=60;                   % [deg] half-FoV width azimuthal plane

SLL_level=15;              % [dB] SLL level outside the FoV
SLLin=10;                  % [dB] SLL level inside the FoV

% SINGLE ELEMENT RPE:
P=1;                       % Set P=0 to achieve an isotropic element pattern, set P=1 for cosine element pattern
Gel=5;                     % Maximum antenna Gain [dB]
load_file=0;               % Set load_file =1 to load antenna element RPE from HFSS,set load_file =0 to generate isotropic pattern
rpe_folder= [actual '\single_element_RPE']; % name of the folder with antenna element RPE from HFSS
rpe_file_name = 'RPE_element.csv'; % name of the file with antenna element RPE from HFSS

% FLAGS:
make_capacity = 1; % if flag==1 do capacity analysis, otherwise don't
make_coefficients=1;% if flag==1 do coefficients analysis, otherwise don't

%_______________________________________________________________
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

%_________________________________________________________________________
scale=3e8/f*1000; % [mm]
lambda=3e8/f; % [m]
beta=2*pi/lambda;

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  LATTICE AND CLUSTER EVALUATION  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LATTICE SELECTION - select the type of lattice
% Generate the basic lattice of grid-points
[Y,Z,NN,MM,Dy,Dz,ArrayMask]=GenerateLattice (Ny,Nz,x1,x2);

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

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%                MASK EVALUATION                  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of array elements
Nel=Nz*Ny;
[Isll_in,Isll_out,Mask_1D,Mask_2D,Mask_EA]=mask_design_v2d0(Nel,Nv,Nw,vv,ww,WW,VV,WWae,Wvae,beta,ELE,AZI,elem,azim,SLL_level,RPE_ele_max);

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

%%% EXCITATIONS
v0=beta*sind(90-ele0)*sind(azi0);
w0=beta*cosd(90-ele0);
Iele=find((ele-ele0)>=0,1);
Iazi=find((azi-azi0)>=0,1);
c0=coefficient_evaluation(w0,v0,Zc_m,Yc_m,Lsub);

%%% FAR FIELD TRANSFORMATION KERNELS
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(Nw, Nv, Lsub, Ac, WV, WW, Wvae, WWae, Yc, Zc, c0, Fel_VW, Nel);

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%              PLOTS               %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(1,2,1)
plot(ele,FF_I_dB(:,Iazi),'b','LineWidth',2)
hold on
plot(ele,(Mask_EA(:,Iazi)),'g','LineWidth',2)
axis([-90,90,-30,max(max(Mask_EA(:,Iazi)))+0.5]);grid
xlabel('\theta');
ylabel('RPE R(\theta,\phi)');
legend('Irregular','Mask')
title('Vertical plane')
subplot(1,2,2)
plot(azi,FF_I_dB(Iele,:),'b','LineWidth',2)
hold on
plot(azi,(Mask_EA (Iele,:)),'g','LineWidth',2)
axis([-90,90,-30,max(max(Mask_EA(Iele,:)))+0.5]);grid
xlabel('\phi');
ylabel('RPE R(\theta,\phi)');
legend('RPE','Mask')
title('Horizontal plane')
legend('Irregular','Mask')

%_________________________________________________________________________
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%              POST-ANALYSIS              %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% CAPACITY ANALYSIS
if make_capacity
    Z_coordinate=reshape(Z,1,Nel);
    Y_coordinate=reshape(Y,1,Nel);

    % Cluster_boolean=C_selected./Lsub';
    for ij=1:Ntrans
        Cc{ij}=[Yc(1:Lsub(ij),ij) Zc(1:Lsub(ij),ij)]; % cluster coordinate
    end

    for ij=1:Ntrans
        for il=1:Lsub(ij)
            Nc{ij}(il)=find(Z_coordinate==Cc{ij}(il,2) & Y_coordinate==Cc{ij}(il,1));
        end
    end

    Cluster_boolean=zeros(Ntrans,Nel);
    Map=zeros(Nz,Ny);
    for ij=1:Ntrans
        Cluster_boolean(ij,Nc{ij})=sqrt(1/Lsub(ij));
        Map(Nc{ij})=ij;
    end

    if make_capacity
        plot(Cluster_boolean)

        cd([actual folder_capacity])
        parameter.f = f;
        parameter.Nz = Nz;
        parameter.Ny = Ny;
        parameter.dz = dist_z;
        parameter.dy = dist_y;
        parameter.Z_coordinate=Z_coordinate;
        parameter.Y_coordinate=Y_coordinate;
        parameter.Cluster_boolean=Cluster_boolean;
        save(capacity_file_name,'parameter')

    end

end

%%% COEFFICIENTS EVALUATION
if make_coefficients
    for kk=1:Ntrans
        ah=Cluster{kk};
        if mod(Ny,2) & mod(Nz,2) %entrambi dispari
            NClu{kk} = [ah(:,1)+(Ny-1)/2+1 (Ny-1)/2+1 ah(:,2)+(Nz-1)/2+1];
        elseif ~mod(Ny,2) & ~mod(Nz,2) %entrambi pari
            NClu{kk} = [ah(:,1)+Ny/2 ah(:,2)+Nz/2];
        elseif ~mod(Ny,2) & mod(Nz,2)
            NClu{kk} = [ah(:,1)+(Ny-1)/2+1 ah(:,2)+Nz/2];
        elseif mod(Ny,2) & ~mod(Nz,2)
            NClu{kk} = [ah(:,1)+(Ny-1)/2+1 ah(:,2)+Nz/2];
        end
    end

    for kk=1:Ntrans
    end

    for kk=1:Ntrans
        Coeff_ampl(NClu{kk}(:,2),NClu{kk}(:,1))=abs(c0(kk)).^2;
        Coeff_phase(NClu{kk}(:,2),NClu{kk}(:,1))=rad2deg(angle(c0(kk)));
    end

    %         Coeff_ampl=flipud(Coeff_ampl); %%%%
    %         Coeff_phase=flipud(Coeff_phase); %%%%

    figure
    subplot(1,2,1)
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
    subplot(1,2,2)
    for ih=1:size(NClu,2)
        RGBcolor=rand(1,3);
        plot(NClu{ih}(:,1),NClu{ih}(:,2),'sq','MarkerEdgeColor',RGBcolor,'MarkerFaceColor',RGBcolor,'MarkerSize',8);
        hold on
    end
    grid
    xlabel('y [m]')
    ylabel('z [m]')
    title('Antenna Sub-arrays')
    axis([1 Ny 1 Nz])

    figure
    for ij=1:Nz
        plot(Coeff_ampl(ij,:))
        hold on
    end

    figure
    subplot(1,2,1)
    for ih=1:size(NClu,2)
        if size(NClu{ih},1)==4
            plot(NClu{ih}(:,1),NClu{ih}(:,2),'sq','MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',8);
        else
            plot(NClu{ih}(:,1),NClu{ih}(:,2),'sq','MarkerEdgeColor','g','MarkerFaceColor','y','MarkerSize',8);
        end
        hold on
    end
    title('Coefficients Amplitude')
    h = colorbar;
    set(get(h,'title'),'string','[W]');
    subplot(1,2,2)
    pcolor(Coeff_ampl);
    title('Coefficients Amplitude')
    h = colorbar;
    set(get(h,'title'),'string','[W]');
    subplot(1,2,2)
    pcolor(Coeff_phase)
    title('Coefficients Phase')
    h = colorbar;
    set(get(h,'title'),'string','[deg]');

    figure
    subplot(1,2,1)
    pcolor(Coeff_ampl);
    title('Coefficients Amplitude')
    h = colorbar;
    set(get(h,'title'),'string','[W]');
    subplot(1,2,2)
    pcolor(Coeff_phase)
    title('Coefficients Phase')
    h = colorbar;
    set(get(h,'title'),'string','[deg]');

    Coeff.phase=Coeff_phase;
    Coeff.ampl=Coeff_ampl;

    list=dir;
    for ij=1:size(list,1)
        ce(ij)=strcmpi(list(ij).name,[file_folder_coeff]);
    end
    if sum(ce)==0
        mkdir(actual, file_folder_coeff)
    end

    cd([actual file_folder_coeff])
    save(coeff_file_name,'Coeff')
end