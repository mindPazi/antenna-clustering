% ANTENNA ARRAY REGULAR/IRREGULAR CLUSTERING
% Version: 1.0;
% Develloped by Milan RC. delivered on 06/14/2018.
%%CLEANING OF WORKSPACE, GLOBAL VARIABLES AND GENERAL PARAMETERS

clc
clear all
close all
actual=cd;

%
%%% INPUT:

folder_results = [actual '\Recursive_tool_results\AOB'];     % name of folder where is the saved file
file_name = 'ref_16x16_4x1_29e9.mat';                        % name of the results file
selezionato=1;                                               % selected element in the results file
color_line='r';                                              % color of the results
label_line='Irregular clustering';                           % label of the results
save_folder='';                                              % where save results

azi0=0;        % [deg] azimuth steering angle
ele0=10;       % [deg] elevation steering angle

elem=15;       % [deg] half-FoV width elevation plane
azim=60;       % [deg] half-FoV width azimuthal plane

SLL_level=15;  % [dB] SLL level outside the FoV
SLLin=10;      % [dB] SLL level inside the FoV

%%% SINGLE ELEMENT RPE:
P=1;            % Set P=0 to achieve an isotropic element pattern, set P=1 for cosine element pattern
Gel=5;          % Maximum antenna Gain [dB]
load_file=0;    % Set load_file =1 to load antenna element RPE from HFSS,set load_file =o to generate isotropic pattern
rpe_folder= [actual '\single_element_RPE']; % name of the folder with antenna element RPE from HFSS
rpe_file_name = 'RPE_element.csv'; % name of the file with antenna element RPE from HFSS

save_data=0;    % if flag==1 save data, otherwise don't
flag_damage=0;
ps_bit=2;

%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  LOAD SIMULATED DATA        %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(folder_results)
load(file_name)
load(['selected_cluster_' file_name]) % 'B'
load(['solution_clusters_' file_name]) % 'simulation'
cd(actual)

f=simulationBF.f;
Nz=simulationBF.Nz;                          % number of rows
Ny=simulationBF.Ny;                          % number of columns
dist_z=simulationBF.dist_z; % antenna distance on z axis [times lambda]
dist_y=simulationBF.dist_y; % antenna distance on y axis [times lambda]
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

%%% [selected rows; cost function; number of selected clusters; number of antenna elements]
for ij=1:size(simulation,1)
    fcost(ij)=simulation(ij,end-2);
    Narray(ij)=simulation(ij,end-1);
    Elarray(ij)=simulation(ij,end);
end

if save_data
    cd(save_folder)
    diary([label_line '_v' num2str(ele0) '_h' num2str(azi0) '.txt'])
    cd(actual)
    diary on
end
disp(['**** SYSTEM PARAMETERS:'])
disp(['-> Working frequency f=' num2str(f/10^9) ' GHz'])
disp(['-> Number of elements Nz=' num2str(Nz) ',Ny=' num2str(Ny) ' --> Tot=' num2str(Nz*Ny)])
disp(['-> Required SLL suppression outside sector SLL_{out}=' num2str(SLL_level) ' dB and inside sector SLL_{in}=' num2str(SLLin) ' dB'])
disp(['-> Steering angle ele_0=' num2str(ele0) '°/ azi_0=' num2str(azi0) '°'])
disp(['-> Inter-elemnt distance dist_z=' num2str(dist_z) '*lambda / dist_y=' num2str(dist_y) '*lambda'])

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
Dy=Dy+x1(1);
Dz=Dz+x2(2);

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

%%% ELEMENT FACTOR - single element radiation pattern
[Fel, Fel_VW, RPE, RPE_ele_max] = ElementPattern_v2d0(P,Gel,ELE,AZI,ELEi,AZIi,load_file,rpe_folder,rpe_file_name);

Nel=Nz*Ny;          % number of array elements
[Isll_in,Isll_out,Mask_1D,Mask_2D,Mask_EA]=mask_design_v2d0(Nel,Nv,Nw,vv,ww,WW,WV,WWae,Wvae,beta,ELE,AZI,elem,azim,SLL_level,RPE_ele_max);
[Isll_in_in,Isll_out_in,Mask_1D_in,Mask_2D_in,Mask_EA_in]=mask_design_v2d0(Nel,Nv,Nw,vv,ww,WW,VV,WWae,Wvae,beta,ELE,AZI,0,0,SLLin,RPE_ele_max);

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vectorrow=simulation(selezionato,1:end-3);

%%% ALGORITHM X SECTION - Sub-array definition  [B0 defines the basic structure of the sub-array]

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

%%%%%/%%%%%%%%%%%%
Phase_ori=exp(-1i*(w0*Zc_m+v0*Yc_m)); % Clustered phase distribution
function_distr=2;
step_deg=360/2^ps_bit;
error_deg=[step_deg/2,step_deg,step_deg*2];
[Phase_m,phase_error_all]=phase_error_array(Phase_ori,ps_bit,error_deg,function_distr);
SubExc=ones(1,size(Yc_m,2));         % Clustered amplitude distribution (normalized to cluster size)
Amplit_m =sqrt(SubExc./Lsub);
c0=Amplit_m.*Phase_m;
%%%%%/%%%%%%%%%%%%

if flag_damage
    quanti_damage=22;
    damage=randperm(Ntrans);
    c0(damage(1:quanti_damage))=0;
    sum(Lsub(damage(1:quanti_damage)))

    figure
    for ih=1:size(Yc,2)
        plot(Yc_m(ih),Zc_m(ih),'sq','MarkerEdgeColor',[65,105,225]./255,'MarkerFaceColor',[65,105,225]./255,'MarkerSize',8);
        hold on
    end
    grid
    xlabel('y [m]')
    ylabel('z [m]')
    title('Antenna Sub-arrays Phase Center')
    axis([min(min(Yc))-0.005 max(max(Yc))+0.005 min(min(Zc))-0.005 max(max(Zc))+0.005])

    plot(Yc_m(damage(1:quanti_damage)),Zc_m(damage(1:quanti_damage)),'sq','MarkerEdgeColor','r','MarkerSize',10);
end

figure
col=['r';'b';'c';'m';'k';'b'];
for ij=1:size(Phase_m,1)
    c0=Amplit_m.*Phase_m(ij,:);
    %%% FAR FIELD TRANSFORMATION KERNELS
    %%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(Nw, Nv, Lsub, Ac, WV, WW, Wvae, WWae, Yc, Zc, c0, Fel_VW, Nel);

    %%% POST PROCESSING %%%
    %%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Cost function evaluation
    Constr=FF_I_dB-Mask_EA;         % SLL constraint
    Cm(ij)=sum(sum(Constr>0));

    %%% Where is the MAXIMUM RPE value
    [maxNumCol, maxIndexCol] = max(FF_I_dB);
    [maxNum, Iazi_max] = max(maxNumCol);
    Iele_max = maxIndexCol(Iazi_max);
    theta_max =ele(Iele_max);
    phi_max= azi(Iazi_max);
    theta_st =ele(Iele);
    phi_st= azi(Iazi);
    disp(' ')
    disp(['max(theta) max(phi) theta_0 phi_0 '])
    disp([theta_max phi_max theta_st phi_st])

    %%% RPE parameters
    c0=c0.';
    G_boresight(ij)=max(max(RPE))+10*log10(sum(Lsub(find(c0~=0))));
    G0(ij)=10*log10 ((max(abs(KerFF_sub*c0)))^2/(sum(Lsub.*abs(c0.').^2)));         %[dB] % clustering with optimization
    SL_maxpointing(ij)=G_boresight(ij)-FF_I_dB(Iele_max,Iazi_max) % scan loss in the maximum RPE beam
    SL_theta_phi(ij)=G_boresight(ij)-FF_I_dB(Iele,Iazi)            % scan loss in the desired direction
    SLL_threshold(ij)=G_boresight(ij)-SLL_level;
    [SLL_in, SLL_out]=SLL_in_out(FF_I_dB,ele,azi,elem,azim,AZI,ELE,Iazi,Iele_max,Iazi_max)

    for ij=1:size(Phase_m,1)
        legend([label_line ' \phi=' num2str(azi0) '◆'],'Mask')
        title('Vertical plane')
        subplot(1,2,2)
        plot(azi,FF_I_dB(Iele,:),col(ij,:),'Linewidth',2)
        hold on
        %         plot(azi,(Mask_EA (Iele,:)),'g','Linewidth',2)
        axis([-90,90,-30,max(max(Mask_EA(Iele,:)))+0.5]);grid
        xlabel('\phi');
        ylabel('RPE R(\theta,\phi)');
        legend('RPE','Mask')
        title('Horizontal plane')
        legend([label_line ' \theta=' num2str(ele0) ' deg'],'Mask')
        if save_data
            cd(save_folder)
            saveas(gcf,['cut_RPE_v' num2str(ele0) 'h' num2str(azi0) '.fig'])
        end


        if save_data
            cd(save_folder)
            saveas(gcf,['3D_RPE_v' num2str(ele0) 'h' num2str(azi0) '.fig'])
        end

    end

    str{1}=['optimal phase'];
    str{2}=['quantize phase'];
    str{3}=['\Delta\theta=' num2str(error_deg(1)) '◆'];
    str{4}=['\Delta\theta=' num2str(error_deg(2)) '◆'];
    str{5}=['\Delta\theta=' num2str(error_deg(3)) '◆'];
    subplot(1,2,1)
    plot(ele,(Mask_EA(:,Iazi)),'g','Linewidth',2)
    legend(str)
    subplot(1,2,2)
    plot(azi,(Mask_EA (Iele,:)),'g','Linewidth',2)
    legend(str)

    figure
    plot(phase_error_all')

    figure
    plot(rad2deg(angle(Phase_m)),'-sq')

    figure
    for ik=1:size(error_deg,2)
        subplot(size(error_deg,2),1,ik)
        errorbar(1:Ntrans,rad2deg(angle(Phase_m(2,:))),phase_error_all(ik+1,:),'r')
        hold on;grid
        plot(1:Ntrans,rad2deg(angle(Phase_m(2,:))))
        xlabel('i^{th} phase element')
        ylabel('Phase value [deg]')
        legend('phase error','phase value')
        axis([1 Ntrans -200 200])
        %         title(['Gaussian error distribution with error=+/- ' num2str(error_deg(ik)) ])
        if function_distr==1 % normal distribution
            title(['Normal distr. \Delta\theta=' num2str(error_deg(ik)) '◆']);
        elseif function_distr==2 % uniform distribution
            title(['Uniform distr. \Delta\theta=' num2str(error_deg(ik)) '◆']);
        end
    end

    if save_data
        diary off
    end
end