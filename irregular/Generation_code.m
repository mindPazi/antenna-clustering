% Generation_code.m
%
% This code is used to generate REGULAR/IRREGULAR clustering for a
% rectangular array. The user can set different kind of parameters
% according to the antenna topology (number of rows, column, grid shape),
% select the single antenna element radiation pattern, the mask
% characteristics and the cluster type.
%
% Version: 1.0;
% Develloped by Laura Resteghini (Milan RC.) delivered on 15/06/2018.
% Version: 2.0;
% Develloped by Laura Resteghini (Milan RC.) delivered on 08/11/2019.
%___________________________________________________________________________
%%CLEANING OF WORKSPACE, GLOBAL VARIABLES AND GENERAL PARAMETERS
clc;clear all;close all
tic
actual=cd;

%%% LOAD INPUT Parameters
% Input_Conf     EEF.RPE(find(GCS.AZI>-25 & GCS.AZI<25 & GCS.ELE>-40 & GCS.ELE<-25))==-1;
% Input_Conf_sub6GHz
% Input_Conf_2bits_PS
% Input_Conf_AiP_macro
% Input_Conf_AiP_macro_4dB
% Input_Conf_AiP_macro_subtle_v2
% Input_Conf_AiP_macro_rotation
% Input_Conf_AiP_macro_fishbone
% Input_Conf_AiP_macro_fabio2
% Input_Conf_AiP_macro_EEF % original it works
% Input_Conf_AiP_macro_EEF_gap
% Input_Conf_AiP_macro_fishbone_2 %works!
% Input_Conf_AiP_macro_fishbone_2_new
% Input_Conf_AiP_macro_thin_array
% Input_Conf_AiP_macro_irregular_CHI
% Input_Conf_AiP_macro_irregular_20B
% Input_Conf_AiP_macro_irregular_U6G_24x20
% Input_Conf_AiP_macro_16x16_75dBm
Input_Conf_AiP_macro_16x16_75dBm_Regular
% Input_Conf_AiP_macro_16x16_75dBm_horizontal

% Input_Conf_AiP_macro_FEF_prova
% theta_shift=-7;
% LAT.dist_z=0.617*2;

% Input_Conf_simple

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  LATTICE AND CLUSTER EVALUATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LATTICE SELECTION - select the type of lattice
SYS.lambda=3e8/SYS.f; % [m]
SYS.beta=2*pi/SYS.lambda;

% Generate the basic lattice of grid-points
[LAT] = Lattice_Definition(LAT,SYS);
[LAT] = GenerateLattice(LAT);
[LAT] = Lattice_modification_v2d0(LAT);

%%% ALGORITHM X SECTION - Sub-array definition  [B0 defines the basic structure of the sub-array]
for bb=1:size(CL.Cluster_type,2)
    [S, Nsub] = FullSubarraySet_Generation (CL.Cluster_type{bb},LAT,CL.rotation_cluster);
    CL.S_all{bb}=S;
    N_all(bb)=Nsub;
    L(bb)=size(S,1);
end

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  POLAR COORDINATE AND STEERING VECTOR  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[GCS]= PolarCoordinate_SteeringAngle(SYS,LAT);

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  RPE and ARRAY FACTOR  %%%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %/% ELEMENT FACTOR - single element radiation pattern

if EEF_mean==4
    for e_el=1:LAT.Nz
        FLAG.rpe_folder=rpe_folder;
        FLAG.rpe_file_name=rpe_file_name{seq(e_el)};
        [EEF] = ElementPattern_v4d0(EEF,LAT,GCS,FLAG);
        %
        EEF.Fel_VW=ones(size(EEF.Fel_VW));
        Fel_VW_all{e_el}=EEF.Fel_VW;
        max_RPE=20*log10(max(max(EEF.Fel_VW)));
    end
    G_boresight=mean(max_RPE)+10*log10(LAT.Nel);
elseif EEF_mean==1
    [RPE,Fel_VW] = EEF_clustered(1,2,theta_shift,0,SYS.f,CL.Cluster_type{1},LAT.dist_z/2,LAT.dist_y,GCS.AZi,GCS.ELi,GCS.WW,GCS.WWae);
    EEF.Fel_VW=Fel_VW;
    EEF.RPE_ele_max=max(max(RPE));
    EEF.RPE=RPE;
    for e_el=1:LAT.Nz
        Fel_VW_all{e_el}=EEF.Fel_VW;
    end
else
    [EEF] = ElementPattern_v4d0(EEF,LAT,GCS,FLAG);
    for e_el=1:LAT.Nz
        Fel_VW_all{e_el}=EEF.Fel_VW;
    end
end
if LAT.Ny>1
    for iy=1:LAT.Ny
        for iz=1:LAT.Nz
            Fel_VW_all{(iy-1)*LAT.Nz+iz}=Fel_VW_all{iz};
        end
    end
end

if LAT.Ny>1
end
end
% Fel_VW_all=Fel_VW_all*ones(1,LAT.Ny);

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%  MASK EVALUATION  %%%%
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Iele=find((GCS.ele-SYS.ele0)>=0,1);
Iazi=find((GCS.azi-SYS.azi0)>=0,1);
[MASK]=mask_design_v2d0(MASK,LAT,GCS,SYS,EEF);
EEF.G_boresight=max(max(EEF.RPE))+10*log10(LAT.Nel);

sss=0;
for ij_cont=1:Niter
    ij_cont
    if ij_cont/100==round(ij_cont/100)
        disp(['Number of iteration:' num2str(ij_cont)])
        disp(['Number of generated solutions:' num2str(size(simulation,1))])
        hist(all_Ntrans)
    end
else
    CL.C_ori=[];
end
[CL.Yc,CL.Zc,CL.Ac]= Index2Position_cluster_v2d0(CL.Cluster,LAT.Y,LAT.Z,ElementExc,LAT.NN,LAT.MM);  % Sub-array partiti

%    LAT.Y(13:end,:)=-LAT.Y(13:end,:);
%    LAT.NN(13:end,:)=-LAT.NN(13:end,:);
%    CL.Yc(find(CL.Zc=0))==-CL.Yc(find(CL.Zc=0));%improve SLL reduction in h -2*SYS.lambda;
%    CL.Zc(find(CL.Zc<=0))=CL.Zc(find(CL.Zc<=0))-SYS.lambda;

CL.Ntrans=size(CL.Yc,2);
CL.Lsub=NaN*zeros(1,CL.Ntrans);
CL.Zc_m=NaN*zeros(1,CL.Ntrans);
CL.Yc_m=NaN*zeros(1,CL.Ntrans);
for kk=1:CL.Ntrans
    CL.Lsub(kk) = size(CL.Cluster{kk},1);
    CL.Zc_m(kk) = mean(CL.Zc(1:CL.Lsub(kk),kk));  % Phase center of sub-array
    CL.Yc_m(kk) = mean(CL.Yc(1:CL.Lsub(kk),kk));  % Phase center of sub-array
end
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% EXCITATIONS
CL=coefficient_evaluation(GCS,CL);
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

round(rad2deg(angle(CL.c0)),2)
diff(ans)

% keyboard
if thin_array
    CL.thin_array=thin_array;
    CL.thin_ratio=thin_ratio;
    CL.c0(CL.rp)=0;
else
    CL.thin_array=false;
end

%    if thin_array
%        CL.thin_array=thin_array;
%        CL.thin_ratio=thin_ratio;
%        CL.rp=randperm(size(CL.c0,2));
%        CL.c0(CL.rp(1:CL.thin_ratio))==0;
%    else
%        CL.thin_array=false;
%    end

%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cd 'C:\Users\100400253\Desktop\SW_ANTENNA_SYNTHESIS\IRREGULAR_ANTENNA_CLUSTERING_TOOL_SW_v3d0\Generate_HFSS_coefficients_file\
% if size(CL.Cluster_type{1},1) % if is clustered
%    phase=reshape(round(rad2deg(angle(CL.c0)),2),[LAT.Nz/2,LAT.Ny]);
%    ampl=ones([LAT.Nz,LAT.Ny]);
%    phasew=[];
%    for ij=1:LAT.Nz/2
%        phasew=[phasew;phase(ij,:).*ones(2,1)];
%    end
%    phase=phasew;
%    save('Coefficients.dat','phase','ampl')
%    else
%        phase=reshape(round(rad2deg(angle(CL.c0)),2),[LAT.Nz,LAT.Ny]);
%        ampl=ones([LAT.Nz,LAT.Ny]);
%    end
% save('Coefficients.dat','phase','ampl')
%%% FAR FIELD TRANSFORMATION KERNELS
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if EEF_mean==4
    [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE_diffrent(GCS.Nw, GCS.Nv, CL.Lsub, CL.Ac, GCS.WV, GCS.WW, GCS.WWae,
else
    [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(GCS.Nw, GCS.Nv, CL.Lsub, CL.Ac, GCS.WV, GCS.WW, GCS.WWae, GCS.WWae,
end
figure
contourf(FF_I_dB)

%%% POST PROCESSING %%%c
%%%%%/%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Cost function evaluation
%    Constr=FF_I_dB-MASK.Mask_EA;        % SLL constraint
%    Cm=sum(sum(Constr>0));
%    if Cm<Cost_thr %& Ntrans==64
%        righe=[];
%        sy=0;
%        for ic=1:size(CL.C_ori,2)
%            sy(ic+1)=size(CL.C_ori{ic},1);
%            righe=[righe selez_riga{ic}+sum(sy(1:ic))];
%        end
%        ttt=zeros(1,sum(sy));
%        ttt(righe)=1;
%    sss=sss+1;
%    %%%% [selected rows; cost function; number of selected clusters; number of antenna elements]
%    simulation(sss,:)=[ttt Cm size(righe,2) sum(CL.Lsub)];  %
%
%    end
%    all_Nel(ij_cont)= sum(CL.Lsub);
%    clear Zc_m Yc_m CL.Lsub

%    all_Cm(ij_cont)=Cm;
%    all_Ntrans(ij_cont)=CL.Ntrans;
end

% disp('*** Results:')
% disp(['Number of antenna elements:' num2str(all_Nel)])
% disp(['Number of TRx chains:' num2str(all_Ntrans)])
% disp(['Clustering factor: 1:' num2str(all_Nel./all_Ntrans)])

if FLAG.save_data
    C_ori=CL.C_ori;
    Smod=CL.S_all;
    B=CL.Cluster_type;
    cd(FLAG.folder_name)
    save(FLAG.file_name,'LAT','C_ori','Smod','B')
    save(['solution_clusters_' FLAG.file_name], 'simulation')

    simulationBF.f=SYS.f;
    %    simulationBF.Nz=LAT.Nz;
    %    simulationBF.Ny=LAT.Ny;
    %    simulationBF.dist_z=LAT.dist_z;
    %    simulationBF.dist_y=LAT.dist_y;
    %    simulationBF.x1=LAT.x1;
    %    simulationBF.x2=LAT.x2;
    simulationBF.LAT=LAT;
    simulationBF.Smod=CL.S_all;
    simulationBF.C_ori=CL.C_ori;

    cd(FLAG.folder_name)
    save(FLAG.file_name,'simulationBF')
    save(['selected_cluster_' FLAG.file_name],'B')
    save(['solution_clusters_' FLAG.file_name], 'simulation')

end

if FLAG.make_plot
    figure
    subplot(1,3,1)
    plot(simulation(:,end-2))
    ylabel('cost function')
    subplot(1,3,2)
    plot(simulation(:,end-1))
    ylabel('number of selected clusters')
    subplot(1,3,3)
    plot(simulation(:,end))
    ylabel('number of antenna elements')

    Narray=simulation(:,end-1);
    fcost=simulation(:,end-2);

    figure
    plot(Narray,fcost,'x');grid
    xlabel('N® clusters');ylabel('N® points exceeding mask')

    x=unique(Narray);
    for il=1:size(x,1)
        q=find(Narray==x(il));
        err(il)=std(fcost(q));
        mm(il)=mean(fcost(q));
    end
    hold on;
    errorbar(x,mm,err,'r')

    figure
    hist(all_Ntrans);grid
    xlabel('Type of TRx chains ')
    ylabel('number of selected TRx chains')

end
used_time_secs=toc
% figure
% plot(GCS.azi,FF_I_dB(Iele,:),'b','LineWidth',2)
% xlabel('\phi');
% ylabel('Normalized Radiation Pattern R(\theta,\phi)');
% legend('WITHOUT tapering','WITH tapering','Mask')
% title('Elevation plane')
% % hold on
% % plot(GCS.azi,10*log10(MASK.Mask_EA (Iele,:)),'g','LineWidth',2)
% % axis([-90,90,-50,0]);grid
% % figure
% plot(GCS.azi,FF_I_dB(:,Iazi),'b','LineWidth',2)
% xlabel('\phi');
% ylabel('Normalized Radiation Pattern R(\theta,\phi)');
% legend('WITHOUT tapering','WITH tapering','Mask')
% title('Elevation plane')

ele_pp=-20:20:20;%-20:5:020;%:20;%0:30:30;
azi_pp=-60:60:60;%-60;%-60:%-10:60;0:10:10;%0:10:10%;
% % ele_pp=[-20:5:20];
% % azi_pp=-60:10:60;
% [FF_I_dB_all,codebook_FF_h,codebook_FF_v,azi0,ele0,SLL_in_all,SLL_out_all,SL]=Codebook_2D_RPE_v2d0(LAT,SYS,CL,EEF,MASK,ele_pp,a
% RPE_norm=EEF.RPE-max(max(EEF.RPE));
% hold on
% plot(GCS.ele,RPE_norm(:,181),'LineWidth',2)
if CL.thin_array
    figure
    plot(CL.Yc_m,CL.Zc_m,'sb');hold on;
    plot(CL.Yc_m(CL.rp),CL.Zc_m(CL.rp),'sr');grid
end
% % zzz=[1 2 5 6 7 8 11 12 85 86 89 90 91 92 95 96];
% % figure
% % plot(CL.Yc_m,CL.Zc_m,'sb');hold on;
% % plot(CL.Yc_m(CL.rp),CL.Zc_m(CL.rp),'sr');grid

% CL.thin_array=false;
% [FF_I_dB_all,codebook_FF_h,codebook_FF_v,azi0,ele0,SLL_in_all,SLL_out_all,SL]=Codebook_2D_RPE_v2d0(LAT,SYS,CL,EEF,MASK,ele_pp,a

% close all
% [FF_I_dB_all,codebook_FF_h,codebook_FF_v,azi0,ele0,SLL_in_all,SLL_out_all,SL_h_all,SL_v_all,SL]=Codebook_2D_RPE_v3d0(LAT,SYS,

hi)');
)

e0,SLL_in_all,SLL_out_all,SL]=Codebook_2D_RPE_v2d0(LAT,SYS,CL,EEF,MASK,ele_pp,azi_pp);

e0,SLL_in_all,SLL_out_all,SL]=Codebook_2D_RPE_v2d0(LAT,SYS,CL,EEF,MASK,ele_pp,azi_pp);

e0,SLL_in_all,SLL_out_all,SL_h_all,SL_v_all,SL]=Codebook_2D_RPE_v3d0(LAT,SYS,CL,EEF,MASK,ele_pp,azi_pp);

azi_pp=-20:20:20;%-20:5:020;%:20;%0:30:30;
ele_pp=-75:5:10;%-60:5:-55;%60;%:10:60;
% cd 'C:\Users\100400253\Desktop\SW_ANTENNA_SYNTHESIS\IRREGULAR_ANTENNA_CLUSTERING_TOOL_SW_v3d0\AiP_pole&macro\AiP_pole\Tilted_FF
Patch_PP