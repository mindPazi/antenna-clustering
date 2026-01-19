function [cdf_I,x_I,N,M]=Statistics_GL_SLL(LAT,SYS,CL,EEF,ele_pp,azi_pp,flag_dummy,thin_array,thin_ratio,rp);
disp('*** STATISTICS ***')

N=[];
M=[];
for i_e=1:size(ele_pp,2)

    for i_a=1:size(azi_pp,2)
        %          if ele_pp(i_e)==0 & azi_pp(i_a)==0
        %               keyboard
        %          end
        disp(['Elaboration: \theta =' num2str(ele_pp(i_e)) ' [◆ - \phi =' num2str(azi_pp(i_a)) ' [◆]'])
        SYS.azi0=azi_pp(i_a);
        SYS.ele0=ele_pp(i_e);
        [GCS]= PolarCoordinate_SteeringAngle(SYS,LAT);
        %%% EXCITATIONS
        CL=coefficient_evaluation_dummy_new_thin(GCS,CL,LAT,flag_dummy,thin_array,thin_ratio,rp);
        Iele_max=find((GCS.ele-SYS.ele0)>=0,1);
        Iazi_max=find((GCS.azi-SYS.azi0)>=0,1);
        [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE_cluster(GCS.Nw, GCS.Nv, CL.Lsub, CL.Ac, GCS.W, GCS.WW, GCS.Wae, GCS.WWae, CL.Yc_m, CL.Zc_m, CL.c0, EEF.Fel_VW_cl,CL.flipped);

        % [FF_norm_dB, FF_I_dB, KerFF_sub, FF_norm]=Kernel1_RPE(GCS.Nw, GCS.Nv, CL.Lsub, CL.Ac, GCS.W, GCS.WW, GCS.Wae, GCS.WWae, CL.Yc, CL.Zc, CL.c0, EEF.Fel_VW, LAT.Nel);

        %          [FF_I_dB_norm]=interferer_signal(FF_I_dB_norm,Iele_max,Iazi_max);
        %          noi=FF_I_dB_norm;
        %          N=[N noi(:)];

        FF_I_dB_norm=FF_I_dB-EEF.G_boresight;
        [FF_I_dB_norm]=interferer_signal(FF_I_dB_norm,Iele_max,Iazi_max);
        %          noi=FF_I_dB_norm;
        %          N=[N noi(:)];

        Iele=find((GCS.ele-SYS.ele0)>=0,1);
        Iazi=find((GCS.azi-SYS.azi0)>=0,1);
        [SLI_in,SLI_out]=interferer_signal_insector(FF_I_dB_norm,GCS.ele,GCS.azi, SYS.ele0, SYS.azi0,GCS.AZI,GCS.ELE,Iazi,Iele_max,Iazi_max);

        N=[N SLI_in'];
        M=[M SLI_out'];
        %%%%%

    end
end
[x_cdf y_cdf]=cdf_plot_v2d0(N(:));
cdf_I{1}=y_cdf;
x_I{1}=x_cdf;

[x_cdf y_cdf]=cdf_plot_v2d0(M(:));
cdf_I{2}=y_cdf;
x_I{2}=x_cdf;