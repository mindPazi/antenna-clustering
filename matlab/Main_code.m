% ANTENNA ARRAY REGULAR CLUSTERING
% Version: 1.0;
% Developed by Milan RC. delivered on 06/30/2017.

clc
clear all
close all
%__________________________________________________________________________
%%% INPUT:

% ANTENNA ARRAY PARAMETERS
f=29e9;         % Frequency [GHz]

Nz=8;           % Number of rows
Ny=8;           % Number of columns

dist_z=0.7;     % antenna distance on z axis [times lambda]
dist_y=0.5;     % antenna distance on y axis [times lambda]

azi0=0;         % [deg] azimuth steering angle
ele0=10;        % [deg] elevation steering angle

%%% SINGLE ELEMENT RPE:
P=1;            % Set P=0 to achieve an isotropic element pattern, set P=1 for
% cosine element pattern
load_file=0;    % Set load_file =1 to load antenna element RPE from HFSS,
% set load_file =o to generate isotropic pattern
file_name = 'RPE_element.csv'; % name of the file with antenna element RPE from HFSS

%%% SELECT CLUSTER TYPE: deselect the one you need
% B=[0,0]; % single element cluster / NO clastering solution
%%% cluster size: 2 antenna elements
B=[0,0;0,1]; % vertical linear cluster
% B=[0,0;1,0]; % horizontal linear cluster
%%% cluster size: 3 antenna elements
% B=[0,0;0,1;0,2]; % vertical linear cluster
% B=[0,0;1,0;2,0]; % horizontal linear cluster
%%% cluster size: 4 antenna elements
% B=[0,0;0,1;0,2;0,3]; % vertical linear cluster
% B=[0,0;1,0;2,0;3,0]; % horizontal linear cluster

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LATTICE PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LATTICE SELECTION - select the type of lattice
scale=3e8/f*1000;  % [mm]
lambda=3e8/f;      % [m]
beta=2*pi/lambda;

%%% Rectangular grid
dz=dist_z*lambda;
dy=dist_y*lambda;
x1=[dy,0];
x2=[0,dz];

% Generate the basic lattice of grid-points
[Y,Z,NN,MM,Dy,Dz,ArrayMask]=GenerateLattice(Ny,Nz,x1,x2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
POLAR COORDINATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AZIMUT AND ELEVATION SAMPLING (for plots)
dele=.5; % angle resolution [deg]
dazi=.5; % angle resolution [deg]
ele=-90:dele:90;
azi=-90:dazi:90;
[AZI,ELE]=meshgrid(azi,ele);
WW=beta*cosd(90-ELE);
VV=beta*sind(90-ELE).*sind(AZI);
Nw=size(WW,2);
Nv=size(VV,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%                 CLUSTER EVALUATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAPPING ALGORITHM - Sub-array definition
Nel=Nz*Ny;          % number of array elements
ElementExc=ones(Nz,Ny); % Fixed array tapering [BFN]

[Cluster,Nsub] = SubArraySet_Generation(B,NN(:),MM(:));
for kk=0:size(Cluster,2)/2-1
    for l1=1:size(Cluster,1)
        Iy=Cluster(l1,2*kk+1)-min(NN(:))+1;
        Iz=Cluster(l1,2*kk+2)-min(MM(:))+1;
        Yc(l1,kk+1)=Y(Iz,Iy);
        Zc(l1,kk+1)=Z(Iz,Iy);
    end
end

Ntrans=size(Yc,2);
for kk=1:Ntrans
    Lsub(kk) = size(B,1);
    Zc_m(kk) = mean(Zc(1:Lsub(kk),kk)); % Phase center of sub-array
    Yc_m(kk) = mean(Yc(1:Lsub(kk),kk)); % Phase center of sub-array
end

figure
plot(Yc,Zc,'sq');grid
xlabel('y [m]')
ylabel('z [m]')
title('Antenna Sub-arrays')

%%% EXCITATIONS
v0=beta*sind(90-ele0)*sind(azi0);
w0=beta*cosd(90-ele0);
Phase_m=exp(-1i*(w0*Zc_m+v0*Yc_m)); % Clustered phase distribution
Amplit_m = ones(1,Ntrans)./Lsub;    % Clustered amplitude distribution (normalized to cluster size)
c0=Amplit_m.*Phase_m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RPE and ARRAY FACTOR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% ELEMENT FACTOR - single element radiation pattern
Fel=ElementPattern(P,AZI,ELE,load_file,file_name);
Fel_VW=interp2(AZI,ELE,Fel,AZI,ELE);

figure
contourf(azi,ele,20*log10(Fel_VW));
xlabel('azimuth \phi [deg]')
ylabel('elevation \theta [deg]')
title('Radiation Pattern R(\theta,\phi)');
c=colorbar;
ylabel(c,'[dB]')

%%% FAR FIELD TRANSFORMATION KERNELS
% sub-array kernel (Sub-Array Radiation pattern)
KerFF_sub=zeros(Nw*Nv,Ntrans); % one coefficient for every antenna element (but equal for each sub-array elements)
for kk=1:Ntrans
    for jj=1:Lsub(kk)
        KerFF_sub(:,kk)=KerFF_sub(:,kk)+exp(1i*(VV(:)*Yc(jj,kk)+WW(:)*Zc(jj,kk))).*Fel_VW(:);
    end
end
FF=KerFF_sub*c0.';
FF_norm=FF./max((FF(:)));
FF_norm_2D=reshape(FF_norm,Nv,Nw);
Fopt_dB=20*log10(abs(FF_norm_2D)); %RPE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure plotting Normalized Radiation Pattern in the visible range
figure
contourf(AZI,ELE,Fopt_dB,[0:-2:-50])
xlabel('azimuth \phi [deg]')
ylabel('elevation \theta [deg]')
title('Normalized Radiation Pattern R(\theta,\phi)');
c=colorbar;
ylabel(c,'[dB]')

% Figure plotting cardinal planes FF cuts
Iele=find((ele-ele0)>=0,1);
Iazi=find((azi-azi0)>=0,1);
figure
subplot(2,1,1)
plot(ele,Fopt_dB(:,Iazi),'b','Linewidth',2); hold on;
plot(ele(Iele),0:-5:-50,'+r')
axis([-90,90,-50,0]);grid
xlabel('\theta');
ylabel('Normalized RPE R(\theta,\phi) [dB]');
legend('RPE regular clustered antenna','steering angle')
title('Vertical plane')
subplot(2,1,2)
plot(azi,Fopt_dB(Iele,:),'b','Linewidth',2); hold on;
plot(azi(Iazi),0:-5:-50,'+r')
axis([-90,90,-50,0]);grid
xlabel('\phi');
ylabel('Normalized RPE R(\theta,\phi) [dB]');
legend('RPE regular clustered antenna','steering angle')
title('Horizontal plane')
