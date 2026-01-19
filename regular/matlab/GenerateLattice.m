function [Y,Z,NN,MM,DY,DZ,Mask,I]=GenerateLattice (Ny,Nz,x1,x2,A,B)
% Version: 1.0;
% Developed by Milan RC. delivered on 06/30/2017.

% This function calculates a regular lattice based on the generation
% vectors x1 and x2. In particular, the function assumes that y-axis
% referes to the horizontal plane while z-axis refers to the vertical
% plane: then x1 is assumed to be aligned with the y-axis, while x2 can any
% other vector of the yz-plane different from x1.
%
% Please NOTE:
%
% - RECTANGULAR LATTICE for a regular rectangular lattice x1 and x2 must be
%   orthogonal. Example x1=[1,0] and x2=[0,1];
%
% - TRIANGULAR LATTICE for a triangular atice x1 and x2 must be
%   45◆ inclined. Example x1=[1,0] and x2=[1,1] for a 45◆ lattice
%
% - HEXAGONAL LATTICE. Example x1=[1,0] and x2=[1,1] for a 30◆ lattice
%
%
%%% INPUT:
% Nz: number of rows [Scalar];
% Ny: number of columns [Scalar];
% x1,x2: distance between elements is z and y coordinates according to
%   lattice type [Scalar];
% A and B: refers to the truncation of the aperture to an ellipsoid
% A refers to y-axis while B to z-axis.
%
%
%%% OUTPUT:
% Y: matrix [Nz x Ny] with element y coordinates;
% Z: matrix [Nz x Ny] with element z coordinates;
% NN: matrix [Nz x Ny] with element index in y coordinates;
% MM: matrix [Nz x Ny] with element index in z coordinates;
% DY: maximum size of array in y in mm [Scalar];
% DZ: maximum size of array in z in mm [Scalar];
% Mask: truncation mask;
% I: truncation index;
%

if nargin==4
    A=Inf;
    B=Inf;
elseif nargin==5
    B=A;
end

% Generate array indexes
if (rem(Nz,2))
    M=-(Nz-1)/2:(Nz-1)/2;
else
    M=-Nz/2+1:Nz/2;
end

if (rem(Ny,2))
    N=-(Ny-1)/2:(Ny-1)/2;
else
    N=-Ny/2+1:Ny/2;
end

[NN,MM]=meshgrid(N,M);

dz=x2(2);
dy=x1(1);
DELTA=max(x2(1),x1(2));

Y=NN*dy;
Z=MM*dz;

Y(2:2:end,:)=Y(2:2:end,:)+DELTA;

DZ=(max(Z(:))-min(Z(:)));
DY=(max(Y(:))-min(Y(:)));

%%% Aperture truncation

I=find((Y./A).^2+(Z./B).^2>1);
Mask=ones(size(Y));
Mask(I)=0;

end
